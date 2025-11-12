import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

try:
    from liger_kernel.transformers import LigerSwiGLUMLP as MLP
except ImportError:
    from .mlp import MLP

try:
    from liger_kernel.transformers import LigerRMSNorm as RMSNorm
except ImportError:
    from torch.nn import RMSNorm


class MultiheadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, num_heads=8, attn_qk_norm=True):
        super().__init__()
        assert q_dim % num_heads == 0, "q_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.W_q = nn.Linear(q_dim, q_dim, bias=False)
        self.W_k = nn.Linear(k_dim, q_dim, bias=False)
        self.W_v = nn.Linear(v_dim, q_dim, bias=False)
        self.W_o = nn.Linear(q_dim, q_dim, bias=False)

        if attn_qk_norm:
            self.q_norm = RMSNorm(q_dim)
            self.k_norm = RMSNorm(q_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def _expand_mask(self, attn_mask):
        S = attn_mask.shape[1]
        return attn_mask[:, None, None, :].expand(-1, self.num_heads, S, -1)

    def _rearrange(self, q, k, v):
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim)
        return q, k, v

    def forward(self, x_q, x_k, x_v, attn_mask=None):
        q = self.W_q(x_q)
        k = self.W_k(x_k)
        v = self.W_v(x_v)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k, v = self._rearrange(q, k, v)

        attn_mask_expanded = (
            self._expand_mask(attn_mask.to(torch.bool)) if attn_mask is not None else None
        )

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=attn_mask_expanded,
        )

        x = rearrange(x, "b h t d -> b t (h d)", h=self.num_heads, d=self.head_dim)
        return self.W_o(x)


class GroupedQueryAttention(MultiheadAttention):
    def __init__(self, q_dim, k_dim, v_dim, num_heads=8, num_kv_heads=4):
        super().__init__(q_dim, k_dim, v_dim, num_heads)
        assert (
            num_heads % num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"
        self.num_kv_heads = num_kv_heads
        self.q_per_kv = num_heads // num_kv_heads
        self.W_k = nn.Linear(k_dim, self.head_dim * num_kv_heads, bias=False)
        self.W_v = nn.Linear(v_dim, self.head_dim * num_kv_heads, bias=False)

        if self.k_norm is not None:
            self.k_norm = RMSNorm(self.head_dim * num_kv_heads)

    def _rearrange(self, q, k, v):
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim)
        k = rearrange(
            k, "b t (h d) -> b h t d", h=self.num_kv_heads, d=self.head_dim
        ).repeat_interleave(self.q_per_kv, dim=1)
        v = rearrange(
            v, "b t (h d) -> b h t d", h=self.num_kv_heads, d=self.head_dim
        ).repeat_interleave(self.q_per_kv, dim=1)
        return q, k, v


class BiEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config)
        self.attn_norm = RMSNorm(config.hidden_size)
        self.mlp_norm = RMSNorm(config.hidden_size)

        k_dim = (
            config.backbone_model_hidden_size
            if config.parallel_encoder
            else config.hidden_size
        )

        v_dim = (
            config.backbone_model_hidden_size
            if config.parallel_encoder
            else config.hidden_size
        )

        self.attn = (
            MultiheadAttention(
                q_dim=config.hidden_size,
                k_dim=k_dim,
                v_dim=v_dim,
                num_heads=config.num_attn_heads,
            )
            if not config.use_gqa
            else GroupedQueryAttention(
                q_dim=config.hidden_size,
                k_dim=k_dim,
                v_dim=v_dim,
                num_heads=config.num_attn_heads,
                num_kv_heads=config.num_kv_attn_heads,
            )
        )

    def _attn_forward(self, x_q, x_k=None, x_v=None, attn_mask=None):
        residual = x_q
        x_q = self.attn_norm(x_q)

        assert (x_k is None) == (
            x_v is None
        ), "x_k and x_v must be both None or both not None"

        if x_k is None and x_v is None:
            x_k = x_q
            x_v = x_q

        return residual + self.attn(x_q, x_k, x_v, attn_mask=attn_mask)

    def _mlp_forward(self, x):
        return x + self.mlp(self.mlp_norm(x))

    def forward(self, x_q, x_k=None, x_v=None, attn_mask=None):
        return self._mlp_forward(self._attn_forward(x_q, x_k, x_v, attn_mask=attn_mask))


class BiEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.in_proj = (
            nn.Sequential(
                RMSNorm(config.backbone_model_hidden_size),
                nn.Linear(
                    config.backbone_model_hidden_size, config.hidden_size, bias=False
                ),
            )
            if config.backbone_model_hidden_size != config.hidden_size
            else None
        )
        self.out_proj = (
            nn.Sequential(
                RMSNorm(config.hidden_size),
                nn.Linear(
                    config.hidden_size, config.backbone_model_hidden_size, bias=False
                ),
            )
            if config.backbone_model_hidden_size != config.hidden_size
            else None
        )

        self.layers = nn.ModuleList(
            [BiEncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, x, decoder_outputs=None, attn_mask=None):
        if self.in_proj is not None:
            x = self.in_proj(x)

        if self.config.parallel_encoder:
            assert (
                decoder_outputs is not None
            ), "decoder_outputs must be provided for parallel encoder"
            assert len(decoder_outputs) == len(
                self.layers
            ), "number of decoder outputs must match number of layers"
        else:
            assert (
                decoder_outputs is None
            ), "decoder_outputs should not be provided for non-parallel encoder"
            decoder_outputs = [None] * len(self.layers)

        for layer, d_out in zip(self.layers, decoder_outputs):
            x_kv = d_out if self.config.parallel_encoder else x
            x = layer(x, x_kv, x_kv, attn_mask=attn_mask)

        if self.out_proj is not None:
            x = self.out_proj(x)

        return x
