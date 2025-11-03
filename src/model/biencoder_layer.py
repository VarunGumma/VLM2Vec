import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from .moe import MLP, SparseMoE


class MultiHeadAttention(nn.Module):
    def __init__(
        self, q_dim: int, k_dim: int = None, v_dim: int = None, n_heads: int = 8
    ):
        super().__init__()
        self.n_heads = n_heads
        k_dim = k_dim if k_dim is not None else q_dim
        v_dim = v_dim if v_dim is not None else q_dim

        self.W_q = nn.Linear(q_dim, q_dim, bias=False)
        self.W_k = nn.Linear(k_dim, q_dim, bias=False)
        self.W_v = nn.Linear(v_dim, q_dim, bias=False)
        self.W_o = nn.Linear(q_dim, q_dim, bias=False)

    def forward(self, x_q, x_k, x_v, attn_mask=None):
        q = rearrange(self.W_q(x_q), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(self.W_k(x_k), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.W_v(x_v), "b s (h d) -> b h s d", h=self.n_heads)

        with torch.backends.cuda.sdpa_kernel(enable_flash=True):
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=0.0,
                is_causal=False,
                need_weights=False,
                attn_mask=attn_mask,
            )

        x = rearrange(x, "b h s d -> b s (h d)", h=self.n_heads)
        return self.W_o(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int = None,
        v_dim: int = None,
        n_heads: int = 8,
        n_kv_heads: int = 4,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.kv_heads = n_kv_heads

        assert (
            n_heads % n_kv_heads == 0
        ), "Number of query heads must be divisible by number of key/value heads"

        k_dim = k_dim if k_dim is not None else q_dim
        v_dim = v_dim if v_dim is not None else q_dim

        self.W_q = nn.Linear(q_dim, q_dim, bias=False)
        self.W_k = nn.Linear(k_dim, self.kv_heads * (q_dim // n_heads), bias=False)
        self.W_v = nn.Linear(v_dim, self.kv_heads * (q_dim // n_heads), bias=False)
        self.W_o = nn.Linear(q_dim, q_dim, bias=False)

    def forward(self, x_q, x_k, x_v, attn_mask=None):
        q = rearrange(self.W_q(x_q), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(
            self.W_k(x_k), "b s (h d) -> b h s d", h=self.kv_heads
        ).repeat_interleave(self.group_size, dim=1)
        v = rearrange(
            self.W_v(x_v), "b s (h d) -> b h s d", h=self.kv_heads
        ).repeat_interleave(self.group_size, dim=1)

        with torch.backends.cuda.sdpa_kernel(enable_flash=True):
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=0.0,
                is_causal=False,
                need_weights=False,
                attn_mask=attn_mask,
            )

        x = rearrange(x, "b h s d -> b s (h d)", h=self.n_heads)
        return self.W_o(x)


class BiEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config)
        self.attn_norm = nn.RMSNorm(config.hidden_size)
        self.mlp_norm = nn.RMSNorm(config.hidden_size)

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
            GroupedQueryAttention(
                q_dim=config.hidden_size,
                n_heads=config.num_heads,
                group_size=config.group_size,
                k_dim=k_dim,
                v_dim=v_dim,
            )
            if config.use_gqa
            else MultiHeadAttention(
                q_dim=config.hidden_size,
                n_heads=config.num_heads,
                k_dim=(
                    config.backbone_model_hidden_size
                    if config.parallel_encoder
                    else config.hidden_size
                ),
                v_dim=(
                    config.backbone_model_hidden_size
                    if config.parallel_encoder
                    else config.hidden_size
                ),
            )
        )

    def _attn_forward(self, x_q, x_k, x_v, attn_mask=None):
        return x_q + self.attn(
            self.attn_norm(x_q), x_k=x_k, x_v=x_v, attn_mask=attn_mask
        )

    def _mlp_forward(self, x):
        return x + self.mlp(self.mlp_norm(x))

    def forward(self, x_q, x_k, x_v, attn_mask=None):
        x = self._attn_forward(x_q, x_k=x_k, x_v=x_v, attn_mask=attn_mask)
        x = self._mlp_forward(x)
        return x


class BiEncoderLayerMoE(BiEncoderLayer):
    def __init__(self, config):
        super().__init__(config=config)
        self.mlp = SparseMoE(config)

    def _mlp_forward(self, x):
        residual = x
        x, router_logits = self.mlp(self.mlp_norm(x))
        x = x + residual
        return x, router_logits


class BiEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_in = (
            nn.Linear(config.backbone_model_hidden_size, config.hidden_size, bias=False)
            if config.backbone_model_hidden_size != config.hidden_size
            else None
        )
        self.W_out = (
            nn.Linear(config.hidden_size, config.backbone_model_hidden_size, bias=False)
            if config.backbone_model_hidden_size != config.hidden_size
            else None
        )

        self.layers = nn.ModuleList(
            [
                (
                    BiEncoderLayer(config)
                    if not config.use_moe
                    else BiEncoderLayerMoE(config)
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, x, decoder_outputs=None, attn_mask=None):
        if self.W_in is not None:
            x = self.W_in(x)

        if self.config.parallel_encoder:
            assert (
                decoder_outputs is not None
            ), "decoder_outputs must be provided for parallel encoder"
            assert len(decoder_outputs) == len(
                self.layers
            ), "number of decoder outputs must match number of layers"

        if self.config.use_moe:
            router_logits = []
            for layer, d_out in zip(self.layers, decoder_outputs):
                x_kv = d_out if self.config.parallel_encoder else x
                x, rl = layer(x, x_kv, x_kv, attn_mask=attn_mask)
                router_logits.append(rl)
        else:
            for layer, d_out in zip(self.layers, decoder_outputs):
                x_kv = d_out if self.config.parallel_encoder else x
                x = layer(x, x_kv, x_kv, attn_mask=attn_mask)

        if self.W_out is not None:
            x = self.W_out(x)

        return (x, router_logits) if self.config.use_moe else x
