import torch.nn as nn
from .attention import MultiheadAttention, GroupedQueryAttention

try:
    from liger_kernel.transformers import LigerSwiGLUMLP as MLP
except ImportError:
    from .mlp import MLP

try:
    from liger_kernel.transformers import LigerRMSNorm as RMSNorm
except ImportError:
    from torch.nn import RMSNorm


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

    def forward(self, x_q, x_k=None, x_v=None, attn_mask=None):
        residual = x_q
        x_q = self.attn_norm(x_q)

        assert (x_k is None) == (
            x_v is None
        ), "x_k and x_v must be both None or both not None"

        if x_k is None and x_v is None:
            x_k = x_q
            x_v = x_q

        x = residual + self.attn(x_q, x_k, x_v, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


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
