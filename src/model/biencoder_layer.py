import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from .moe import MLP, SparseMoE


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        inter_dim: int = None,
        activation: str = "gelu",
        n_heads: int = 8,
        use_moe: bool = False,
        **moe_config
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.use_moe = use_moe
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.attn_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)

        if use_moe:
            self.mlp = SparseMoE(
                hidden_dim=dim,
                intermediate_size=inter_dim,
                activation=activation,
                **moe_config
            )
        else:
            self.mlp = MLP(
                hidden_size=dim, intermediate_size=inter_dim, activation=activation
            )

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.attn_norm(x)

        qkv = self.qkv_proj(x) * self.scale
        q, k, v = rearrange(qkv, "b s (k h d) -> k b h s d", k=3, h=self.n_heads)

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=attn_mask,
        )

        x = rearrange(x, "b h s d -> b s (h d)")
        x = self.o_proj(x) + residual

        residual = x
        x = self.mlp_norm(x)

        if self.use_moe:
            x, router_logits = self.mlp(x)
        else:
            x = self.mlp(x)

        x = x + residual
        return (x, router_logits) if self.use_moe else x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        inter_dim: int,
        n_heads: int,
        num_layers: int,
        use_moe: bool = False,
        **moe_config
    ):
        super().__init__()
        self.use_moe = use_moe
        self.proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=dim,
                    inter_dim=inter_dim,
                    n_heads=n_heads,
                    use_moe=use_moe,
                    **moe_config
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attn_mask=None):
        x = self.proj(x)

        if self.use_moe:
            router_logits = []
            for layer in self.layers:
                x, router_logits_ = layer(x, attn_mask=attn_mask)
                router_logits.append(router_logits_)
            return x, router_logits
        else:
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask)
            return x
