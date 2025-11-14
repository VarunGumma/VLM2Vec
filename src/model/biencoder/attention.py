import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

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
            self._expand_mask(attn_mask.to(torch.bool))
            if attn_mask is not None
            else None
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
