import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union

try:
    from liger_kernel.transformers import LigerSwiGLUMLP as MLP
except ImportError:
    from .mlp import MLP
    

def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: int,
    top_k: int,
    attn_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:

    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        concatenated_gate_logits = torch.cat(gate_logits, dim=0)

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)

    if attn_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        B, S = attn_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (B * S)

        expert_attn_mask = (
            attn_mask[None, :, :, None, None]
            .expand((num_hidden_layers, B, S, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
        )

        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attn_mask, dim=0
        ) / torch.sum(expert_attn_mask, dim=0)

        router_per_expert_attn_mask = (
            attn_mask[None, :, :, None]
            .expand((num_hidden_layers, B, S, num_experts))
            .reshape(-1, num_experts)
        )

        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attn_mask, dim=0
        ) / torch.sum(router_per_expert_attn_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Experts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        for _ in range(config.num_experts):
            self.append(MLP(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        D = hidden_states.shape[-1]
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, D)
            current_hidden_states = (
                self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            ).to(hidden_states.dtype)
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        return final_hidden_states


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = Experts(config)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def route_tokens_to_experts(self, router_logits, dtype):
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_index = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = hidden_states.shape
        hidden_states = hidden_states.view(-1, D)
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(
            router_logits, hidden_states.dtype
        )
        final_hidden_states = self.experts(
            hidden_states, top_k_index, top_k_weights
        ).reshape(B, S, D)

        return (
            (final_hidden_states, router_logits)
            if self.training
            else (final_hidden_states, None)
        )
