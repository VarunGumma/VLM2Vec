# Based of https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py

import torch.nn as nn
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb


class LigerSwiGLUMLPWithBias(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, x):
        return self.down_proj(
            LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
        )


def apply_liger_kernel_to_qwen2_vl():
    from src.model.vlm_backbone.qwen2_vl import modeling_qwen2_vl

    modeling_qwen2_vl.Qwen2RMSNorm = LigerRMSNorm
    modeling_qwen2_vl.Qwen2MLP = LigerSwiGLUMLP
    modeling_qwen2_vl.LayerNorm = LigerLayerNorm
    modeling_qwen2_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb


def apply_liger_kernel_to_qwen2_5_vl():
    from src.model.vlm_backbone.qwen2_5_vl import modeling_qwen2_5_vl

    modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
    # this is a hack as the Qwen2_5_VLMLP has bias terms
    modeling_qwen2_5_vl.Qwen2_5_VLMLP = LigerSwiGLUMLPWithBias
    modeling_qwen2_5_vl.LayerNorm = LigerLayerNorm
    modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (
        liger_multimodal_rotary_pos_emb
    )
