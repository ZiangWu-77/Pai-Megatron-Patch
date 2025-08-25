# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
    TEColumnParallelLinear,
    TorchLayerNormTEColumnParallelLinear,
    TorchLinear,
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear
)
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from .attention_vision import SelfAttention
from .attention import SelfAttention as Qwen2VLSelfAttention
from typing import Optional
import warnings

from megatron.core.transformer.moe.experts import GroupedMLP, TEGroupedMLP, SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_shared_expert_gate: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    mlp = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
    )

    # experts spec
    if moe_grouped_gemm:
        ## use GroupedMLP
        if use_te and TEColumnParallelGroupedLinear is not None and not moe_use_legacy_grouped_gemm:
            ## use TEGroupedLinear
            expert_module = TEGroupedMLP
            expert_submodule = MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
            )
        else:
            ## use legacy GroupedMLP
            expert_module = GroupedMLP
            expert_submodule = None
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
    else:
        ## use SequentialMLP
        expert_module = SequentialMLP
        if use_te and not is_te_min_version("1.7.0.dev0"):
            warnings.warn(
                "Only transformer-engine>=1.7.0 supports MoE experts, "
                f"but your version is {get_te_version()}. Use local linear implementation instead."
            )
            expert_submodule = MLPSubmodules(
                linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
            )
        else:
            expert_submodule = mlp

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": use_shared_expert_gate}, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec
    
# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    qk_layernorm: bool = False,
    num_experts: int = None,
    moe_grouped_gemm: bool = None
) -> ModuleSpec:
    mlp = get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    qk_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Qwen2VLSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )

def get_qwen2vl_vision_model_spec(
    is_vit=False     
) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask # THD --> causal_pad

    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Helper function to get module spec for MLP/MoE
def get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False, add_norm: bool = True, moe_use_legacy_grouped_gemm: Optional[bool] = False
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        if add_norm:
            return ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                ),
            )
        else:
            return ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                ),
            )            
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec(
            use_te=True,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm
        )
    
### debug [Insure the forward process right.]
# # Use this spec to use lower level Transformer Engine modules (required for fp8 training)
# def get_gpt_layer_with_transformer_engine_spec(
#     qk_layernorm: bool = False
# ) -> ModuleSpec:
#     mlp = get_mlp_module_spec(
#         use_te=True, num_experts=None, moe_grouped_gemm=False
#     )
#     return ModuleSpec(
#         module=TransformerLayer,
#         submodules=TransformerLayerSubmodules(
#             self_attention=ModuleSpec(
#                 module=Qwen2VLSelfAttention,
#                 params={"attn_mask_type": AttnMaskType.causal},
#                 submodules=SelfAttentionSubmodules(
#                     linear_qkv=TorchLayerNormTEColumnParallelLinear,
#                     core_attention=TEDotProductAttention,
#                     linear_proj=TorchLinear,
#                     q_layernorm=TENorm if qk_layernorm else IdentityOp,
#                     k_layernorm=TENorm if qk_layernorm else IdentityOp,
#                 ),
#             ),
#             self_attn_bda=get_bias_dropout_add,
#             pre_mlp_layernorm=IdentityOp,
#             mlp=mlp,
#             mlp_bda=get_bias_dropout_add,
#         ),
#     )

# def get_qwen2vl_vision_model_spec(
#     is_vit=False     
# ) -> ModuleSpec:
#     attn_mask_type = AttnMaskType.no_mask # THD --> causal_pad

#     mlp = ModuleSpec(
#         module=MLP,
#         submodules=MLPSubmodules(
#             linear_fc1=TorchLayerNormTEColumnParallelLinear,
#             linear_fc2=TorchLinear,
#         ),
#     )
#     return ModuleSpec(
#         module=TransformerLayer,
#         submodules=TransformerLayerSubmodules(
#             self_attention=ModuleSpec(
#                 module=SelfAttention,
#                 params={"attn_mask_type": attn_mask_type},
#                 submodules=SelfAttentionSubmodules(
#                     linear_qkv=TorchLayerNormTEColumnParallelLinear,
#                     core_attention=TEDotProductAttention,
#                     linear_proj=TorchLinear,
#                     q_layernorm=IdentityOp,
#                     k_layernorm=IdentityOp,
#                 ),
#             ),
#             self_attn_bda=get_bias_dropout_add,
#             pre_mlp_layernorm=IdentityOp,
#             mlp=mlp,
#             mlp_bda=get_bias_dropout_add,
#         ),
#     )

# # Helper function to get module spec for MLP/MoE
# def get_mlp_module_spec(
#     use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False, add_norm: bool = True
# ) -> ModuleSpec:
#     if num_experts is None:
#         # Dense MLP w/ or w/o TE modules.
#         if add_norm:
#             return ModuleSpec(
#                 module=MLP,
#                 submodules=MLPSubmodules(
#                     linear_fc1=TorchLayerNormTEColumnParallelLinear if use_te else ColumnParallelLinear,
#                     linear_fc2=TorchLinear if use_te else RowParallelLinear,
#                 ),
#             )
#         else:
#             return ModuleSpec(
#                 module=MLP,
#                 submodules=MLPSubmodules(
#                     linear_fc1=TorchLinear if use_te else ColumnParallelLinear,
#                     linear_fc2=TorchLinear if use_te else RowParallelLinear,
#                 ),
#             )            
#     else:
#         # Mixture of experts with modules in megatron core.
#         raise NotImplementedError()