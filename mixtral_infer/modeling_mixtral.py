"""Inference-only Mixtral model."""
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tensor_model_parallel_group)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               ReplicatedLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.layernorm import RMSNorm
from transformers import MixtralConfig
from torch import nn
import math
import torch.nn.functional as F
import transformers
from typing import List, Optional, Tuple
import sys
import numpy as np

import os
import torch
torch.zeros(1).cuda()


class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


class MoEBlockForOnnxExport(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        router_logits,
        batch_size,
        sequence_length,
        hidden_dim,
        top_k,
        num_experts,
        hidden_act,
        ffn_dim,
        start_expert_id,
        expert_weights_1,
        expert_weights_2,
        expert_weights_3,
    ):
        if get_tensor_model_parallel_world_size() > 1:
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            return final_hidden_states
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, top_k, dim=-1)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(num_experts):

            # NOTE(bowbao): A little bit of rewrite from nn.Linear to nn.functional.linear
            # expert_layer = self.experts[expert_idx]
            expert_weight_1 = expert_weights_1[expert_idx].T
            expert_weight_2 = expert_weights_2[expert_idx].T
            expert_weight_3 = expert_weights_3[expert_idx].T
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            # NOTE(bowbao): cont - rewrite nn.Linear with nn.functional.linear
            # current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            expert_layer_out = torch.nn.functional.linear(
                current_state, expert_weight_1)
            expert_layer_out = ACT2FN[hidden_act](expert_layer_out)
            expert_layer_out = expert_layer_out * \
                torch.nn.functional.linear(current_state, expert_weight_3)
            expert_layer_out = torch.nn.functional.linear(
                expert_layer_out, expert_weight_2)

            current_hidden_states = expert_layer_out * \
                routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states, router_logits, batch_size, sequence_length, hidden_dim, top_k, num_experts, hidden_act, ffn_dim,
                 start_expert_id,
                 expert_weights_1, expert_weights_2, expert_weights_3):
        moe_experts_bias1 = torch.zeros(
            num_experts, ffn_dim, dtype=hidden_states.type().dtype())
        moe_experts_bias2 = torch.zeros(
            get_tensor_model_parallel_world_size()*num_experts, hidden_dim, dtype=hidden_states.type().dtype())
        moe_experts_bias3 = torch.zeros(
            num_experts, ffn_dim, dtype=hidden_states.type().dtype())

        bias1 = g.op("Constant", value_t=moe_experts_bias1)
        bias2 = g.op("Constant", value_t=moe_experts_bias2)
        bias3 = g.op("Constant", value_t=moe_experts_bias3)
        None_value = g.op("Constant", value_t=torch.tensor(
            [], dtype=torch.float16))

        if get_tensor_model_parallel_world_size() > 1:
            final_hidden_states = g.op("com.microsoft::ShardedMoE", hidden_states, router_logits, expert_weights_1, bias1, expert_weights_2,
                                       bias2, expert_weights_3, activation_type_s="silu", k_i=top_k, normalize_routing_weights_i=1, local_experts_start_index_i=start_expert_id)
        else:
            final_hidden_states = g.op("com.microsoft::MoE", hidden_states, router_logits, expert_weights_1, None_value, bias1, expert_weights_2,
                                       None_value, bias2, expert_weights_3, None_value, bias3, activation_type_s="silu", k_i=top_k, normalize_routing_weights_i=1)
        final_hidden_states.setType(hidden_states.type())
        return final_hidden_states


class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_act = config.hidden_act
        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}.")
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(range(
            self.num_total_experts), self.tp_size)[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(
                f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList([
            MixtralMLP(self.num_total_experts,
                       config.hidden_size,
                       config.intermediate_size,
                       linear_method=linear_method)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     linear_method=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        if torch.onnx.is_in_onnx_export():
            # NOTE(bowbao): Alternatively, the interface can be cleaned up to avoid passing the scalar values such as
            # batch_size, etc. These are not used in onnx contrib op, and are only used in autograd.Function.forward
            # to provide parity with the original MoE implementation.
            final_hidden_states = MoEBlockForOnnxExport.apply(
                hidden_states,
                router_logits,
                batch_size,
                sequence_length,
                hidden_dim.item(),
                self.top_k,
                len(self.expert_indicies),
                self.hidden_act,
                int(self.experts[self.expert_indicies[0]
                                 ].w1.weight.shape[0].item()),
                self.expert_indicies[0],
                torch.stack(
                    [expert.w1.weight.T for expert in self.experts if expert is not None], dim=0),
                torch.stack(
                    [expert.w2.weight.T for expert in self.experts if expert is not None], dim=0),
                torch.stack(
                    [expert.w3.weight.T for expert in self.experts if expert is not None], dim=0),
            )
            return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)

            current_hidden_states = expert_layer(hidden_states).mul_(
                expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return tensor_model_parallel_all_reduce(final_hidden_states).view(
            batch_size, sequence_length, hidden_dim)


class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Copied from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Mixtral

# use when cache is not None
class GQAForOnnxExport(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_states,
        key_states,
        value_states,
        past_key,
        past_value,
        attention_mask,
        seqlens_k,
        total_seq_len,
        num_heads,
        head_dim,
        num_key_value_heads,
        num_key_value_groups,
    ):
        bsz, seq_len, hidden_size = query_states.size()

        query_states = query_states.view(
            bsz, -1, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, -1, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, -1, num_key_value_heads, head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        key_cache, value_cache = past_key, past_value

        kv_seq_len += key_cache.shape[-2]

        key_states = torch.cat([key_cache, key_states], dim=-2)
        value_states = torch.cat([value_cache, value_states], dim=-2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32[]
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, -1, hidden_size)

        return attn_output, key_states, value_states


    @staticmethod
    def symbolic(g: torch.Graph,
                 query_states,
                 key_states,
                 value_states,
                 past_key,
                 past_value,
                 attention_mask,
                 seqlens_k,
                 total_seq_len,
                 num_heads,
                 head_dim,
                 num_key_value_heads,
                 num_key_value_groups):
        import torch._C._onnx as _C_onnx
        total_sequence_length = g.op("Cast", total_seq_len, to_i=_C_onnx.TensorProtoDataType.INT32)
        sequence_lengths_k = g.op("Cast", seqlens_k, to_i=_C_onnx.TensorProtoDataType.INT32)
        if get_tensor_model_parallel_world_size() > 1:
            outputs = g.op("com.microsoft::GroupQueryAttention",
                            query_states,
                            key_states,
                            value_states,
                            past_key,
                            past_value,
                            sequence_lengths_k,
                            total_sequence_length,
                            num_heads_i=num_heads,
                            kv_num_heads_i=num_key_value_heads,
                            scale_f=0.08838834764,
                            outputs=3)
        else:
            outputs = g.op("com.microsoft::GroupQueryAttention",
                            query_states,
                            key_states,
                            value_states,
                            past_key,
                            past_value,
                            sequence_lengths_k,
                            total_sequence_length,
                            num_heads_i=num_heads,
                            kv_num_heads_i=num_key_value_heads,
                            scale_f=0.08838834764,
                            outputs=3)
        attn_output, present_key, present_value = outputs[0], outputs[1], outputs[2]
        attn_output.setType(query_states.type())
        present_key.setType(past_key.type())
        present_value.setType(past_value.type())

        return attn_output, present_key, present_value


class MixtralNotPagedAttn(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MixtralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads//tp_size
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads//tp_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seqlens_k: Optional[torch.Tensor] = None,
        past_key_value=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is not None and torch.onnx.is_in_onnx_export():
            key_cache, value_cache = past_key_value
            kv_seq_len = key_states.shape[-2]
            kv_seq_len += key_cache.shape[-2]
            attn_output, present_key, present_value = GQAForOnnxExport.apply(
                query_states,
                key_states,
                value_states,
                key_cache,
                value_cache,
                attention_mask,
                seqlens_k,
                kv_seq_len,
                self.num_heads,
                self.head_dim,
                self.num_key_value_heads,
                self.num_key_value_groups,
            )
            return attn_output, (present_key, present_value)

        bsz, seq_len, hidden_size = query_states.size()

        query_states = query_states.view(
            bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            key_cache, value_cache = past_key_value
            # print("key_cache", key_cache.shape)
            # print("value_cache", value_cache.shape)

            kv_seq_len += key_cache.shape[-2]

            key_states = torch.cat([key_cache, key_states], dim=-2)
            value_states = torch.cat([value_cache, value_states], dim=-2)
        past_key_value = (key_states, value_states)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32[]
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, -1, hidden_size)

        return attn_output, past_key_value


class MixtralAttention(nn.Module):
    def __init__(self,
                 layer_idx: Optional[int],
                 config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = MixtralNotPagedAttn(
            config, layer_idx
        )

    def forward(
        self,
        positions: torch.Tensor,
        attention_mask,
        seqlens_k,
        hidden_states: torch.Tensor,
        kv_cache,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # k_cache, v_cache = kv_cache
        attn_output, kv_cache = self.attn(q, k, v, attention_mask, seqlens_k, kv_cache)
        output, _ = self.o_proj(attn_output)
        return output, kv_cache


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        layer_idx,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            layer_idx,
            config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window,
            linear_method=linear_method)
        self.block_sparse_moe = MixtralMoE(config=config,
                                           linear_method=linear_method)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        seqlens_k: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states, kv_cache = self.self_attn(
            positions=positions,
            attention_mask=attention_mask,
            seqlens_k=seqlens_k,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual, kv_cache


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(_, config, linear_method=linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens_k: torch.Tensor,
        positions: torch.Tensor,
        kv_caches,
    ):
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        batch_size, seq_length = input_ids.shape
        past_key_values_length = 0
        if kv_caches is not None:
            past_key_values_length = kv_caches[0][0].shape[-2]
        if positions is None:
            device = input_ids.device
            positions = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            positions = positions.unsqueeze(0).view(-1, seq_length)
        else:
            positions = positions.view(-1, seq_length).long()
        attention_mask = _prepare_4d_causal_attention_mask(
            None,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
        kv_caches_list = [None] * len(self.layers)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual, kv_caches_list[i] = layer(positions, attention_mask, seqlens_k, hidden_states,
                                                               kv_caches[i] if kv_caches is not None else None,
                                                               residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, kv_caches_list


class MixtralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = MixtralModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.vocab_size = config.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        seqlens_k: torch.Tensor = None,
        positions: torch.Tensor = None,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states, kv_caches = self.model(input_ids, seqlens_k, positions, kv_caches)
        logits = torch.matmul(hidden_states, self.lm_head.weight.t())
        logits = tensor_model_parallel_all_gather(logits)
        return logits, kv_caches

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False):
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # if 'model.layers.' in name and 'model.layers.0' not in name:
                #    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # if 'model.layers.' in name and 'model.layers.0' not in name:
                #    continue
                # Skip experts that are not assigned to this worker.
                if ("block_sparse_moe.experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
