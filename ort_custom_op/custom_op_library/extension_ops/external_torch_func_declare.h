// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/dlpack.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cache.h>
#include <ops.h>

// flas_api.cc
std::vector<at::Tensor> mha_varlen_fwd(
    const at::Tensor
        &q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor
        &k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor
        &v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor>
        &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q, // b+1
    const at::Tensor &cu_seqlens_k, // b+1
    c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    const int window_size_left, int window_size_right,
    const bool return_softmax, c10::optional<at::Generator> gen_);