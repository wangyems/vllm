// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "paged_attn_ops.h"

#include "cuda_context.h"
#include "cuda_ops.h"
#include "external_torch_func_declare.h"
#include "pt_to_ort_resource.h"
#include "raw_tensor.h"
#include "utils.h"

using namespace Ort::Custom;

namespace Cuda {
void DebugIntoPython(const torch::Tensor &input);
InputMetadata *GetInputMetadataFromPyObj() {
  auto &input_metadata = Resource::get_instance().input_metadata;
  if (input_metadata == nullptr) {
    GilGuard guard;
    auto py_module = py::module::import("vllm.model_executor.input_c_metadata");
    auto resultobj = py_module.attr("ConvertInputMetadataToC")();
    int64_t result = resultobj.cast<int64_t>();
    input_metadata = reinterpret_cast<InputMetadata *>(result);
  }
  return input_metadata;
}

template <typename T>
void PagedAttentionOpCompute(
    torch::Tensor output_torch, torch::Tensor &query_torch,
    torch::Tensor &key_torch, torch::Tensor &value_torch,
    torch::Tensor &key_cache_torch, torch::Tensor &value_cache_torch,
    InputMetadata *input_metadata, AttnParam &attn_param_,
    cudaStream_t cuda_stream, int device_id) {
  auto query_view =
      query_torch.view({-1, attn_param_.num_heads_, attn_param_.head_size_});
  auto key_view =
      key_torch.view({-1, attn_param_.num_kv_heads_, attn_param_.head_size_});
  auto value_view =
      value_torch.view({-1, attn_param_.num_kv_heads_, attn_param_.head_size_});
  auto output_view = output_torch.view({-1, attn_param_.num_heads_, attn_param_.head_size_});

  // if (input_metadata->cache_events.events[0]) {
  //   (cudaStreamWaitEvent(input_metadata->cache_stream,
  //                        input_metadata->cache_events.events[0]));
  //   std::copy(input_metadata->cache_events.events + 1,
  //             input_metadata->cache_events.events + 80,
  //             input_metadata->cache_events.events);
  // }

  input_metadata = input_metadata ? input_metadata : GetInputMetadataFromPyObj();

  if (key_cache_torch.numel() > 3 && value_cache_torch.numel() > 3) {
    auto slot_mapping_torch = torch::from_blob(
        reinterpret_cast<int64_t *>(input_metadata->slot_mapping),
        {query_torch.sizes()[0], query_torch.sizes()[1]},
        torch::TensorOptions()
            .device(torch::kCUDA, device_id)
            .dtype(torch::kInt64));

    reshape_and_cache(key_view, value_view, key_cache_torch, value_cache_torch,
                      slot_mapping_torch);
    CHECK_CUDA_ERROR();
  }

  int64_t is_prompt = input_metadata->is_prompt;
  if (is_prompt > 0) {
    auto seqstart_q_torch =
        torch::from_blob(reinterpret_cast<int32_t *>(
                             input_metadata->attn_bias.q_seqinfo.seqstart),
                         {input_metadata->attn_bias.batchsize + 1},
                         torch::TensorOptions()
                             .device(torch::kCUDA, device_id)
                             .dtype(torch::kInt32));

    auto seqstart_k_torch =
        torch::from_blob(reinterpret_cast<int32_t *>(
                             input_metadata->attn_bias.k_seqinfo.seqstart),
                         {input_metadata->attn_bias.batchsize + 1},
                         torch::TensorOptions()
                             .device(torch::kCUDA, device_id)
                             .dtype(torch::kInt32));

    auto max_seqlen_q = input_metadata->attn_bias.q_seqinfo.max_seqlen;
    auto max_seqlen_k = input_metadata->attn_bias.k_seqinfo.max_seqlen;
    auto optional_output_torch = c10::make_optional(output_view);
    // multihead_attention_cuda
    auto dprops = at::cuda::getCurrentDeviceProperties();
    c10::optional<torch::Tensor> none;
    if (dprops->major >= 8 && dprops->minor >= 0) {
      mha_varlen_fwd(query_view, key_view, value_view, optional_output_torch,
                     seqstart_q_torch, seqstart_k_torch,
                     none,
                     max_seqlen_q, max_seqlen_k,
                     0.0f, attn_param_.scale_, false, true, -1,
                     -1, false, c10::nullopt);
    } else {
      c10::OperatorName full_name(
          "xformers::efficient_attention_forward_cutlass", "");
      auto op = torch::jit::findOperatorFor(full_name);
      TORCH_INTERNAL_ASSERT(op);

      torch::jit::Stack stack;
      torch::jit::push(stack, query_view.unsqueeze(0));
      torch::jit::push(stack, key_view.unsqueeze(0));
      torch::jit::push(stack, value_view.unsqueeze(0));
      torch::jit::push(stack, c10::nullopt); // bias
      torch::jit::push(stack, seqstart_q_torch); // seqstart_q
      torch::jit::push(stack, seqstart_k_torch); // seqstart_k
      torch::jit::push(stack, max_seqlen_q);     // max_seqlen_q_
      torch::jit::push(stack, 0.0);
      torch::jit::push(stack, false); // compute_logsumexp
      torch::jit::push(stack, 1);     // custom_mask_type
      torch::jit::push(stack, attn_param_.scale_);
      torch::jit::push(stack, c10::nullopt); // seqlen_k
      torch::jit::push(stack, c10::nullopt); // window_size

      op->getOperation()(stack);
      auto rets = torch::jit::pop(stack, 4)[0];
      const auto &tensor1 = (rets).toTensor();
      // memory efficient attention
      // auto ret = xformers::efficient_attention_forward_cutlass(
      //    query_view.unsqueeze(0), key_view.unsqueeze(0),
      //    value_view.unsqueeze(0), c10::nullopt, seqstart_q_torch,
      //    seqstart_k_torch, max_seqlen_q, 0.0, false, 1, scale_,
      //    c10::nullopt, c10::nullopt);
      // auto torch_out = std::get<0>(ret);

      // output_torch.copy_(tensor1.view(output_torch.sizes()));
      cudaMemcpyAsync(output_torch.data_ptr(), tensor1.data_ptr(),
                      output_torch.numel() * sizeof(T),
                      cudaMemcpyDeviceToDevice, cuda_stream);
    }
    CHECK_CUDA_ERROR();
  }

  auto query_shape = query_torch.sizes();

  if (is_prompt == 0) {
    constexpr int PARTITION_SIZE = 512;
    int max_num_partitions =
        ((input_metadata->max_context_len + PARTITION_SIZE - 1) /
         PARTITION_SIZE);
    // TODO : Tune this heuristic.
    bool use_v1 =
        input_metadata->max_context_len <= 8192 &&
        (max_num_partitions == 1 ||
         (query_shape[0] * query_shape[1] * query_shape[2]) > PARTITION_SIZE);

    torch::Tensor block_tables = torch::from_blob(
        reinterpret_cast<int32_t *>(input_metadata->block_tables),
        {query_shape[0], input_metadata->block_tables_size_1},
        torch::TensorOptions()
            .device(torch::kCUDA, device_id)
            .dtype(torch::kInt32));
    torch::Tensor context_lens = torch::from_blob(
        reinterpret_cast<int32_t *>(input_metadata->context_lens),
        {query_shape[0]},
        torch::TensorOptions()
            .device(torch::kCUDA, device_id)
            .dtype(torch::kInt32));
    if (use_v1) {
      paged_attention_v1(output_torch, query_view, key_cache_torch,
                         value_cache_torch, attn_param_.num_kv_heads_,
                         attn_param_.scale_, block_tables, context_lens,
                         value_cache_torch.size(3),
                         input_metadata->max_context_len, c10::nullopt);
      CHECK_CUDA_ERROR();

    } else {
      torch::Tensor tmp_output =
          torch::empty({query_view.sizes()[0], attn_param_.num_heads_,
                        max_num_partitions, attn_param_.head_size_},
                       torch::TensorOptions()
                           .dtype(PTTypeInfo<T>::type)
                           .device(torch::kCUDA, device_id));
      torch::Tensor exp_sums = torch::empty(
          {query_view.sizes()[0], attn_param_.num_heads_, max_num_partitions},
          torch::TensorOptions()
              .device(torch::kCUDA, device_id)
              .dtype(torch::kFloat32));
      torch::Tensor max_logits = torch::empty_like(exp_sums);
      paged_attention_v2(output_torch, exp_sums, max_logits, tmp_output,
                         query_view, key_cache_torch, value_cache_torch,
                         attn_param_.num_kv_heads_, attn_param_.scale_,
                         block_tables, context_lens, value_cache_torch.size(3),
                         input_metadata->max_context_len, c10::nullopt);
      CHECK_CUDA_ERROR();
    }
  }
}


KernelPagedAttentionOp::KernelPagedAttentionOp(const OrtApi &api,
                                               const OrtKernelInfo *info)
    : _api(api), _info(info) {
  int64_t num_heads = 0;
  int64_t head_size = 0;
  CUSTOM_ENFORCE(api.KernelInfoGetAttribute_int64(info, "num_heads",
                                                  &num_heads) == nullptr &&
                     num_heads > 0,
                 "num_heads attribute is not set");
  attn_param_.num_heads_ = static_cast<int32_t>(num_heads);

  int64_t num_kv_heads = num_heads;
  api.KernelInfoGetAttribute_int64(info, "num_kv_heads", &num_kv_heads);
  CUSTOM_ENFORCE(api.KernelInfoGetAttribute_int64(info, "head_size",
                                                  &head_size) == nullptr &&
                     head_size > 0,
                 "head_size attribute is not set");
  attn_param_.head_size_ = static_cast<int32_t>(head_size);

  attn_param_.num_kv_heads_ = static_cast<int32_t>(num_kv_heads);
  attn_param_.num_queries_per_kv_ =
      attn_param_.num_heads_ / attn_param_.num_kv_heads_;

  torch::Tensor head_mapping_host =
      torch::empty({attn_param_.num_heads_}, torch::kInt32);

  auto *head_mapping_host_ptr = head_mapping_host.data_ptr<int32_t>();
  for (int i = 0; i < attn_param_.num_kv_heads_; i++) {
    for (int j = 0; j < attn_param_.num_queries_per_kv_; j++) {
      head_mapping_host_ptr[i * attn_param_.num_queries_per_kv_ + j] = i;
    }
  }

  attn_param_.head_mapping_ = head_mapping_host.to(torch::kCUDA);

  CUSTOM_ENFORCE(api.KernelInfoGetAttribute_float(
                     info, "scale", &attn_param_.scale_) == nullptr &&
                     attn_param_.scale_ > 0,
                 "scale attribute is not set");
}

void KernelPagedAttentionOp::Compute(OrtKernelContext *context) {
  Ort::Custom::CudaContext cuda_ctx;
  cuda_ctx.Init(*context);
  Ort::KernelContext ctx(context);

  const char *LOCAL_RANK = getenv("LOCAL_RANK");
  int device_id = LOCAL_RANK ? std::stoi(LOCAL_RANK) : 0;

  at::cuda::CUDAStream myStream =
      at::cuda::getStreamFromExternal(cuda_ctx.cuda_stream, device_id);
  at::cuda::setCurrentCUDAStream(myStream);

  // get input tensors
  const auto query = ctx.GetInput(0);
  const auto key = ctx.GetInput(1);
  const auto value = ctx.GetInput(2);
  const auto key_cache = ctx.GetInput(3);
  const auto value_cache = ctx.GetInput(4);

  using T = Ort::Float16_t;
  const auto &query_shape = query.GetTensorTypeAndShapeInfo().GetShape();

  TensorShape output_shape = query_shape;
  CUSTOM_ENFORCE(query_shape.back() ==
                     attn_param_.num_heads_ * attn_param_.head_size_,
                 "invlaid query shape");

  auto output = ctx.GetOutput(0, output_shape);

  torch::Tensor query_torch =
      torch::from_blob(const_cast<T *>(query.GetTensorData<T>()),
                       query.GetTensorTypeAndShapeInfo().GetShape(),
                       torch::TensorOptions()
                           .device(torch::kCUDA, device_id)
                           .dtype(PTTypeInfo<T>::type));
  torch::Tensor key_torch =
      torch::from_blob(const_cast<T *>(key.GetTensorData<T>()),
                       key.GetTensorTypeAndShapeInfo().GetShape(),
                       torch::TensorOptions()
                           .device(torch::kCUDA, device_id)
                           .dtype(PTTypeInfo<T>::type));
  torch::Tensor value_torch =
      torch::from_blob(const_cast<T *>(value.GetTensorData<T>()),
                       value.GetTensorTypeAndShapeInfo().GetShape(),
                       torch::TensorOptions()
                           .device(torch::kCUDA, device_id)
                           .dtype(PTTypeInfo<T>::type));
  torch::Tensor key_cache_torch =
      torch::from_blob(const_cast<T *>(key_cache.GetTensorData<T>()),
                       key_cache.GetTensorTypeAndShapeInfo().GetShape(),
                       torch::TensorOptions()
                           .device(torch::kCUDA, device_id)
                           .dtype(PTTypeInfo<T>::type));
  torch::Tensor value_cache_torch =
      torch::from_blob(const_cast<T *>(value_cache.GetTensorData<T>()),
                       value_cache.GetTensorTypeAndShapeInfo().GetShape(),
                       torch::TensorOptions()
                           .device(torch::kCUDA, device_id)
                           .dtype(PTTypeInfo<T>::type));
  torch::Tensor output_torch =
      torch::from_blob(output.GetTensorMutableData<T>(),
                       output.GetTensorTypeAndShapeInfo().GetShape(),
                       torch::TensorOptions()
                           .device(torch::kCUDA, device_id)
                           .dtype(PTTypeInfo<T>::type));

  PagedAttentionOpCompute<T>(output_torch, query_torch, key_torch, value_torch,
                             key_cache_torch, value_cache_torch, nullptr,
                             attn_param_, cuda_ctx.cuda_stream, device_id);
  return;
}

} // namespace Cuda

namespace torch_ext {

void reset_ort_input_metadata() {
  Cuda::Resource::get_instance().input_metadata = nullptr;
}

torch::Tensor paged_attention_forwad(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key,
    torch::Tensor &value, c10::optional<torch::Tensor> &key_cache_op,
    c10::optional<torch::Tensor> &value_cache_op, int64_t t_input_metadata,
    float scale, c10::optional<torch::Tensor> alibi_slopes, int num_head,
    int num_kv_head, int head_size) {
  InputMetadata *input_metadata =
      reinterpret_cast<InputMetadata *>(t_input_metadata);
  Cuda::AttnParam attn_param_;
  attn_param_.num_heads_ = num_head;
  attn_param_.num_kv_heads_ = num_kv_head;
  attn_param_.head_size_ = head_size;
  attn_param_.num_queries_per_kv_ =
      attn_param_.num_heads_ / attn_param_.num_kv_heads_;
  attn_param_.scale_ = scale;
  attn_param_.head_mapping_ =
      torch::empty({attn_param_.num_heads_}, torch::kInt32);
  auto *head_mapping_host_ptr = attn_param_.head_mapping_.data_ptr<int32_t>();
  for (int i = 0; i < attn_param_.num_kv_heads_; i++) {
    for (int j = 0; j < attn_param_.num_queries_per_kv_; j++) {
      head_mapping_host_ptr[i * attn_param_.num_queries_per_kv_ + j] = i;
    }
  }

  torch::Tensor key_cache, value_cache;
  if (key_cache_op.has_value()) {
    key_cache = key_cache_op.value();
  } else {
    key_cache = torch::empty({0}, torch::kFloat16);
  }
  if (value_cache_op.has_value()) {
    value_cache = value_cache_op.value();
  } else {
    value_cache = torch::empty({0}, torch::kFloat16);
  }
  attn_param_.head_mapping_ = attn_param_.head_mapping_.to(torch::kCUDA);
  Cuda::PagedAttentionOpCompute<Ort::Float16_t>(
      out, query, key, value, key_cache, value_cache, input_metadata,
      attn_param_, at::cuda::getCurrentCUDAStream(), query.device().index());
  return out;
}

} // namespace torch_ext
