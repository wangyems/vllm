"""Inference-only Mixtral model."""
from multiprocessing import Process, set_start_method
from vllm.worker.worker import _init_distributed_environment
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tensor_model_parallel_group)
from vllm.config import ParallelConfig
from transformers import MixtralConfig
from torch import nn
import math
import shutil
import torch.nn.functional as F
import transformers
from typing import List, Optional, Tuple
import sys
import numpy as np
from modeling_mixtral import MixtralForCausalLM
import os
import torch
from pathlib import Path
torch.zeros(1).cuda()

model_name = 'mistralai/Mixtral-8x7B-v0.1'
#model_name = "/wy/onnx_models/mixtral/mixtral"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def init_test_distributed_environment(tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str = "51408"):
    parallel_config = ParallelConfig(1, tensor_parallel_size,
                                     worker_use_ray=True)
    distributed_init_method = f"tcp://127.0.0.1:{distributed_init_port}"
    torch.cuda.set_device(rank)
    _init_distributed_environment(
        parallel_config, rank, distributed_init_method)


def export_onnx(tensor_parallel_size, rank):
    init_test_distributed_environment(tensor_parallel_size, rank)
    torch.set_default_dtype(torch.float16)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    inputs = tokenizer("once upon a time ", return_tensors="pt").to("cuda")
    torch.cuda.set_device(rank)
    model = MixtralForCausalLM(config)
    model.load_weights(model_name)
    model.eval()
    model.to("cuda")
    out, past_key_value = model(**inputs)
    onnx_pt_inputs = (inputs.input_ids, None, None, None, None)
    tmp_onnx = Path(f"./tmp{rank}/mixtral_rank{rank}.onnx")
    tmp_onnx.parent.exists() and shutil.rmtree(tmp_onnx.parent)
    tmp_onnx.parent.mkdir(exist_ok=True)
    onnx_model_path = Path(f"./onnx_models/mixtral_rank{rank}.onnx").absolute()
    onnx_model_path.parent.mkdir(exist_ok=True)
    onnx_model_path.exists() and onnx_model_path.unlink()
    (onnx_model_path.parent/onnx_model_path.with_suffix('.data').name).exists() and (
        onnx_model_path.parent/onnx_model_path.with_suffix('.data').name).unlink()
    onnx_inp_names = ("input_ids",)
    onnx_out_names = ("last_hidden_state",)
    for layer_idx in range(model.config.num_hidden_layers):
        onnx_out_names = onnx_out_names + \
            (f"present.key.{layer_idx}",
             f"present.value.{layer_idx}")
    torch.onnx.export(model=model, args=tuple(onnx_pt_inputs), f=str(tmp_onnx), verbose=False, opset_version=17,
                      input_names=tuple(onnx_inp_names), output_names=tuple(onnx_out_names),
                      dynamic_axes={"input_ids": {
                          0: "batch_size", 1: "seq_len"}},
                      autograd_inlining=False)
    torch.distributed.barrier(group=get_tensor_model_parallel_group())
    import onnx
    onnx_model = onnx.load(str(tmp_onnx))
    onnx.save_model(onnx_model, str(onnx_model_path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=tmp_onnx.with_suffix('.data').name, size_threshold=1024, convert_attribute=False)

    seqlens_k = torch.tensor([inputs.input_ids.shape[0]]).cuda()
    onnx_pt_inputs = (inputs.input_ids, None, seqlens_k, None, past_key_value)
    tmp_onnx = Path(f"./tmp{rank}/mixtral_with_past_rank{rank}.onnx")
    tmp_onnx.parent.exists() and shutil.rmtree(tmp_onnx.parent)
    tmp_onnx.parent.mkdir(exist_ok=True)
    onnx_pastmodel_path = Path(
        f"./onnx_models/mixtral_with_past_rank{rank}.onnx").absolute()
    onnx_pastmodel_path.exists() and onnx_pastmodel_path.unlink()
    (onnx_pastmodel_path.parent/onnx_pastmodel_path.with_suffix('.data').name).exists() and (
        onnx_pastmodel_path.parent/onnx_pastmodel_path.with_suffix('.data').name).unlink()
    onnx_inp_names = ("input_ids", "seqlens_k")
    for layer_idx in range(model.config.num_hidden_layers):
        onnx_inp_names = onnx_inp_names + \
            (f"past.key.{layer_idx}", f"past.value.{layer_idx}")
    dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}, "seqlens_k": {0: "batch_size"}}
    for layer_idx in range(model.config.num_hidden_layers):
        dynamic_axes[f"past.key.{layer_idx}"] = {
            0: "batch_size", 2: "seq_len", 1: "num_heads", 3: "head_dim"}
        dynamic_axes[f"past.value.{layer_idx}"] = {
            0: "batch_size", 2: "seq_len", 1: "num_heads", 3: "head_dim"}
    torch.onnx.export(model=model, args=tuple(onnx_pt_inputs), f=str(tmp_onnx), verbose=False, opset_version=17,
                      input_names=tuple(onnx_inp_names), output_names=tuple(onnx_out_names),
                      dynamic_axes=dynamic_axes,
                      autograd_inlining=False)
    torch.distributed.barrier(group=get_tensor_model_parallel_group())
    import onnx
    onnx_model = onnx.load(str(tmp_onnx))
    onnx.save_model(onnx_model, str(onnx_pastmodel_path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=tmp_onnx.with_suffix('.data').name, size_threshold=1024, convert_attribute=False)


def infer_model(tensor_parallel_size, rank, model_or_sess):
    torch.set_default_dtype(torch.float16)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    inputs = tokenizer("once upon a time ", return_tensors="pt").to("cuda")

    if not isinstance(model_or_sess, nn.Module):
        batch_size = inputs.input_ids.shape[0]
        seq_len = inputs.input_ids.shape[1]
        print(inputs.input_ids.shape, inputs.input_ids)
        onnx_inputs = {"input_ids": inputs.input_ids.cpu().numpy()}
        onnx_inputs["seqlens_k"] = np.array([seq_len + 1] * batch_size) # hack
        for layer_idx in range(config.num_hidden_layers):
            onnx_inputs[f"past.key.{layer_idx}"] = np.zeros([batch_size, 2, 0, 128]).astype(np.float16)
            onnx_inputs[f"past.value.{layer_idx}"] = np.zeros([batch_size, 2, 0, 128]).astype(np.float16)
        ortout = model_or_sess.run(None, onnx_inputs)
        out = torch.from_numpy(ortout[0]).cuda()
        #onnx_model_path = Path(f"./onnx_models/mixtral_with_past_rank{rank}.onnx").absolute()
        # import onnxruntime
        # from vllm import paged_attn
        # session_options = onnxruntime.SessionOptions()
        # session_options.register_custom_ops_library(paged_attn.__file__)
        # provider_opt = {"device_id": rank, }
        # model_or_sess = onnxruntime.InferenceSession(str(onnx_model_path), providers=[(
        #     "CUDAExecutionProvider", provider_opt)], sess_options=session_options)
    else:
        out, past_key_value = model_or_sess(**inputs)
    gen_ids = []
    while len(gen_ids) < 100:
        seqlens_k = len(gen_ids) + 1
        next_id = out[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(next_id)
        if isinstance(model_or_sess, nn.Module):
            out, past_key_value = model_or_sess(
                next_id, kv_caches=past_key_value)
        else:
            onnx_inputs = {"input_ids": next_id.cpu().numpy()}
            onnx_inputs["seqlens_k"] = np.array([seqlens_k] * next_id.shape[0]) # hack
            for layer_idx in range(config.num_hidden_layers):
                onnx_inputs[f"past.key.{layer_idx}"] = ortout[1+layer_idx*2]
                onnx_inputs[f"past.value.{layer_idx}"] = ortout[1+layer_idx*2+1]

            ortout = model_or_sess.run(None, onnx_inputs)
            out = torch.from_numpy(ortout[0]).cuda()
    gen_ids = torch.cat(gen_ids, dim=-1)
    if rank == 0:
        print(gen_ids[0])
        print(tokenizer.decode(gen_ids[0]))


def test_model_load(tensor_parallel_size, rank, test_torch=True):
    init_test_distributed_environment(tensor_parallel_size, rank)
    torch.set_default_dtype(torch.float16)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True)

    os.environ['LOCAL_WORLD_SIZE'] = str(tensor_parallel_size)
    os.environ['LOCAL_RANK'] = str(rank)
    torch.cuda.set_device(rank)
    if test_torch:
        model = MixtralForCausalLM(config)
        model.load_weights(model_name)
        model.eval()
        model.to("cuda")
        #out, past_key_value = model(**inputs)
        # del model,out
        # torch.cuda.empty_cache()
    else:
        import onnxruntime
        from vllm import paged_attn
        session_options = onnxruntime.SessionOptions()
        session_options.register_custom_ops_library(paged_attn.__file__)
        provider_opt = {"device_id": rank, }
        # onnx_model_path = Path(
        #    f"./onnx_models/mixtral_rank{rank}.onnx").absolute()
        # onnx_model_path = Path(
        #     f"/home/jicwen/work/vllm/mixtral_infer/onnx_models/mixtral_rank{rank}.onnx").absolute()
        onnx_model_path = Path(f"./onnx_models/mixtral_with_past_rank{rank}.onnx").absolute()
        sess = onnxruntime.InferenceSession(str(onnx_model_path), providers=[(
            "CUDAExecutionProvider", provider_opt)], sess_options=session_options)

    infer_model(tensor_parallel_size, rank, model if test_torch else sess)


def process_entry(tensor_parallel_size, rank):
    #export_onnx(tensor_parallel_size, rank)
    #test_model_load(tensor_parallel_size, rank, test_torch=True)
    test_model_load(tensor_parallel_size, rank, test_torch=False)


if __name__ == "__main__":
    tensor_parallel_size = 4
    if tensor_parallel_size == 1:
        test_model_load(tensor_parallel_size, 0)
    else:
        set_start_method("spawn", force=True)

        processes = []
        for rank in range(tensor_parallel_size):
            p = Process(target=process_entry,
                        args=(tensor_parallel_size, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
