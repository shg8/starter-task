"""Baseline: SGLang's NVFP4 MoE expert block on Hopper (the Marlin W4A16 path).

This file is thin glue around SGLang (https://github.com/sgl-project/sglang), which is licensed
under the Apache License 2.0. Every kernel it runs belongs to SGLang. It reproduces what
MuseSpark1xFp4Experts._forward_marlin and _marlin_expert_outputs do in
python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py at SGLang v0.5.23
(commit 297e54fd105e974cdf401e5b4d20253a22c16f90), starting after the pre-expert norm and
without the cross-GPU all-reduce, with expert parallelism (32 of 256 experts on this GPU):

  prepare:   prepare_moe_nvfp4_layer_for_marlin (srt/layers/quantization/marlin_utils_fp4.py)
  kernel_fn: map global expert ids to local ones -> moe_align_block_size ->
             moe_wna16_marlin_gemm (4096 -> 8192) -> silu_and_mul ->
             moe_wna16_marlin_gemm (4096 -> 4096) -> fused_post_expert_norm_weighted_sum

It runs inside the SGLang container image (see README.md). It follows the same interface as
submission.py.
"""

from __future__ import annotations

import torch

_BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)   # as in muse_spark_fp4_moe.py


def prepare(weights):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    # A stand-in for the layer object with the attributes the SGLang function reads. As in the
    # layer, the post-expert norm is MuseSpark1xRMSNormWithInputScale with gain centre 1, so its
    # effective weight is (post_norm_w + 1) in bfloat16.
    layer = torch.nn.Module()
    layer.w13_weight = weights["w13"]
    layer.w2_weight = weights["w2"]
    layer.w13_weight_scale = weights["w13_sf"]
    layer.w2_weight_scale = weights["w2_sf"]
    layer.w13_weight_scale_2 = weights["w13_gs"]
    layer.w2_weight_scale_2 = weights["w2_gs"]
    layer.intermediate_size_per_partition = weights["w13"].shape[1] // 2
    layer.params_dtype = torch.bfloat16
    layer.moe_runner_config = type("MoeRunnerConfig", (), {"is_gated": True})()
    prepare_moe_nvfp4_layer_for_marlin(layer, group_size=16)
    return {
        "w13": layer.w13_weight,
        "w2": layer.w2_weight,
        "w13_scale": layer.w13_weight_scale,
        "w2_scale": layer.w2_weight_scale,
        "w13_scale_2": layer.w13_weight_scale_2,
        "w2_scale_2": layer.w2_weight_scale_2,
        "norm_w": (weights["post_norm_w"] + 1.0).to(torch.bfloat16).contiguous(),
        "expert_maps": {},
    }


def _expert_map(state, cfg, device):
    # Built once per config, as the layer builds it once in __init__ (derived from the config,
    # not from the inputs).
    off, n_local, n_all = cfg["local_expert_offset"], cfg["num_local_experts"], cfg["num_experts"]
    key = (off, n_local, n_all)
    if key not in state["expert_maps"]:
        m = torch.full((n_all,), -1, dtype=torch.int32)
        m[off:off + n_local] = torch.arange(n_local, dtype=torch.int32)
        state["expert_maps"][key] = m.to(device)
    return state["expert_maps"][key]


def kernel_fn(data, state):
    from sgl_kernel.scalar_type import scalar_types

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        _map_topk_ids_to_local_experts,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    cfg = data["config"]
    hidden_states = data["hidden_states"].contiguous()
    topk_ids = data["topk_ids"]
    topk_scores = data["topk_weights"]
    num_tokens, size_k = hidden_states.shape
    topk = topk_ids.shape[1]
    size_n = cfg["intermediate_size"]
    gemm1_n = 2 * size_n
    n_local = cfg["num_local_experts"]
    off = cfg["local_expert_offset"]
    expert_map = _expert_map(state, cfg, hidden_states.device)

    # This GPU's experts: a view of the repacked tensors (the layer holds only these).
    w13 = state["w13"][off:off + n_local]
    w2 = state["w2"][off:off + n_local]
    w13_scale = state["w13_scale"][off:off + n_local]
    w2_scale = state["w2_scale"][off:off + n_local]
    w13_scale_2 = state["w13_scale_2"][off:off + n_local]
    w2_scale_2 = state["w2_scale_2"][off:off + n_local]

    local_topk_ids = _map_topk_ids_to_local_experts(topk_ids.to(torch.int32), expert_map)
    local_topk_ids = local_topk_ids.contiguous()
    topk_weights = topk_scores.to(torch.float32).contiguous()

    for block_size_m in _BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * topk / n_local / block_size_m < 0.9:
            break

    # With an expert map the layer aligns over the global expert count and does not use the
    # single-token aligner or the inactive-prefix count.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        local_topk_ids, block_size_m, cfg["num_experts"], return_inactive_prefix_count=False
    )

    workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)
    cache2 = torch.empty((num_tokens * topk, size_n), device=hidden_states.device,
                         dtype=hidden_states.dtype)
    cache13 = torch.empty((num_tokens * topk * max(gemm1_n, size_k),),
                          device=hidden_states.device, dtype=hidden_states.dtype)
    cache1 = cache13[: num_tokens * topk * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * topk * size_k].view(-1, size_k)

    cache1 = moe_wna16_marlin_gemm(
        hidden_states, cache1, w13, None, w13_scale, w13_scale_2, None, None, None, workspace,
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
        moe_block_size=block_size_m, top_k=topk, mul_topk_weights=False, is_ep=True,
        b_q_type=scalar_types.float4_e2m1f, size_m=num_tokens, size_n=gemm1_n, size_k=size_k,
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        expert_ids_are_sorted=True, has_inactive_prefix_count=False,
    )
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_topk_ids.view(-1),
                 expert_step=1)
    expert_outputs = moe_wna16_marlin_gemm(
        cache2, cache3, w2, None, w2_scale, w2_scale_2, None, None, None, workspace,
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
        moe_block_size=block_size_m, top_k=1, mul_topk_weights=False, is_ep=True,
        b_q_type=scalar_types.float4_e2m1f, size_m=num_tokens * topk, size_n=size_k,
        size_k=size_n, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
        is_zp_float=False, expert_ids_are_sorted=True, has_inactive_prefix_count=False,
    ).view(num_tokens, topk, size_k)

    return fused_post_expert_norm_weighted_sum(
        expert_outputs, topk_scores, norm_weight=state["norm_w"], eps=cfg["post_norm_eps"],
        weight_first=True, slot_expert_ids=local_topk_ids,
    )
