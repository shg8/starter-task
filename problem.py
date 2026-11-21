"""The problem: shapes, weight and input generation, the FP32 reference, the checks, the floor.

The judge imports this file; a submission must not.
"""

from __future__ import annotations

import math

import torch

import nvfp4

HIDDEN = 4096            # model width seen by the experts (K of the first projection)
INTER = 4096             # expert intermediate width (gate and up are INTER each)
NUM_EXPERTS = 256
NUM_LOCAL = 32           # experts held by this GPU: [offset, offset + 32)
NUM_GPUS = NUM_EXPERTS // NUM_LOCAL
TOP_K = 8
POST_NORM_EPS = 1e-8

SIZES = (1, 8, 16, 32, 128, 256)     # timed batch sizes (tokens per step)
TIMING_OFFSET = 96                    # the timed GPU holds experts 96..127

# Tests: (name, tokens, first local expert, Zipf exponent of expert popularity, seed)
ZIPF_S = 0.8          # normal skew
ZIPF_S_HEAVY = 1.6    # heavily skewed routing
TESTS = (
    ("1 token", 1, 0, ZIPF_S, 101),
    ("4 tokens", 4, 224, ZIPF_S, 102),
    ("33 tokens", 33, 96, ZIPF_S, 103),
    ("128 tokens, heavily skewed", 128, 32, ZIPF_S_HEAVY, 104),
    ("257 tokens", 257, 160, ZIPF_S, 105),
)

WEIGHT_SEED = 7
REL_L2_BOUND = 0.015           # whole output, against the FP32 reference
ROW_FACTOR = 5.0               # no row more than 5x over the bound
NEAR_ZERO_ROW = 0.1            # a row's error is measured against max(|ref row|, 0.1 * mean |ref row|)

# Floor model
HBM_BYTES_PER_S = 8.0e12
TENSOR_FLOPS = 2.25e15
EXPERT_BYTES = (2 * INTER * HIDDEN // 2 + 2 * INTER * HIDDEN // 16      # w13 codes + block scales
                + HIDDEN * INTER // 2 + HIDDEN * INTER // 16             # w2 codes + block scales
                + 2 * 4)                                                  # two global scales
FLOPS_PER_ROUTE = 2 * (HIDDEN * 2 * INTER + INTER * HIDDEN)


# ----------------------------------------------------------------------------------------------
# Weights


def make_weights(device="cuda", seed: int = WEIGHT_SEED) -> dict:
    """All 256 experts in NVFP4 plus the post-expert norm weight (see submission.Weights)."""
    w13, w13_sf, w13_gs = nvfp4.random_experts(NUM_EXPERTS, 2 * INTER, HIDDEN, seed, device)
    w2, w2_sf, w2_gs = nvfp4.random_experts(NUM_EXPERTS, HIDDEN, INTER, seed + 1, device)
    gen = torch.Generator(device=device).manual_seed(seed + 2)
    post_norm_w = (0.1 * torch.randn(HIDDEN, generator=gen, device=device)).to(torch.bfloat16)
    return {
        "w13": w13, "w13_sf": w13_sf, "w13_gs": w13_gs,
        "w2": w2, "w2_sf": w2_sf, "w2_gs": w2_gs,
        "post_norm_w": post_norm_w,
    }


def weights_nbytes(weights: dict) -> int:
    return sum(t.numel() * t.element_size() for t in weights.values())


# ----------------------------------------------------------------------------------------------
# Inputs


_CHANNEL_SCALE = {}


def _channel_scale(device):
    # Fixed per-channel magnitudes of the normalized activations (log-normal, sigma 0.5).
    key = str(device)
    if key not in _CHANNEL_SCALE:
        g = torch.Generator().manual_seed(1234)
        _CHANNEL_SCALE[key] = torch.exp(0.5 * torch.randn(HIDDEN, generator=g)).to(device)
    return _CHANNEL_SCALE[key]


def make_routing(tokens: int, offset: int, zipf_s: float, seed: int):
    """Top-8 of 256 experts per token, on the CPU.

    Expert popularity is Zipf-like (p ~ 1 / rank^s) over a fresh random ranking of the experts.
    Each token picks 8 distinct experts with probability proportional to popularity
    (Gumbel top-k). Then the experts of the GPU that received the most routes are swapped with
    this GPU's experts, so this GPU is always the busiest of the eight.
    Returns topk_ids int32 [tokens, 8] and topk_weights float32 [tokens, 8].
    """
    g = torch.Generator().manual_seed(seed)
    rank = torch.randperm(NUM_EXPERTS, generator=g)
    logp = -zipf_s * torch.log(rank.double() + 1.0)
    u = torch.rand(tokens, NUM_EXPERTS, generator=g, dtype=torch.float64).clamp_(1e-12, 1 - 1e-12)
    gumbel = -torch.log(-torch.log(u))
    ids = torch.topk(logp + gumbel, TOP_K, dim=1).indices
    counts = torch.bincount((ids // NUM_LOCAL).flatten(), minlength=NUM_GPUS)
    mine = offset // NUM_LOCAL
    busiest = int(torch.argmax(counts))
    if counts[busiest] > counts[mine]:
        grp = ids // NUM_LOCAL
        ids = torch.where(grp == busiest, ids + (mine - busiest) * NUM_LOCAL,
                          torch.where(grp == mine, ids + (busiest - mine) * NUM_LOCAL, ids))
    logits = torch.randn(tokens, TOP_K, generator=g)
    weights = torch.softmax(logits, dim=1)
    return ids.to(torch.int32), weights.to(torch.float32)


def make_config(tokens: int, offset: int) -> dict:
    return {
        "M": tokens,
        "local_expert_offset": offset,
        "num_local_experts": NUM_LOCAL,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "post_norm_eps": POST_NORM_EPS,
    }


def make_case(tokens: int, seed: int, offset: int = TIMING_OFFSET, zipf_s: float = ZIPF_S,
              device="cuda") -> dict:
    """One fresh call: {"hidden_states", "topk_ids", "topk_weights", "config"}."""
    ids, w = make_routing(tokens, offset, zipf_s, seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=gen, device=device) * _channel_scale(device)
    return {
        "hidden_states": x.to(torch.bfloat16),
        "topk_ids": ids.to(device),
        "topk_weights": w.to(device),
        "config": make_config(tokens, offset),
    }


# ----------------------------------------------------------------------------------------------
# Reference


@torch.no_grad()
def reference(data: dict, weights: dict) -> torch.Tensor:
    """FP32 reference output [M, 4096] (float32). Dequantizes only the experts that are used."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        cfg = data["config"]
        off = cfg["local_expert_offset"]
        x = data["hidden_states"].float()
        ids = data["topk_ids"].long()
        tw = data["topk_weights"].float()
        norm_w = 1.0 + weights["post_norm_w"].float()
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
        for e in torch.unique(ids).tolist():
            if not off <= e < off + NUM_LOCAL:
                continue
            tok, slot = torch.nonzero(ids == e, as_tuple=True)
            w13 = nvfp4.dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e])
            h = x[tok] @ w13.T
            del w13
            a = torch.nn.functional.silu(h[:, :INTER]) * h[:, INTER:]
            w2 = nvfp4.dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e])
            y = (a @ w2.T) * norm_w
            del w2
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
            out.index_add_(0, tok, y * tw[tok, slot].unsqueeze(1))
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def check(out, ref: torch.Tensor) -> tuple[bool, str, dict]:
    """The correctness check. Returns (ok, message, numbers)."""
    M = ref.shape[0]
    if not isinstance(out, torch.Tensor):
        return False, f"output is a {type(out).__name__}, not a tensor", {}
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (M, HIDDEN):
        return False, f"output is {out.dtype} {tuple(out.shape)}, expected bfloat16 {(M, HIDDEN)}", {}
    o = out.float()
    if not torch.isfinite(o).all():
        return False, "output contains NaN or Inf", {}
    diff = o - ref
    rel = float(diff.norm() / ref.norm().clamp_min(1e-30))
    row_ref = ref.norm(dim=1)
    denom = torch.maximum(row_ref, NEAR_ZERO_ROW * row_ref.mean()).clamp_min(1e-30)
    row_err = diff.norm(dim=1) / denom
    worst = int(torch.argmax(row_err))
    worst_err = float(row_err[worst])
    nums = {"rel_l2": rel, "worst_row": worst, "worst_row_rel": worst_err}
    if rel > REL_L2_BOUND:
        return False, f"relative L2 error {rel:.4%} > {REL_L2_BOUND:.1%}", nums
    if worst_err > ROW_FACTOR * REL_L2_BOUND:
        return False, (f"row {worst} has relative error {worst_err:.4%} > "
                       f"{ROW_FACTOR:g} x {REL_L2_BOUND:.1%}"), nums
    return True, f"relative L2 {rel:.4%}, worst row {worst_err:.4%}", nums


# ----------------------------------------------------------------------------------------------
# Floor


def floor(data: dict) -> dict:
    """Lower bound on the step's time: bytes it must move at 8 TB/s or its arithmetic at
    2.25 PFLOP/s (dense BF16), whichever is longer.

    Bytes: codes, block scales and global scales of every local expert with at least one route,
    plus the activations (hidden_states in, routing in, output out). Arithmetic: two GEMMs per
    local route (4096 -> 8192 and 4096 -> 4096), 2 FLOP per multiply-add.
    """
    cfg = data["config"]
    off, M = cfg["local_expert_offset"], cfg["M"]
    ids = data["topk_ids"].cpu()
    local = (ids >= off) & (ids < off + NUM_LOCAL)
    routes = int(local.sum())
    experts = int(torch.unique(ids[local]).numel())
    nbytes = experts * EXPERT_BYTES + M * HIDDEN * 2 * 2 + M * TOP_K * 8
    flops = routes * FLOPS_PER_ROUTE
    t = max(nbytes / HBM_BYTES_PER_S, flops / TENSOR_FLOPS)
    return {"floor_us": t * 1e6, "bytes": nbytes, "flops": flops, "local_routes": routes,
            "local_experts": experts}


if __name__ == "__main__":
    # Routing statistics of the timed cases (CPU only).
    for M in SIZES + (128,):
        s = ZIPF_S
        rs, es = [], []
        for seed in range(200):
            ids, _ = make_routing(M, TIMING_OFFSET, s, seed)
            local = (ids >= TIMING_OFFSET) & (ids < TIMING_OFFSET + NUM_LOCAL)
            rs.append(int(local.sum()))
            es.append(int(torch.unique(ids[local]).numel()))
        rs, es = torch.tensor(rs).float(), torch.tensor(es).float()
        print(f"M={M:4d} s={s}: local routes {rs.mean():.1f} ({rs.min():.0f}-{rs.max():.0f}), "
              f"distinct local experts {es.mean():.1f} ({es.min():.0f}-{es.max():.0f})")
