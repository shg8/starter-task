r"""The file you edit: the expert block of the MoE layer for one GPU.

The judge calls
    state = prepare(weights)          once, untimed, at most 300 s
    out = kernel_fn(data, state)      captured as a CUDA graph and replayed with fresh inputs

This starting version is plain PyTorch. It is correct and slow: for each of the 32 local experts
it decodes the full NVFP4 weights to bfloat16 and multiplies every token, routed or not.

Rules (see README.md): CUDA-graph safe (no .item(), .cpu(), synchronize, no allocation or
branching that depends on input values; branching on shapes and config is fine); nothing derived
from the inputs is kept between calls; the weights stay 4-bit (prepare may repack them
losslessly); activations stay bfloat16 where SGLang keeps them bfloat16; CUDA C++ goes in this
file and is built with torch.utils.cpp_extension.load_inline; the file does not import the judge
or the reference, start processes or use the network.

Adding a CUDA C++ kernel
-----------------------
Start by replacing the routing-weight calculation for one expert. The example
below assigns one CUDA thread per token and sums the weights of its matching
routes. Copy the helper into this file, add
`state["cuda_ops"] = _build_cuda_ops()` before returning from prepare(), and
replace the line that assigns `g` in kernel_fn() with
`g = state["cuda_ops"].route_weights(ids.contiguous(), tw.contiguous(), e)`.

    def _build_cuda_ops():
        from torch.utils.cpp_extension import load_inline

        # load_inline generates the Python binding from this declaration.
        cpp = "at::Tensor route_weights(at::Tensor ids, at::Tensor weights, int64_t expert);"
        cuda = r'''
        #include <c10/cuda/CUDAGuard.h>
        #include <c10/cuda/CUDAException.h>
        #include <c10/cuda/CUDAStream.h>

        __global__ void route_weights_kernel(
            const int32_t* ids, const float* weights, float* out,
            int64_t tokens, int64_t top_k, int64_t expert) {
            const int64_t token = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (token >= tokens) return;
            float sum = 0.0f;
            for (int64_t slot = 0; slot < top_k; ++slot) {
                const int64_t index = token * top_k + slot;
                if (ids[index] == expert) sum += weights[index];
            }
            out[token] = sum;
        }

        // Inputs are contiguous CUDA tensors on the same device, as passed above.
        at::Tensor route_weights(at::Tensor ids, at::Tensor weights, int64_t expert) {
            const c10::cuda::CUDAGuard guard(ids.device());
            const int64_t tokens = ids.size(0);
            auto out = at::empty({tokens}, weights.options());
            if (tokens == 0) return out;

            constexpr int threads = 128;
            const auto blocks = (tokens + threads - 1) / threads;
            const auto stream = c10::cuda::getCurrentCUDAStream(ids.get_device());
            route_weights_kernel<<<blocks, threads, 0, stream.stream()>>>(
                ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
                out.data_ptr<float>(), tokens, ids.size(1), expert);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return out;
        }
        '''
        return load_inline(
            name="starter_route_weights",
            cpp_sources=cpp,
            cuda_sources=cuda,
            functions=["route_weights"],
        )

Build in prepare() so compilation finishes before graph capture. Launch on
PyTorch's current stream so the kernel participates in capture. The output
allocation and grid depend only on the input shape. Route IDs are read and
compared on the GPU, and the launcher returns without synchronizing.

From the repository root on Cayenne, check slurm-windows and run the eager
and CUDA graph correctness tests with one GPU.

    sbatch --partition=main --gres=gpu:1 --time=00:20:00 \\
        --wrap='./run.sh submission.py --tests-only'

PyTorch load_inline reference:
https://docs.pytorch.org/docs/stable/cpp_extension.html
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch


class Weights(TypedDict):
    """All 256 experts, NVFP4 (see nvfp4.py). Given to prepare() once."""
    w13: torch.Tensor        # uint8 [256, 8192, 2048]: rows 0..4095 gate, 4096..8191 up; K = 4096
    w13_sf: torch.Tensor     # float8_e4m3fn [256, 8192, 256]: one scale per 16 values
    w13_gs: torch.Tensor     # float32 [256]: one scale per expert
    w2: torch.Tensor         # uint8 [256, 4096, 2048]: K = 4096
    w2_sf: torch.Tensor      # float8_e4m3fn [256, 4096, 256]
    w2_gs: torch.Tensor      # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [4096]: the post-expert norm uses (1 + post_norm_w)


class Config(TypedDict):
    """Static per graph: the judge captures one graph per config."""
    M: int                    # tokens in this step
    local_expert_offset: int  # this GPU holds experts [offset, offset + 32)
    num_local_experts: int    # 32
    num_experts: int          # 256
    top_k: int                # 8
    hidden_size: int          # 4096
    intermediate_size: int    # 4096
    post_norm_eps: float      # 1e-8


class Inputs(TypedDict):
    """New on every call."""
    hidden_states: torch.Tensor  # bfloat16 [M, 4096]
    topk_ids: torch.Tensor       # int32 [M, 8], expert ids 0..255 (distinct within a row)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


# Values of the 16 E2M1 codes (bit 3 is the sign).
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def prepare(weights: Weights) -> Any:
    """Runs once. May repack the weights into any lossless 4-bit layout. Returns the state."""
    device = weights["w13"].device
    state = dict(weights)
    state["e2m1"] = torch.tensor(_E2M1, dtype=torch.bfloat16, device=device)
    state["norm_w"] = 1.0 + weights["post_norm_w"].float()       # float32 [4096]
    return state


def _decode(packed: torch.Tensor, sf: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """One expert's NVFP4 matrix without its global scale, as bfloat16 [N, K].

    code * block scale is exact in bfloat16 (at most 5 significant bits), so this is lossless.
    """
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    codes = torch.stack((lo, hi), dim=-1).flatten(-2)                 # [N, K]
    vals = table[codes]                                              # bfloat16 [N, K]
    scale = sf.to(torch.bfloat16).repeat_interleave(16, dim=-1)      # [N, K]
    return vals * scale


def kernel_fn(data: Inputs, state: Any) -> torch.Tensor:
    """Returns bfloat16 [M, 4096]."""
    x = data["hidden_states"]
    ids = data["topk_ids"]
    tw = data["topk_weights"]
    cfg = data["config"]
    M, inter, eps = cfg["M"], cfg["intermediate_size"], cfg["post_norm_eps"]
    off = cfg["local_expert_offset"]

    out = torch.zeros(M, cfg["hidden_size"], dtype=torch.float32, device=x.device)
    for e in range(off, off + cfg["num_local_experts"]):
        # Routing weight of expert e for every token (0 where the token does not use it).
        g = (tw * (ids == e)).sum(dim=1)                             # float32 [M]

        w13 = _decode(state["w13"][e], state["w13_sf"][e], state["e2m1"])
        h = (x @ w13.T).float() * state["w13_gs"][e]                 # [M, 8192]
        a = (torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]).to(torch.bfloat16)
        del w13

        w2 = _decode(state["w2"][e], state["w2_sf"][e], state["e2m1"])
        y = ((a @ w2.T).float() * state["w2_gs"][e]).to(torch.bfloat16)  # [M, 4096]
        del w2

        y = y.float() * state["norm_w"]
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
        out += g.unsqueeze(1) * y
    return out.to(torch.bfloat16)
