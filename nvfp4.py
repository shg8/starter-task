"""NVFP4: the weight format of the experts. This file is the exact definition.

A weight matrix W [N, K] (N output rows, K inputs) is stored as three tensors:

  packed  uint8 [N, K // 2]        two 4-bit codes per byte: element 2i in the low nibble,
                                   element 2i + 1 in the high nibble
  sf      float8_e4m3fn [N, K // 16]  one block scale per 16 consecutive values of a row
  gs      float32 (one per expert)    one global scale per expert

and the value of element (n, k) is

  W[n, k] = E2M1[code(n, k)] * float(sf[n, k // 16]) * gs

A 4-bit code is a sign bit (bit 3) and a 3-bit magnitude index (bits 0-2) into
E2M1_MAGNITUDES = (0, 0.5, 1, 1.5, 2, 3, 4, 6). Code 0b1000 is -0.

For the experts, w13 is [E, 8192, 2048] (rows 0..4095 are the gate projection, rows 4096..8191
the up projection, K = 4096) and w2 is [E, 4096, 2048] (K = 4096). Each expert has one gs for
w13 and one for w2.
"""

from __future__ import annotations

import torch

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BLOCK = 16
FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0


def e2m1_table(device=None, dtype=torch.float32) -> torch.Tensor:
    """The 16 values of the 4-bit codes, indexed by code."""
    mags = list(E2M1_MAGNITUDES)
    return torch.tensor(mags + [-m for m in mags], dtype=dtype, device=device)


def pack(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [..., K] (values 0..15) -> uint8 [..., K // 2]."""
    assert codes.dtype == torch.uint8 and codes.shape[-1] % 2 == 0
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K // 2] -> uint8 codes [..., K]."""
    assert packed.dtype == torch.uint8
    lo = packed & 0x0F
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def dequantize(packed: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor | float) -> torch.Tensor:
    """Exact float32 values of one or more NVFP4 matrices.

    packed [..., N, K // 2], sf [..., N, K // 16], gs a float or a tensor broadcastable to the
    leading dimensions (e.g. shape [E] for packed [E, N, K // 2]).
    """
    codes = unpack(packed).long()
    vals = e2m1_table(packed.device)[codes]
    scale = sf.float().repeat_interleave(BLOCK, dim=-1)
    out = vals * scale
    if isinstance(gs, torch.Tensor):
        out = out * gs.float().reshape(*gs.shape, *([1] * (out.dim() - gs.dim())))
    else:
        out = out * gs
    return out


def quantize(w: torch.Tensor, gs: torch.Tensor | None = None):
    """Quantize float weights [..., N, K] of ONE expert to NVFP4, the way ModelOpt does.

    gs = amax(|w|) / (448 * 6) unless given; block scale = e4m3(block amax / 6 / gs);
    code = nearest E2M1 value of w / (block scale * gs). Returns (packed, sf, gs).
    """
    w = w.float()
    if gs is None:
        gs = w.abs().amax() / (FP8_E4M3_MAX * E2M1_MAX)
    gs = torch.as_tensor(gs, dtype=torch.float32, device=w.device)
    blocks = w.unflatten(-1, (-1, BLOCK))
    amax = blocks.abs().amax(dim=-1)
    sf = (amax / E2M1_MAX / gs).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    denom = (sf.float() * gs).unsqueeze(-1)
    scaled = torch.where(denom > 0, blocks / denom.clamp_min(1e-30), torch.zeros_like(blocks))
    mag = scaled.abs().clamp(max=E2M1_MAX)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=w.device)
    idx = torch.bucketize(mag, mids).to(torch.uint8)
    codes = idx | ((scaled < 0).to(torch.uint8) << 3)
    codes = torch.where(idx == 0, torch.zeros_like(codes), codes)  # no -0
    return pack(codes.flatten(-2)), sf, gs


def random_expert(n: int, k: int, generator: torch.Generator, device, std: float = 0.02):
    """Random float weights [n, k] with realistic structure, then quantized to NVFP4.

    Rows get a log-normal scale (sigma 0.25) and every value a Gaussian scale mixture (heavier
    tails than a Gaussian), so the block scales spread over about two octaves as in trained
    checkpoints. Returns (packed, sf, gs).
    """
    row = torch.exp(0.25 * torch.randn(n, 1, generator=generator, device=device))
    z = torch.randn(n, k, generator=generator, device=device)
    z = z * torch.exp(0.3 * torch.randn(n, k, generator=generator, device=device))
    return quantize(z * row * std)


def random_experts(num_experts: int, n: int, k: int, seed: int, device):
    """NVFP4 weights for num_experts experts: packed [E, n, k//2], sf [E, n, k//16], gs [E].

    Expert e depends only on (seed, e), so any subset can be regenerated identically.
    """
    packed = torch.empty(num_experts, n, k // 2, dtype=torch.uint8, device=device)
    sf = torch.empty(num_experts, n, k // BLOCK, dtype=torch.float8_e4m3fn, device=device)
    gs = torch.empty(num_experts, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    for e in range(num_experts):
        gen.manual_seed(seed * 100003 + e)
        p, s, g = random_expert(n, k, gen, device)
        packed[e], sf[e], gs[e] = p, s, g
    return packed, sf, gs


if __name__ == "__main__":
    # Self-check: pack/unpack round trip, and quantize -> dequantize error of a Gaussian.
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (4, 64), generator=g, dtype=torch.uint8)
    assert torch.equal(unpack(pack(codes)), codes)
    assert pack(torch.tensor([[1, 2]], dtype=torch.uint8)).item() == 0x21
    w = torch.randn(64, 256, generator=g)
    p, s, gs = quantize(w)
    err = (dequantize(p, s, gs) - w).norm() / w.norm()
    print(f"pack/unpack ok; NVFP4 relative error of a Gaussian matrix: {err:.4f}")
