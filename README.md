# Starter Task

Over two weeks, use coding agents to optimize a production GPU kernel. Have the agents
write the code while you choose what to try, direct the profiling, and verify their
explanations. By the walkthrough, you should be able to explain how your kernel works,
what limits its performance, and where your understanding stops. You can start without
a GPU performance background.

## The kernel

You'll optimize the NVFP4 expert block in Muse Spark Open, a mixture-of-experts
language model. The judge runs the work for one of eight expert shards on a single
B200, without the cross-GPU reduction. The baseline calls SGLang's Marlin W4A16
implementation. [SGLang layer](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py#L520) · [Marlin CUDA kernel](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h)

Each token is routed to 8 of the model's 256 experts. This GPU holds 32 of them and computes only the routes
that land on its own experts:

```python
for t, e, g in local_routes:               # token, expert, routing weight
    h = x[t] @ dequant(w13[e]).T              # 4,096 -> 8,192
    gate, up = split(h)
    y = (silu(gate) * up) @ dequant(w2[e]).T  # 4,096 -> 4,096
    y = bf16(y) * (1 + post_norm_w)
    out[t] += g * rmsnorm(y)                  # eps = 1e-8, no weight of its own
```

**Inputs**

| Name | Type and shape | Meaning |
|---|---|---|
| `hidden_states` | bfloat16 [M, 4096] | the M tokens of this step |
| `topk_ids` | int32 [M, 8] | each token's 8 experts, as ids 0–255 |
| `topk_weights` | float32 [M, 8] | the routing weight of each |
| `config` | dict | M, and which 32 experts are local |

The output is bfloat16 [M, 4096]. A token with no local expert gets a row of zeros.

**Weights, given once**

| Name | Type and shape | Meaning |
|---|---|---|
| `w13` | uint8 [256, 8192, 2048] | first projection, two 4-bit values per byte |
| `w13_sf` | FP8 [256, 8192, 256] | one scale per 16 values |
| `w13_gs` | float32 [256] | one scale per expert |
| `w2`, `w2_sf`, `w2_gs` | [256, 4096, 2048], [256, 4096, 256], [256] | second projection, same format |
| `post_norm_w` | bfloat16 [4096] | the norm's weight |

- **The weight format is NVFP4.** A value is `code × block scale × expert scale`. The 4-bit code is a sign and
  one of 0, 0.5, 1, 1.5, 2, 3, 4, 6. `nvfp4.py` in the repo is the exact definition.
- **One expert is about 28 MB.**
- **`prepare(weights)`** runs once, untimed, for up to 300 s. The weights may be repacked there into any
  lossless layout, but they must stay 4-bit: the state `prepare` returns may hold at most as much GPU memory
  as the weights it was given (6.75 GiB) plus 1 GiB.
- **All 256 experts are given to `prepare`;** which 32 are local is in `config` (`local_expert_offset`). The
  timed GPU holds experts 96–127; the tests use other groups too.
- **Routing is skewed,** as real routers are, and every test case makes this GPU the busiest of the eight.

### Batch sizes and performance floor

The judge times batches of 1, 8, 16, 32, 128, and 256 tokens. For each input,
it estimates a floor as the larger of two times:

- **Memory.** Divide the bytes for the active local experts' NVFP4 codes and scales, input and output activations, and routing data by 8 TB/s.
- **Compute.** Divide the FLOPs for both projections across all local routes by 2.25 PFLOP/s, the B200's peak dense BF16 Tensor Core rate. This models the W4A16 baseline's arithmetic.

These ceilings are per GPU. [NVIDIA B200 bandwidth](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) · [NVIDIA HGX B200 Tensor Core specifications](https://www.nvidia.com/en-us/data-center/hgx/)

Slurm job 24601 measured the starting `submission.py` against SGLang on one B200
on Cayenne. The table shows medians from 100 timed pairs per size. The SGLang/floor
column uses the median of the per-pair ratios.

| Tokens | SGLang (µs) | Starter (µs) | B200 floor (µs) | SGLang / floor | Local routes | Local experts used |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59.4 | 27690.0 | 7.1 | 8.09× | 2 | 2 |
| 8 | 114.7 | 27845.6 | 31.9 | 3.60× | 14 | 9 |
| 16 | 151.5 | 28122.3 | 42.5 | 3.51× | 25 | 12 |
| 32 | 209.0 | 28105.1 | 60.2 | 3.50× | 49 | 17 |
| 128 | 537.6 | 28366.3 | 102.9 | 5.32× | 193 | 29 |
| 256 | 744.9 | 28998.1 | 110.2 | 6.72× | 383.5 | 31 |

The score is the geometric mean of the six per-size speedups. The starter scored
0.007× with a 95% interval of [0.007, 0.008].

**Correctness**

- The output's relative L2 error against an FP32 reference is at most 1.5%.
- No single row is more than 5× over that bound, and there is no NaN or Inf. (A row whose reference is near
  zero is measured against a tenth of the average row.)
- Five tests run before any timing: 1, 4, 33, 128 (with heavily skewed routing) and 257 tokens. Each test
  calls `kernel_fn` directly, then captures it as a CUDA graph and replays it on two new inputs. Every 4th
  timed call (and the first) is checked as well.

**How a kernel is timed**

- Each batch size is captured as a CUDA graph and timed by replaying it.
- Every replay starts with a cold L2 cache (the judge reads a 512 MiB buffer first).
- Your replays and SGLang's take turns on the same fresh inputs, on the same GPU, each side in its own
  process. The speedup is the median of the paired ratios, reported with a 95% interval of at least ±1.5%.
- The judge measures each process's fixed graph replay overhead and subtracts it from its timings.

**Constraints**

- **Graph-safe.** The timed call is replayed as a CUDA graph: no host syncs (`.item()`, `.cpu()`,
  `torch.cuda.synchronize()`), and no allocation or branching on input values. Branching on shapes and
  `config` is fine.
- **Fresh inputs.** Nothing derived from the inputs may be kept between calls.
- **Precision.** Weights stay NVFP4 (decoding them inside the kernel is fine). Activations stay bfloat16
  wherever SGLang keeps them bfloat16.
- **One file.** CUDA C++ inside a single Python file, built with `torch.utils.cpp_extension.load_inline`.
- **Hands off the judge.** The submission does not touch timing, import the reference, start processes or use
  the network.

## Run the judge on Cayenne

From the repository root, check the queue and submit a job for one B200.
The first run may need to pull the container image and compile SGLang's kernels.

```bash
slurm-windows
sbatch --partition=main --gres=gpu:1 --time=00:20:00 \
    --wrap='./run.sh submission.py --repeats 100 --json b200-results.json'
squeue -u "$USER"
```

Slurm writes the judge's output to `slurm-<job-id>.out`. The JSON file includes
every timed pair. To reproduce the baseline on both sides, use
`./run.sh baseline.py --repeats 100` as the batch command.

The launcher mounts the repository at `/work` in the pinned SGLang v0.5.23 image,
`lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`.
The container gets its GPU from the Slurm allocation. JIT caches persist under
`${XDG_CACHE_HOME:-$HOME/.cache}/starter-task`. Inside the container, you can
call `python3 judge.py` directly.

Change the command inside `--wrap` to run a shorter check, compare two implementations,
or collect a profile.

```bash
./run.sh submission.py --tests-only
./run.sh submission.py --sizes 1,8
./run.sh mine.py --baseline other.py
./run.sh submission.py --profile 8
```

By default, the judge starts checking for convergence after 100 timed pairs per
size. It aims for a bootstrap interval within ±0.5% and stops at 400 pairs or
once it reaches the 40-second timing budget. Use `--repeats` to set a fixed pair count.

### Profile with Nsight Compute

Inside a one-GPU allocation, run:

```bash
./run.sh --exec ncu --profile-from-start off --set full \
    -o prof python3 judge.py submission.py --ncu 8 --iters 1
```

The judge brackets the selected calls with `cudaProfilerStart` and
`cudaProfilerStop`. Nsight Compute needs access to GPU performance counters.
If clock locking fails, add `--clock-control none` to profile at normal boost
clocks. Use `-k regex:<name>` to limit collection to your kernel.

## Deliverables

1. **Code and writeup for the two checkpoints**. Each is a git tag whose `submission.py` passes every
   test, and each tests a distinct optimization idea. Use at most one page per checkpoint to cover the following.
   - what changed;
   - why it is faster, in terms of what the hardware is doing;
   - one profiler number that moved, before and after;
   - what limits the kernel now, and how far it is from the theoretical limit;
   - which parts of the kernel you can explain line by line, and which you can't.
2. **A 30-minute in-person discussion** about your work and what you learned about GPU architecture.

## A suggested pace

| Days | Aim |
|---|---|
| 1–2 | Run the judge on the starting submission and the baseline. Write the measurement note. |
| 3–5 | First CUDA kernel that passes the tests. Profile it and SGLang's. Checkpoint 1. |
| 6–10 | Find what limits checkpoint 1 and improve it -> Checkpoint 2. |
| 11-12 | Write-up |

## Goals

Roughly in order of importance:
1. **You understand your kernel.** You can walk through the kernel code at both checkpoints and explain what it does and what limits it,
   and you know where your understanding stops.
2. **You learned how to profile kernels.**
3. **You can characterize how agents approach writing kernels and have thoughts on their strengths and weaknesses.**
4. **The kernel got faster.**

## Questions about GPU architecture

You probably have to pick up a little GPU programming knowledge for this task. By the walkthrough you should be able to
answer some of these for your own code. It's much better to be able to explain a few of these in-depth rather than all of them superficially.

- How a kernel launch maps onto the GPU: grids, blocks, threads, warps, and how many run at once on one SM.
- Where data lives: registers, shared memory, the L2 cache, HBM. What it costs to move between them.
- Which weight bytes one step has to read, and what a kernel must do to read them at close to 8 TB/s.
- How a 4-bit weight becomes a number, and what that costs in instructions per byte.
- Why `prepare()` may repack the weights, and what a different layout changes.
- Why the sums are accumulated in FP32 when the inputs and outputs are bfloat16.
- What a tensor core does, and at which batch size it starts to matter.
- What a CUDA graph is, and why launch overhead shows up at these time scales.
- How to read a Nsight Compute (`ncu`) report: which numbers say a kernel is waiting on memory, and which say
  it is short of instructions.

Good starting points are NVIDIA's CUDA C++ Programming Guide, the CUDA C++ Best Practices Guide and the
Nsight Compute documentation. Your agents will suggest more; check what they tell you against these.

## Credits

`baseline.py` calls SGLang (https://github.com/sgl-project/sglang, Apache License 2.0); all the kernels it runs
are SGLang's.
