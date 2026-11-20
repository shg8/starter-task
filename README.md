# Starter Task

Over two weeks you will take one production GPU kernel, have coding agents make it faster, and
reach an understanding of how your agents improved the kernel. The agents should write all the code. Your part is to steer and understand:
decide what to try, have to profile it, check what the agent tells you, and understand the result. No GPU
performance background is assumed, and we'd like to see how quickly you can pick up new concepts. By the end you should know this kernel well,
and know the limitations of your understanding.

## The kernel

The expert block of a mixture-of-experts language model (Muse Spark Open), as the
[SGLang](https://github.com/sgl-project/sglang) inference server runs it on one of eight H200s. SGLang's own
implementation is the baseline you are timed against:

- the layer, [`MuseSpark1xFp4Experts._forward_marlin`](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py#L520);
- the CUDA kernel under it, [`marlin_moe/marlin_template.h`](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h).

Each token is routed to 8 of the model's 256 experts. This GPU holds 32 of them and computes only the routes
that land on its own experts:

```python
for t, (e, g) in routes if e is local:       # token t, expert e, routing weight g
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

**Batch sizes and the baseline**

M is 1, 8, 16, 32, 128 or 256 tokens. The floor is the time to read the data the step needs at 4.8 TB/s or do
its arithmetic at 989 TFLOP/s, whichever is longer: the codes and scales of every local expert that receives
at least one route, plus the activations. SGLang's times were measured with the judge in the repo on one H200
(median over 100 timed calls per size; the floor is the median over the same calls).

| Tokens | SGLang | Floor | SGLang / floor | Local routes | Local experts used |
|---|---|---|---|---|---|
| 1 | 55.8 µs | 11.8 µs | 4.6× | 2 | 2 |
| 8 | 123.9 µs | 53.1 µs | 2.4× | 14 | 9 |
| 16 | 165.7 µs | 70.8 µs | 2.2× | 25 | 12 |
| 32 | 219.5 µs | 100.4 µs | 2.3× | 49 | 17 |
| 128 | 588.6 µs | 171.5 µs | 3.5× | 193 | 29 |
| 256 | 814.7 µs | 183.7 µs | 4.4× | 384 | 31 |

The score is the geometric mean, over the six batch sizes, of SGLang's time divided by yours.

**Correctness**

- The output's relative L2 error against an FP32 reference is at most 1.5%.
- No single row is more than 5× over that bound, and there is no NaN or Inf. (A row whose reference is near
  zero is measured against a tenth of the average row.)
- Five tests run before any timing: 1, 4, 33, 128 (with heavily skewed routing) and 257 tokens. Each test
  calls `kernel_fn` directly, then captures it as a CUDA graph and replays it on two new inputs. Every 4th
  timed call (and the first) is checked as well.

For scale: SGLang's own kernel is at about 0.5% on these tests and the starting `submission.py` at about
0.4%. Accumulating the first projection in bfloat16 instead of FP32 gives about 2.8% and fails.

**How a kernel is timed**

- Each batch size is captured as a CUDA graph and timed by replaying it.
- Every replay starts with a cold L2 cache (the judge reads a 512 MiB buffer first).
- Your replays and SGLang's take turns on the same fresh inputs, on the same GPU, each side in its own
  process. The speedup is the median of the paired ratios, reported with a 95% interval of at least ±1.5%.
- A graph replay's fixed cost, about 5.6 µs, is subtracted from both sides.

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

### Running the judge

Everything runs inside the public SGLang v0.5.23 image,
`lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`
(SGLang's source is at `/sgl-workspace/sglang` inside it). On a machine with an H200, docker and the NVIDIA
Container Toolkit, `./run.sh` starts it with this repo mounted at `/work`; if you are already inside the
container, call `python3 judge.py` directly.

```bash
./run.sh submission.py --tests-only          # the five tests only (about 10 s)
./run.sh submission.py                       # tests, then timing at all six sizes
./run.sh submission.py --sizes 1,8           # timing at some sizes only
./run.sh submission.py --repeats 200         # exactly 200 timed pairs per size
./run.sh submission.py --json results.json   # also write every timed pair as JSON
./run.sh mine.py --baseline other.py         # time two of your own files against each other
./run.sh submission.py --profile 8           # torch profiler table for M=8 (eager calls, warm L2)
```

The output ends with a table like this (the starting `submission.py`):

```
tokens   SGLang us   yours us  speedup      95% interval  floor us yours/floor  pairs
     1        56.2    41744.5    0.001    [0.001, 0.001]      11.8    3533.93x    400
     8       123.4    41932.0    0.003    [0.003, 0.003]      53.1     790.10x    400
  ...
   256       811.4    43434.5    0.019    [0.018, 0.019]     183.7     236.26x    400

score (geometric mean of speedups): 0.005  95% interval [0.005, 0.005]
```

The number of timed pairs per size adapts: at least 100, then more until the size's interval is within ±0.5%,
up to 400 pairs or 40 s per size. A full run of SGLang against itself takes about 50 s; a full run of the
starting `submission.py` about 160 s (the first run in a fresh container adds a few minutes of compiling
SGLang's kernels; `run.sh` keeps the compiled kernels in `~/.cache/starter-task`). A full run uses up to
44 GB of GPU memory: each side's process holds about 23 GB (the weights, the copy handed to `prepare`, the
prepared state and the CUDA graphs).

When the judge is run with SGLang's kernel on both sides, every size reads 1.00 within its interval; in two
runs SGLang's times agreed to within 1%.

**Nsight Compute.** The `--ncu M` mode runs a few calls of
one batch size between `cudaProfilerStart` and `cudaProfilerStop`, so profile it with
`--profile-from-start off`:

```bash
DOCKER_ARGS=--cap-add=SYS_ADMIN ./run.sh --exec ncu --profile-from-start off --set full \
    -o prof python3 judge.py submission.py --ncu 8 --iters 1
```

Inside a container, `ncu` needs access to the GPU's performance counters (`--cap-add=SYS_ADMIN`, or the host
setting `NVreg_RestrictProfilingToAdminUsers=0`). If it reports `Failed to lock GPU clock frequencies`, add
`--clock-control none`; durations are then measured at the GPU's normal boost clocks rather than locked base
clocks. Use `-k regex:<name>` to profile only your own kernel.

## Deliverables

1. **Notes on how to measure performance accurately.**
   - What the judge measures, in your own words, and why it replays CUDA graphs and clears the L2 cache.
   - The SGLang times above, reproduced on your machine.
   - The 1-token floor (elaborate this or find a tutorial on calculating the theoretical upper bound) derived by hand: which bytes must be read, and how many.
   - One naive timing of your own (a Python loop around the call) and why it disagrees with the judge.
2. **Two checkpoints** along the optimization path. Each is a git tag whose `submission.py` passes every
   test, and each rests on a different idea, not a retuned constant. For each, at most one page:
   - what changed;
   - why it is faster, in terms of what the hardware is doing;
   - one profiler number that moved, before and after;
   - what limits the kernel now, and how far it is from the theoretical limit;
   - which parts of the kernel you can explain line by line, and which you can't.
3. **`UNKNOWNS.md`:** everything in the code or the profiler output that you do not understand. Be specific:
   "I don't know why 8 warps per SM beat 16 here".
4. **A 30-minute walkthrough** of your final kernel and your notes.

## How to work
- **It is fine not to understand a part.** Write it down in `UNKNOWNS.md`. The parts you do understand should
  be solid enough to go through line by line. "I don't know; here is how I would find out" is a good answer.
- **A kernel you can mostly explain beats a faster one you can't.** You choose how far to let the agent go.
- **Consider noise in your profiling and do repeated measurements.**
- **Questions about compute access or logistics:** any time.

## A suggested pace

| Days | Aim |
|---|---|
| 1–2 | Run the judge on the starting submission and the baseline. Write the measurement note. |
| 3–5 | First CUDA kernel that passes the tests. Profile it and SGLang's. Checkpoint 1. |
| 6–10 | Find what limits checkpoint 1 and improve it -> Checkpoint 2. |
| 11-12 | Write-up |

This is a guide, not a requirement. How you get to three checkpoints is yours to decide.

## What a good two weeks looks like

Roughly in order of importance:

1. **You understand your kernel.** You can walk through the kernel code of the three checkpoints and explain what it does and what limits it,
   and you know where your understanding stops.
2. **You learned how to profile kernels.**
3. **You can characterize how agents approach hyperoptimizing kernels and have thoughts on their strengths and weaknesses.**
4. **The kernel got faster.**

## Questions about GPU architecture

You probably have to pick up a little GPU programming knowledge for this task. By the walkthrough you should be able to
answer some of these for your own code. It's much better to be able to explain a few of these in-depth rather than all of them superficially.

- How a kernel launch maps onto the GPU: grids, blocks, threads, warps, and how many run at once on one SM.
- Where data lives: registers, shared memory, the L2 cache, HBM. What it costs to move between them.
- Which weight bytes one step has to read, and what a kernel must do to read them at close to 4.8 TB/s.
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
