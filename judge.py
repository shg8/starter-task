#!/usr/bin/env python3
"""The judge: tests a submission, then times it against SGLang's kernel.

    python judge.py submission.py                    tests, then timing at all six batch sizes
    python judge.py submission.py --tests-only       only the five tests
    python judge.py submission.py --sizes 1,8        time only these batch sizes
    python judge.py submission.py --repeats 100      exactly 100 timed pairs per size (default: adaptive)
    python judge.py a.py --baseline b.py             time two candidate files against each other
    python judge.py submission.py --json out.json    also write the results as JSON
    python judge.py submission.py --profile 8        torch profiler table of one batch size
    ncu --profile-from-start off -o prof python judge.py submission.py --ncu 8
                                                     a few calls of one batch size, for Nsight Compute

How a run works
  * Each side (the candidate and the baseline) runs in its own process. The parent sends both the
    same seeds and alternates between them; the two never run on the GPU at the same time.
  * Each process builds the same weights (seeded), runs prepare() once (limit 300 s; the state it
    keeps may hold at most the weights' size + 1 GiB of GPU memory) and, for each batch size,
    captures kernel_fn as a CUDA graph.
  * Every timed call: a fresh case (seeded) is copied in place into the graph's input buffers, the
    L2 cache is flushed by reading a 512 MiB buffer, then the replay is timed with CUDA events.
  * The two sides take turns, in swapped order on alternate pairs. Per size the speedup is the
    median of the paired ratios baseline / candidate. The score is their geometric mean, with a
    95% bootstrap interval that is never reported narrower than +-1.5%.
  * Each process measures the fixed cost of replaying a one-kernel graph the same way and
    subtracts it from its times.
  * Every 4th timed output (and the first) is checked against the FP32 reference.
  * Pairs per size (unless --repeats): at least 100, then more until the size's 95% interval is
    within +-0.5%, up to 400 pairs or 40 s per size.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

PREPARE_LIMIT_S = 300.0
PREPARE_SLACK_BYTES = 1 << 30
FLUSH_BYTES = 512 << 20
CHECK_EVERY = 4
FIXED_COST_SAMPLES = 50
MIN_PAIRS = 100
MAX_PAIRS = 400
SIZE_BUDGET_S = 40.0        # adaptive mode: stop a size after this much timing
TARGET_HALF_WIDTH = 0.005   # adaptive mode: or once the size's 95% interval is +-0.5%
CI_FLOOR = 0.015
BOOTSTRAP = 2000
TEST_GRAPH_REPLAYS = 2
GRAPH_HINT = ("kernel_fn is not CUDA-graph safe. Inside kernel_fn there must be no host syncs "
              "(.item(), .cpu(), .tolist(), .numpy(), print(tensor), torch.cuda.synchronize()), "
              "no data-dependent shapes (nonzero, boolean-mask indexing, torch.unique) and no "
              "branching on input values.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("prepare", "kernel_fn"):
        if not callable(getattr(mod, fn, None)):
            raise RuntimeError(f"{path} does not define {fn}()")
    return mod


# ==============================================================================================
# Worker: one candidate file in its own process.


class Candidate:
    def __init__(self, path, label):
        import torch

        import problem

        self.torch, self.problem = torch, problem
        torch.backends.cuda.matmul.allow_tf32 = False
        self.dev = torch.device("cuda")
        self.label = label
        self.mod = load_module(path, f"candidate_{label}")
        self.weights = problem.make_weights(self.dev)
        self.weights_bytes = problem.weights_nbytes(self.weights)
        torch.cuda.synchronize()
        gc.collect()
        base = torch.cuda.memory_allocated()
        given = {k: v.clone() for k, v in self.weights.items()}  # the reference keeps the originals
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            self.state = self.mod.prepare(given)
        torch.cuda.synchronize()
        self.prepare_s = time.perf_counter() - t0
        del given
        gc.collect()
        self.state_bytes = torch.cuda.memory_allocated() - base
        self.flush_buf = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device=self.dev)
        self.flush_out = torch.zeros((), dtype=torch.float32, device=self.dev)
        self.graph = None

    # -- helpers ---------------------------------------------------------------------------
    def flush_l2(self):
        self.torch.sum(self.flush_buf, dim=0, out=self.flush_out)

    def case(self, M, seed, offset, zipf_s):
        return self.problem.make_case(M, seed, offset, zipf_s, self.dev)

    def capture(self, case):
        torch = self.torch
        self.graph = None
        gc.collect()
        static = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()}
        static["config"] = dict(case["config"])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(2):
                self.mod.kernel_fn(static, self.state)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.no_grad(), torch.cuda.graph(g):
                out = self.mod.kernel_fn(static, self.state)
        except Exception as e:  # noqa: BLE001
            torch.cuda.synchronize()
            raise RuntimeError(f"{GRAPH_HINT}\nCapture failed with: {type(e).__name__}: {e}") from e
        self.graph, self.static, self.static_out = g, static, out

    def load(self, case):
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            self.static[k].copy_(case[k])

    def timed_replay(self):
        torch = self.torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self.flush_l2()
        start.record()
        self.graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0  # us

    def fixed_cost(self):
        torch = self.torch
        x = torch.zeros(1, device=self.dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            x.add_(1.0)
        times = []
        for _ in range(FIXED_COST_SAMPLES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self.flush_l2()
            start.record()
            g.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        times.sort()
        return times[len(times) // 2]

    # -- commands --------------------------------------------------------------------------
    def cmd_tests(self):
        torch, problem = self.torch, self.problem
        results = []
        for name, M, offset, zipf_s, seed in problem.TESTS:
            r = {"name": name, "M": M, "offset": offset, "ok": False}
            results.append(r)
            try:
                case = self.case(M, seed, offset, zipf_s)
                ref = problem.reference(case, self.weights)
                call = {k: (v.clone() if isinstance(v, torch.Tensor) else dict(v))
                        for k, v in case.items()}
                with torch.no_grad():
                    out = self.mod.kernel_fn(call, self.state)
                torch.cuda.synchronize()
                ok, msg, nums = problem.check(out, ref)
                r.update(eager=nums)
                if not ok:
                    r["message"] = f"direct call: {msg}"
                    continue
                self.capture(case)
                for j in range(TEST_GRAPH_REPLAYS):
                    fresh = self.case(M, seed * 1000 + j + 1, offset, zipf_s)
                    self.load(fresh)
                    self.graph.replay()
                    torch.cuda.synchronize()
                    ok, msg2, nums = problem.check(self.static_out, problem.reference(fresh, self.weights))
                    if not ok:
                        r["message"] = f"graph replay with new inputs: {msg2}"
                        break
                else:
                    r.update(ok=True, message=msg)
            except Exception as e:  # noqa: BLE001
                r["message"] = f"{type(e).__name__}: {e}"
                r["traceback"] = traceback.format_exc()
            finally:
                self.graph = None
        return {"tests": results}

    def cmd_setup(self, M, offset, zipf_s, seed):
        t0 = time.perf_counter()
        self.capture(self.case(M, seed, offset, zipf_s))
        self.M, self.offset, self.zipf_s = M, offset, zipf_s
        return {"capture_s": time.perf_counter() - t0, "fixed_us": self.fixed_cost()}

    def cmd_time(self, seed, check):
        torch, problem = self.torch, self.problem
        case = self.case(self.M, seed, self.offset, self.zipf_s)
        self.load(case)
        t = self.timed_replay()
        res = {"us": t, "floor": problem.floor(case)}
        if check:
            ok, msg, nums = problem.check(self.static_out, problem.reference(case, self.weights))
            res.update(checked=True, ok=ok, message=msg, rel_l2=nums.get("rel_l2"))
        return res

    def cmd_memory(self):
        torch = self.torch
        return {"max_allocated": torch.cuda.max_memory_allocated(),
                "max_reserved": torch.cuda.max_memory_reserved()}


def worker_main(conn, path, label):
    sys.path.insert(0, HERE)
    try:
        cand = Candidate(path, label)
        conn.send({"ok": True, "prepare_s": cand.prepare_s, "state_bytes": cand.state_bytes,
                   "weights_bytes": cand.weights_bytes})
    except Exception as e:  # noqa: BLE001
        conn.send({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
        return
    while True:
        msg = conn.recv()
        op = msg.pop("op")
        if op == "quit":
            conn.send({"ok": True})
            return
        try:
            res = getattr(cand, "cmd_" + op)(**msg)
            res["ok_call"] = True
        except Exception as e:  # noqa: BLE001
            res = {"ok_call": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        conn.send(res)


class Worker:
    """Parent-side handle of a worker process."""

    def __init__(self, path, label, timeout=1800):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.label = label
        self.proc = ctx.Process(target=worker_main, args=(child, os.path.abspath(path), label), daemon=True)
        self.proc.start()
        self.info = self._recv(timeout)

    def _recv(self, timeout):
        if not self.conn.poll(timeout):
            self.proc.kill()
            raise RuntimeError(f"{self.label}: no answer within {timeout} s")
        try:
            return self.conn.recv()
        except EOFError:
            raise RuntimeError(f"{self.label}: worker process died (exit code {self.proc.exitcode})")

    def call(self, op, timeout=1800, **kw):
        self.conn.send({"op": op, **kw})
        res = self._recv(timeout)
        if not res.get("ok_call", True):
            raise RuntimeError(f"{self.label} failed in {op}: {res['error']}\n{res.get('traceback', '')}")
        return res

    def close(self):
        try:
            self.conn.send({"op": "quit"})
            self._recv(60)
        except Exception:  # noqa: BLE001
            pass
        self.proc.join(30)
        if self.proc.is_alive():
            self.proc.kill()


# ==============================================================================================
# Statistics


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def bootstrap_medians(ratios, n, rng):
    import numpy as np
    r = np.asarray(ratios)
    idx = rng.integers(0, len(r), size=(n, len(r)))
    return np.median(r[idx], axis=1)


def size_interval(ratios, rng, n=BOOTSTRAP):
    import numpy as np
    meds = bootstrap_medians(ratios, n, rng)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def score_interval(per_size, rng, n=BOOTSTRAP):
    import numpy as np
    logs = np.zeros(n)
    for ratios in per_size:
        logs += np.log(bootstrap_medians(ratios, n, rng))
    geo = np.exp(logs / len(per_size))
    return float(np.percentile(geo, 2.5)), float(np.percentile(geo, 97.5))


def widen(center, lo, hi):
    return min(lo, center * (1 - CI_FLOOR)), max(hi, center * (1 + CI_FLOOR))


# ==============================================================================================
# Modes


def run_tests(w):
    res = w.call("tests")
    ok = True
    for r in res["tests"]:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['name']:<28} (experts {r['offset']}..{r['offset'] + 31}): {r.get('message', '')}")
        if not r["ok"]:
            ok = False
            if "traceback" in r and "CUDA-graph safe" not in r.get("message", ""):
                print("        " + r["traceback"].replace("\n", "\n        "))
    return ok, res["tests"]


def time_size(cand, base, M, args, rng, seed0):
    import problem
    sc = cand.call("setup", M=M, offset=problem.TIMING_OFFSET, zipf_s=problem.ZIPF_S, seed=seed0)
    sb = base.call("setup", M=M, offset=problem.TIMING_OFFSET, zipf_s=problem.ZIPF_S, seed=seed0)
    rows = []
    t0 = time.perf_counter()
    i = 0
    while True:
        seed = seed0 + 1 + i
        check = (i % CHECK_EVERY == 0)
        order = (cand, base) if i % 2 == 0 else (base, cand)
        got = {}
        for w in order:
            got[w.label] = w.call("time", seed=seed, check=check)
        for w in (cand, base):
            g = got[w.label]
            if check and not g["ok"]:
                raise SystemExit(f"\nFAIL: {w.label} output wrong at M={M}, seed {seed}: {g['message']}")
        tc = got["candidate"]["us"] - sc["fixed_us"]
        tb = got["baseline"]["us"] - sb["fixed_us"]
        rows.append({"seed": seed, "candidate_us": tc, "baseline_us": tb,
                     "floor_us": got["candidate"]["floor"]["floor_us"],
                     "local_routes": got["candidate"]["floor"]["local_routes"],
                     "local_experts": got["candidate"]["floor"]["local_experts"]})
        i += 1
        if args.repeats:
            if i >= args.repeats:
                break
            continue
        if i >= MAX_PAIRS:
            break
        if i >= MIN_PAIRS and i % 10 == 0:
            if time.perf_counter() - t0 > SIZE_BUDGET_S:
                break
            ratios = [r["baseline_us"] / r["candidate_us"] for r in rows]
            lo, hi = size_interval(ratios, rng, 500)
            med = median(ratios)
            if (hi - lo) / 2 <= TARGET_HALF_WIDTH * med:
                break
    return {"M": M, "pairs": rows, "fixed_us": {"candidate": sc["fixed_us"], "baseline": sb["fixed_us"]},
            "capture_s": {"candidate": sc["capture_s"], "baseline": sb["capture_s"]},
            "seconds": time.perf_counter() - t0}


def summarize(sizes_out, rng):
    per_size = []
    for s in sizes_out:
        p = s["pairs"]
        ratios = [r["baseline_us"] / r["candidate_us"] for r in p]
        med = median(ratios)
        lo, hi = size_interval(ratios, rng)
        s["summary"] = {
            "pairs": len(p),
            "baseline_us": median([r["baseline_us"] for r in p]),
            "candidate_us": median([r["candidate_us"] for r in p]),
            "speedup": med,
            "speedup_ci_raw": [lo, hi],
            "speedup_ci": list(widen(med, lo, hi)),
            "floor_us": median([r["floor_us"] for r in p]),
            "candidate_over_floor": median([r["candidate_us"] / r["floor_us"] for r in p]),
            "baseline_over_floor": median([r["baseline_us"] / r["floor_us"] for r in p]),
            "local_routes": median([r["local_routes"] for r in p]),
            "local_experts": median([r["local_experts"] for r in p]),
        }
        per_size.append(ratios)
    score = math.exp(sum(math.log(s["summary"]["speedup"]) for s in sizes_out) / len(sizes_out))
    lo, hi = score_interval(per_size, rng)
    return {"score": score, "score_ci_raw": [lo, hi], "score_ci": list(widen(score, lo, hi))}


def print_table(sizes_out, total, base_name):
    bn = "SGLang" if base_name == "baseline.py" else "baseline"
    print()
    print(f"{'tokens':>6} {bn + ' us':>11} {'yours us':>10} {'speedup':>8} {'95% interval':>17} "
          f"{'floor us':>9} {'yours/floor':>11} {'pairs':>6}")
    for s in sizes_out:
        m = s["summary"]
        lo, hi = m["speedup_ci"]
        print(f"{s['M']:>6} {m['baseline_us']:>11.1f} {m['candidate_us']:>10.1f} {m['speedup']:>8.3f} "
              f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>17} {m['floor_us']:>9.1f} "
              f"{m['candidate_over_floor']:>10.2f}x {m['pairs']:>6}")
    lo, hi = total["score_ci"]
    print(f"\nscore (geometric mean of speedups): {total['score']:.3f}  95% interval [{lo:.3f}, {hi:.3f}]")


def run_profile(path, M, ncu, iters):
    import torch

    import problem
    torch.cuda.set_device(0)
    dev = torch.device("cuda")
    mod = load_module(path, "candidate_profile")
    weights = problem.make_weights(dev)
    with torch.no_grad():
        state = mod.prepare(weights)
        cases = [problem.make_case(M, 9000 + i, problem.TIMING_OFFSET, problem.ZIPF_S, dev)
                 for i in range(iters + 3)]
        for c in cases[:3]:
            mod.kernel_fn(c, state)
        torch.cuda.synchronize()
        if ncu:
            torch.cuda.cudart().cudaProfilerStart()
            for c in cases[3:]:
                torch.cuda.nvtx.range_push(f"kernel_fn M={M}")
                mod.kernel_fn(c, state)
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"ran {iters} calls of kernel_fn at M={M} between cudaProfilerStart/Stop")
            return
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for c in cases[3:]:
                mod.kernel_fn(c, state)
            torch.cuda.synchronize()
    key = "device_time_total"
    try:
        table = prof.key_averages().table(sort_by=key, row_limit=25)
    except Exception:  # noqa: BLE001  (older torch)
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    print(table)
    trace = os.path.abspath(f"profile_M{M}.json")
    prof.export_chrome_trace(trace)
    print(f"{iters} calls at M={M} (not CUDA graphs, warm L2). Chrome trace: {trace}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", nargs="?", default="submission.py")
    ap.add_argument("--baseline", default=os.path.join(HERE, "baseline.py"),
                    help="file to time against (default: SGLang's kernel, baseline.py)")
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--sizes", default=None, help="comma-separated batch sizes (default 1,8,16,32,128,256)")
    ap.add_argument("--repeats", type=int, default=0, help="timed pairs per size (default: adaptive, %d-%d)" % (MIN_PAIRS, MAX_PAIRS))
    ap.add_argument("--json", default=None, help="write results to this file")
    ap.add_argument("--seed", type=int, default=0, help="base seed of the timed cases")
    ap.add_argument("--skip-tests", action="store_true", help="time without running the tests first")
    ap.add_argument("--profile", type=int, metavar="M", help="torch profiler on one batch size")
    ap.add_argument("--ncu", type=int, metavar="M", help="a few calls of one batch size for ncu")
    ap.add_argument("--iters", type=int, default=5, help="calls in --profile / --ncu mode")
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    import numpy as np

    import problem

    if args.profile or args.ncu:
        run_profile(args.submission, args.profile or args.ncu, bool(args.ncu), args.iters)
        return 0

    t_start = time.time()
    out = {"submission": os.path.abspath(args.submission), "baseline": os.path.abspath(args.baseline),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rc = 1
    workers = []

    def finish(code):
        out["wall_s"] = time.time() - t_start
        for w in workers:
            try:
                out.setdefault("gpu_memory", {})[w.label] = w.call("memory", timeout=60)
            except Exception:  # noqa: BLE001
                pass
            w.close()
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=1)
        print(f"\nwall time {out['wall_s']:.0f} s")
        return code

    print(f"== {args.submission}: prepare")
    cand = Worker(args.submission, "candidate")
    workers.append(cand)
    if not cand.info["ok"]:
        print(f"FAIL: could not load or prepare: {cand.info['error']}\n{cand.info.get('traceback', '')}")
        return finish(1)
    out["prepare"] = {k: cand.info[k] for k in ("prepare_s", "state_bytes", "weights_bytes")}
    print(f"  prepare took {cand.info['prepare_s']:.1f} s; state holds {cand.info['state_bytes'] / 2**30:.2f} GiB "
          f"of GPU memory (weights {cand.info['weights_bytes'] / 2**30:.2f} GiB)")
    if cand.info["prepare_s"] > PREPARE_LIMIT_S:
        print(f"FAIL: prepare took more than {PREPARE_LIMIT_S:.0f} s")
        return finish(1)
    if cand.info["state_bytes"] > cand.info["weights_bytes"] + PREPARE_SLACK_BYTES:
        print("FAIL: the state from prepare holds more GPU memory than the weights plus 1 GiB; "
              "the weights must stay 4-bit")
        return finish(1)

    if not args.skip_tests:
        print(f"== tests ({len(problem.TESTS)})")
        ok, tests = run_tests(cand)
        out["tests"] = tests
        out["tests_passed"] = ok
        if not ok:
            print("FAIL: tests failed")
            return finish(1)
        print("  all tests passed")
    if args.tests_only:
        return finish(0)

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else list(problem.SIZES)
    print(f"== timing against {os.path.basename(args.baseline)}: prepare")
    base = Worker(args.baseline, "baseline")
    workers.append(base)
    if not base.info["ok"]:
        print(f"FAIL: baseline could not load: {base.info['error']}\n{base.info.get('traceback', '')}")
        return finish(1)
    rng = np.random.default_rng(12345)
    sizes_out = []
    try:
        for k, M in enumerate(sizes):
            print(f"  M={M}: ", end="", flush=True)
            s = time_size(cand, base, M, args, rng, seed0=args.seed * 1_000_000 + (k + 1) * 100_000)
            sizes_out.append(s)
            print(f"{len(s['pairs'])} pairs in {s['seconds']:.0f} s; fixed cost "
                  f"{s['fixed_us']['candidate']:.1f} / {s['fixed_us']['baseline']:.1f} us", flush=True)
    except SystemExit as e:
        print(e)
        out["sizes"] = sizes_out
        return finish(1)
    total = summarize(sizes_out, rng)
    out["sizes"] = sizes_out
    out["score"] = total
    print_table(sizes_out, total, os.path.basename(args.baseline))
    rc = 0
    return finish(rc)


if __name__ == "__main__":
    sys.exit(main())
