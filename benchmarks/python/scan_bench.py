import argparse
import json
import math
import os
import platform
import subprocess
import time
from functools import partial

import mlx.core as mx
import numpy as np

N_WARMUP = 5
N_ITER = 100
AXIS_LENGTHS = [2**i for i in range(16, 25)]  # 2^16 .. 2^24
D_TYPES = ("float32", "float16")


def get_device_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").splitlines()[0].strip()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def bench_mlx(x):
    mx.eval(mx.cumsum(x, axis=-1))


def measure(fn):
    for _ in range(N_WARMUP):
        fn()
    start = time.perf_counter_ns()
    for _ in range(N_ITER):
        fn()
    return (time.perf_counter_ns() - start) * 1e-9


def run_bench():
    device = get_device_name()
    results = {"device": device, "n_iter": N_ITER, "cases": []}

    for dtype in D_TYPES:
        np_dtype = getattr(np, dtype)
        rng = np.random.default_rng(42)
        for length in AXIS_LENGTHS:
            x_np = rng.normal(0.0, 1.0, (1, length)).astype(np_dtype)
            x_mlx = mx.array(x_np)
            elapsed = measure(partial(bench_mlx, x_mlx))
            ms = elapsed / N_ITER * 1e3
            total_bytes = length * np_dtype().itemsize * 2 * N_ITER
            gbps = (total_bytes / float(1024**3)) / elapsed

            results["cases"].append(
                {
                    "dtype": dtype,
                    "length": length,
                    "time_ms": round(ms, 6),
                    "gbps": round(gbps, 2),
                }
            )
            print(f"  {dtype} (1, {length:>8d}):" f"  {ms:>8.4f} ms  {gbps:>7.1f} GB/s")

    return results


def plot(current_path, baseline_path, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    with open(current_path) as f:
        current = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    def _pow2_fmt(value, _pos):
        if value <= 0:
            return ""
        exp = int(round(math.log2(value)))
        return f"$2^{{{exp}}}$"

    def _build_map(data):
        m = {}
        for c in data["cases"]:
            m[(c["dtype"], c["length"])] = c
        return m

    cur_map = _build_map(current)
    base_map = _build_map(baseline)
    device = current.get("device", "GPU")

    os.makedirs(output_dir, exist_ok=True)

    n_dtypes = len(D_TYPES)
    fig, axs = plt.subplots(
        n_dtypes, 2, figsize=(10, 4 * n_dtypes), layout="constrained"
    )
    if n_dtypes == 1:
        axs = [axs]
    formatter = FuncFormatter(_pow2_fmt)

    for row, dtype in enumerate(D_TYPES):
        lengths, cur_gbps, base_gbps = [], [], []
        for length in AXIS_LENGTHS:
            key = (dtype, length)
            if key in cur_map and key in base_map:
                lengths.append(length)
                cur_gbps.append(cur_map[key]["gbps"])
                base_gbps.append(base_map[key]["gbps"])

        if not lengths:
            continue

        ax_perf, ax_speedup = axs[row]

        ax_perf.plot(lengths, cur_gbps, "tab:blue", label="current")
        ax_perf.plot(lengths, base_gbps, "tab:orange", linestyle="--", label="baseline")
        ax_perf.set_xscale("log", base=2)
        ax_perf.set_xticks(lengths)
        ax_perf.xaxis.set_major_formatter(formatter)
        ax_perf.set_ylabel("GB/s")
        ax_perf.set_xlabel("axis length")
        ax_perf.set_title(dtype)
        ax_perf.grid(True, which="both", linestyle=":", alpha=0.4)
        ax_perf.legend()

        speedup = np.array(cur_gbps) / np.array(base_gbps)
        ax_speedup.plot(lengths, speedup, "tab:green")
        ax_speedup.axhline(1.0, color="tab:gray", linestyle="--")
        ax_speedup.set_xscale("log", base=2)
        ax_speedup.set_xticks(lengths)
        ax_speedup.xaxis.set_major_formatter(formatter)
        ax_speedup.set_ylabel("Speedup (current / baseline)")
        ax_speedup.set_xlabel("axis length")
        ax_speedup.set_title(dtype)
        ax_speedup.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.suptitle(f"{device} | cumsum(axis=-1) batch=1")
    path = os.path.join(
        output_dir,
        f"{device.replace(' ', '_')}_scan.png",
    )
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Scan (cumsum) benchmark")
    parser.add_argument(
        "-o", "--output", default="results/scan.json", help="Output JSON path"
    )
    parser.add_argument(
        "--plot",
        nargs=2,
        metavar=("CURRENT", "BASELINE"),
        help="Plot current vs baseline from two JSON files instead of running",
    )
    parser.add_argument(
        "--plot-dir", default="results", help="Output directory for plot images"
    )

    args = parser.parse_args()

    if args.plot:
        plot(args.plot[0], args.plot[1], args.plot_dir)
    else:
        results = run_bench()
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
