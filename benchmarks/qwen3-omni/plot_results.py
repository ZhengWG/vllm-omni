"""Plot Qwen3-Omni benchmark results.

Generates comparison bar charts:

- TTFP (Time-to-First-Audio-Packet) across concurrency levels
- E2E latency across concurrency levels
- RTF (Real-Time Factor) across concurrency levels
- Audio throughput across concurrency levels

Usage:
    # Compare two configs (vllm-omni vs hf_transformers):
    python plot_results.py \
        --results results/bench_async_chunk_*.json results/bench_hf_transformers_*.json \
        --labels "vllm-omni" "hf_transformers" \
        --output results/qwen3_omni_benchmark.png

    # Single config:
    python plot_results.py \
        --results results/bench_async_chunk_*.json \
        --labels "vllm-omni" \
        --output results/qwen3_omni_benchmark.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(result_files: list[str]) -> list[list[dict]]:
    all_results = []
    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)
        all_results.append(data)
    return all_results


def plot_comparison(
    all_results: list[list[dict]],
    labels: list[str],
    output_path: str,
    title_prefix: str = "Qwen3-Omni",
):
    n_configs = len(all_results)

    all_concurrencies = [set(r["concurrency"] for r in results) for results in all_results]
    concurrencies = sorted(set.union(*all_concurrencies))

    ttfp_data = {label: [] for label in labels}
    e2e_data = {label: [] for label in labels}
    rtf_data = {label: [] for label in labels}
    throughput_data = {label: [] for label in labels}

    for results, label in zip(all_results, labels):
        conc_map = {r["concurrency"]: r for r in results}
        for c in concurrencies:
            r = conc_map.get(c)
            ttfp_data[label].append(r.get("mean_ttfp_ms") if r else None)
            e2e_data[label].append(r.get("mean_e2e_ms") if r else None)
            rtf_data[label].append(r.get("mean_rtf") if r else None)
            throughput_data[label].append(r.get("audio_throughput") if r else None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} Performance Benchmark", fontsize=16, fontweight="bold")

    x = np.arange(len(concurrencies))
    width = 0.35 if n_configs == 2 else 0.5
    if n_configs > 1:
        offsets = np.linspace(-width / 2 * (n_configs - 1), width / 2 * (n_configs - 1), n_configs)
    else:
        offsets = [0]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107"]

    def plot_metric(ax, data_dict, ylabel, title, fmt=".1f"):
        for i, (label, values) in enumerate(data_dict.items()):
            plot_values = [v if v is not None else 0 for v in values]
            color = colors[i % len(colors)]
            bar = ax.bar(x + offsets[i], plot_values, width, label=label, color=color, alpha=0.85)
            max_val = max((v for v in values if v is not None), default=1)
            for rect, val in zip(bar, values):
                if val is not None and val > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + max_val * 0.02,
                        f"{val:{fmt}}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )
        ax.set_xlabel("Concurrency", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    plot_metric(axes[0, 0], ttfp_data, "TTFP (ms)", "Time to First Audio Packet (TTFP)")
    plot_metric(axes[0, 1], e2e_data, "E2E Latency (ms)", "End-to-End Latency (E2E)")
    plot_metric(axes[1, 0], rtf_data, "RTF", "Real-Time Factor (RTF)", fmt=".3f")
    plot_metric(axes[1, 1], throughput_data, "Audio-sec / Wall-sec", "Audio Throughput", fmt=".2f")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_single_summary(results: list[dict], label: str, output_path: str):
    concurrencies = [r["concurrency"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Qwen3-Omni Benchmark - {label}", fontsize=15, fontweight="bold")

    def percentile_bars(ax, key_prefix, ylabel, title):
        means = [r.get(f"mean_{key_prefix}_ms", 0.0) for r in results]
        medians = [r.get(f"median_{key_prefix}_ms", 0.0) for r in results]
        p90s = [r.get(f"p90_{key_prefix}_ms", 0.0) for r in results]
        p99s = [r.get(f"p99_{key_prefix}_ms", 0.0) for r in results]
        x = np.arange(len(concurrencies))
        w = 0.2
        ax.bar(x - 1.5 * w, means, w, label="mean", color="#2196F3")
        ax.bar(x - 0.5 * w, medians, w, label="median", color="#4CAF50")
        ax.bar(x + 0.5 * w, p90s, w, label="p90", color="#FF9800")
        ax.bar(x + 1.5 * w, p99s, w, label="p99", color="#F44336")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    percentile_bars(axes[0], "ttfp", "TTFP (ms)", "Time to First Audio Packet")
    percentile_bars(axes[1], "e2e", "E2E (ms)", "End-to-End Latency")

    ax = axes[2]
    means = [r.get("mean_rtf", 0.0) for r in results]
    medians = [r.get("median_rtf", 0.0) for r in results]
    x = np.arange(len(concurrencies))
    ax.bar(x - 0.15, means, 0.3, label="mean", color="#2196F3")
    ax.bar(x + 0.15, medians, 0.3, label="median", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("RTF")
    ax.set_title("Real-Time Factor")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def print_comparison_table(all_results: list[list[dict]], labels: list[str]):
    concurrencies = sorted({r["concurrency"] for results in all_results for r in results})

    print("\n## Benchmark Results\n")
    header = "| Metric | Concurrency |"
    sep = "| --- | --- |"
    for label in labels:
        header += f" {label} |"
        sep += " --- |"
    print(header)
    print(sep)

    for metric, key, fmt in [
        ("TTFP (ms)", "mean_ttfp_ms", ".1f"),
        ("TTFT (ms)", "mean_ttft_ms", ".1f"),
        ("E2E (ms)", "mean_e2e_ms", ".1f"),
        ("RTF", "mean_rtf", ".3f"),
        ("Throughput (audio-s/s)", "audio_throughput", ".2f"),
    ]:
        for c in concurrencies:
            row = f"| {metric} | {c} |"
            for results in all_results:
                conc_map = {r["concurrency"]: r for r in results}
                val = conc_map.get(c, {}).get(key, 0) or 0
                row += f" {val:{fmt}} |"
            print(row)

    if len(all_results) == 2:
        print(f"\n## Improvement ({labels[0]} vs {labels[1]})\n")
        print("| Metric | Concurrency | Improvement |")
        print("| --- | --- | --- |")
        for metric, key in [("TTFP", "mean_ttfp_ms"), ("E2E", "mean_e2e_ms"), ("RTF", "mean_rtf")]:
            for c in concurrencies:
                m0 = {r["concurrency"]: r for r in all_results[0]}
                m1 = {r["concurrency"]: r for r in all_results[1]}
                v0 = m0.get(c, {}).get(key, 0) or 0
                v1 = m1.get(c, {}).get(key, 0) or 0
                if v1 > 0:
                    pct = (v1 - v0) / v1 * 100
                    print(f"| {metric} | {c} | {pct:+.1f}% |")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Qwen3-Omni benchmark results")
    parser.add_argument("--results", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="results/qwen3_omni_benchmark.png")
    parser.add_argument("--title", type=str, default="Qwen3-Omni")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert len(args.results) == len(args.labels), "--results and --labels must have the same count"

    all_results = load_results(args.results)
    print_comparison_table(all_results, args.labels)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if len(all_results) == 1:
        plot_single_summary(all_results[0], args.labels[0], args.output)
    else:
        plot_comparison(all_results, args.labels, args.output, title_prefix=args.title)
