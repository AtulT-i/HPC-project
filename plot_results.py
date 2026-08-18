import csv
import os
import matplotlib
matplotlib.use("Agg")  # no display available on the cluster, just save to file
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "benchmark_data.csv")


def load_results():
    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["seq_ms"] = float(row["seq_ms"])
            row["omp_ms"] = float(row["omp_ms"])
            row["cuda_ms"] = float(row["cuda_ms"])
            row["omp_speedup"] = float(row["omp_speedup"])
            row["cuda_speedup"] = float(row["cuda_speedup"])
            rows.append(row)
    return rows


def plot_execution_time(rows, operation, filename):
    subset = [r for r in rows if r["operation"] == operation]
    if not subset:
        return

    labels = [r["config"] for r in subset]
    seq_times = [r["seq_ms"] for r in subset]
    omp_times = [r["omp_ms"] for r in subset]
    cuda_times = [r["cuda_ms"] for r in subset]

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], seq_times, width, label="Sequential")
    ax.bar([i for i in x], omp_times, width, label="OpenMP")
    ax.bar([i + width for i in x], cuda_times, width, label="CUDA")

    ax.set_yscale("log")
    ax.set_ylabel("Execution Time (ms, log scale)")
    ax.set_xlabel("Input -> Output Configuration")
    ax.set_title(f"{operation.upper()} - Execution Time Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_speedup(rows, operation, filename):
    subset = [r for r in rows if r["operation"] == operation]
    if not subset:
        return

    labels = [r["config"] for r in subset]
    omp_speedup = [r["omp_speedup"] for r in subset]
    cuda_speedup = [r["cuda_speedup"] for r in subset]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], omp_speedup, width, label="OpenMP Speedup")
    ax.bar([i + width/2 for i in x], cuda_speedup, width, label="CUDA Speedup")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1x)")
    ax.set_ylabel("Speedup vs Sequential (x)")
    ax.set_xlabel("Input -> Output Configuration")
    ax.set_title(f"{operation.upper()} - Speedup Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    rows = load_results()

    for op in ["conv", "pool", "fc"]:
        plot_execution_time(rows, op, f"{op}_execution_time.png")
        plot_speedup(rows, op, f"{op}_speedup.png")

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()