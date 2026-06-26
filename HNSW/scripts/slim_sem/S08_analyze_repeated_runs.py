import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


ROOT = Path("/home/u08/workspace/HNSW")
DEFAULT_REPORT_DIR = ROOT / "data" / "reports"
DEFAULT_OUT_DIR = ROOT / "data" / "reports" / "slim_sem_repeated_run_analysis"


def safe_get(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def geometric_mean(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if len(arr) == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(arr))))


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_summary_files(report_dir, n, pattern):
    files = sorted(
        report_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if n is not None:
        files = files[-n:]
    return files


def flatten_one_run(path):
    data = load_json(path)
    final_dedup = data.get("final_dedup") or {}

    row = {
        "file": str(path),
        "filename": path.name,
        "timestamp": data.get("timestamp"),
        "method": data.get("method", "slim_sem_kmeans_pairwise_semdedup"),
        "model": data.get("model"),
        "total_pipeline_time_sec": data.get("total_pipeline_time_sec"),
        "total_pipeline_time_min": (
            data.get("total_pipeline_time_sec") / 60
            if data.get("total_pipeline_time_sec") is not None
            else None
        ),
        "final_n_total": final_dedup.get("n_total"),
        "final_threshold": final_dedup.get("sim_threshold"),
        "final_n_pairs_after_threshold": final_dedup.get("n_pairs_after_threshold"),
        "final_n_duplicate_components": final_dedup.get("n_duplicate_components"),
        "final_max_component_size": final_dedup.get("max_component_size"),
        "final_n_keep": final_dedup.get("n_keep"),
        "final_n_drop": final_dedup.get("n_drop"),
        "final_drop_rate": final_dedup.get("drop_rate"),
        "final_drop_rate_percent": (
            final_dedup.get("drop_rate") * 100
            if final_dedup.get("drop_rate") is not None
            else None
        ),
        "final_time_sec": final_dedup.get("time_sec"),
    }

    for step in data.get("steps", []):
        step_name = step.get("step")
        step_time = step.get("time_sec")
        if step_name is not None:
            row[f"step_time__{step_name}"] = step_time

    return row


def summarize_numeric(df):
    rows = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        values = df[col].dropna().values
        if len(values) == 0:
            continue

        rows.append({
            "metric": col,
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "geometric_mean": geometric_mean(values),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })

    return pd.DataFrame(rows)


def metric_stats(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "mean": np.nan,
            "geometric_mean": np.nan,
            "median": np.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "geometric_mean": geometric_mean(arr),
        "median": float(np.median(arr)),
    }


def plot_metric(df, metric, ylabel, title, out_path):
    if plt is None:
        return None
    if metric not in df.columns:
        return None

    values = df[metric].astype(float)
    if values.dropna().empty:
        return None

    x = np.arange(1, len(values) + 1)
    stats = metric_stats(values.dropna().values)

    plt.figure(figsize=(8, 5))
    plt.plot(x, values.values, marker="o", linewidth=2, label="Run value")

    reference_lines = [
        ("Arithmetic mean", stats["mean"], "#d62728", "--"),
        ("Geometric mean", stats["geometric_mean"], "#2ca02c", "-."),
        ("Median", stats["median"], "#9467bd", ":"),
    ]

    for label, value, color, linestyle in reference_lines:
        if np.isfinite(value):
            plt.axhline(value, color=color, linestyle=linestyle, linewidth=2, label=f"{label}: {value:.6g}")

    plt.xticks(x, [f"run {i}" for i in x])
    plt.xlabel("Run")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def write_plots(runs_df, out_dir, timestamp):
    if plt is None:
        print("\n[Warning] matplotlib is not installed. CSV files were saved, but plots were skipped.")
        print("Install it with: pip install matplotlib")
        return []

    paths = []

    runtime_plot = plot_metric(
        runs_df,
        "total_pipeline_time_sec",
        "Runtime (sec)",
        "SlimSem Pipeline Runtime Across Runs",
        out_dir / f"plot_slim_sem_runtime_{timestamp}.png",
    )
    if runtime_plot is not None:
        paths.append(runtime_plot)

    drop_rate_plot = plot_metric(
        runs_df,
        "final_drop_rate_percent",
        "Drop rate (%)",
        "SlimSem Final Drop Rate Across Runs",
        out_dir / f"plot_slim_sem_drop_rate_{timestamp}.png",
    )
    if drop_rate_plot is not None:
        paths.append(drop_rate_plot)

    n_drop_plot = plot_metric(
        runs_df,
        "final_n_drop",
        "Dropped documents",
        "SlimSem Dropped Documents Across Runs",
        out_dir / f"plot_slim_sem_n_drop_{timestamp}.png",
    )
    if n_drop_plot is not None:
        paths.append(n_drop_plot)

    return paths


def print_key_results(runs_df, summary_df):
    print("\n=== Runs ===")
    cols = [
        "filename",
        "timestamp",
        "total_pipeline_time_sec",
        "total_pipeline_time_min",
        "final_threshold",
        "final_n_total",
        "final_n_drop",
        "final_drop_rate",
        "final_drop_rate_percent",
    ]
    available = [c for c in cols if c in runs_df.columns]
    print(runs_df[available].to_string(index=False))

    print("\n=== Key Summary ===")
    key_metrics = [
        "total_pipeline_time_sec",
        "total_pipeline_time_min",
        "final_n_drop",
        "final_drop_rate",
        "final_drop_rate_percent",
    ]
    key_summary = summary_df[summary_df["metric"].isin(key_metrics)]
    print(key_summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze latest SlimSem repeated pipeline summaries."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument(
        "--pattern",
        type=str,
        default="pipeline_summary_slim_sem_*.json",
        help="Summary glob pattern.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = get_summary_files(args.report_dir, args.n, args.pattern)
    if len(files) == 0:
        raise RuntimeError(f"No summary files found in {args.report_dir} with pattern {args.pattern}")

    print("[*] Using summary files:")
    for path in files:
        print(f"  {path}")

    runs_df = pd.DataFrame([flatten_one_run(path) for path in files])
    summary_df = summarize_numeric(runs_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_csv = args.out_dir / f"slim_sem_repeated_runs_raw_{timestamp}.csv"
    summary_csv = args.out_dir / f"slim_sem_repeated_runs_summary_{timestamp}.csv"

    runs_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plot_paths = write_plots(runs_df, args.out_dir, timestamp)

    print_key_results(runs_df, summary_df)

    print("\nSaved:")
    print(f"  raw runs:       {raw_csv}")
    print(f"  metric summary: {summary_csv}")
    for path in plot_paths:
        print(f"  plot:           {path}")


if __name__ == "__main__":
    main()

