import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_REPORT_DIR = Path("data") / "reports"
DEFAULT_OUT_DIR = DEFAULT_REPORT_DIR / "repeated_run_analysis"


STEP_FINISHED_RE = re.compile(r"^FINISHED STEP:\s*(?P<step>.+?)\s*$")
STEP_TIME_RE = re.compile(r"^STEP TIME SEC:\s*(?P<time>[0-9.]+)\s*$")
SCRIPT_START_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+SCRIPT START:\s*(?P<script>.+?)\s*$"
)
SCRIPT_FINISH_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+SCRIPT FINISHED\s*$"
)
PIPELINE_START_RE = re.compile(r"^PIPELINE START:\s*(?P<ts>\d{8}_\d{6})\s*$")


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


def get_summary_files(report_dir, n=None):
    files = sorted(
        report_dir.glob("pipeline_summary_*.json"),
        key=lambda p: p.stat().st_mtime,
    )

    if n is not None:
        files = files[-n:]

    return files


def get_log_files(log_dir, n=None, pattern="pipeline_*.log"):
    files = sorted(
        log_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )

    if n is not None:
        files = files[-n:]

    return files


def flatten_one_summary(path):
    data = load_json(path)

    row = {
        "source_type": "summary_json",
        "file": str(path),
        "filename": path.name,
        "timestamp": data.get("timestamp"),
        "total_pipeline_time_sec": data.get("total_pipeline_time_sec"),
        "final_n_total": safe_get(data, ["final_dedup", "n_total"]),
        "final_threshold": safe_get(data, ["final_dedup", "sim_threshold"]),
        "final_n_pairs_after_threshold": safe_get(data, ["final_dedup", "n_pairs_after_threshold"]),
        "final_n_used_pairs": safe_get(data, ["final_dedup", "n_used_pairs"]),
        "final_n_skipped_pairs": safe_get(data, ["final_dedup", "n_skipped_pairs"]),
        "final_n_keep": safe_get(data, ["final_dedup", "n_keep"]),
        "final_n_drop": safe_get(data, ["final_dedup", "n_drop"]),
        "final_drop_rate": safe_get(data, ["final_dedup", "drop_rate"]),
        "hnsw_recall_at_10": safe_get(data, ["hnsw_eval", "recall_at_10"]),
        "hnsw_build_time_sec": safe_get(data, ["hnsw_eval", "build_time_sec"]),
        "hnsw_query_time_sec": safe_get(data, ["hnsw_eval", "query_time_sec"]),
        "hnsw_qps": safe_get(data, ["hnsw_eval", "qps"]),
        "hnsw_exact_time_sec": safe_get(data, ["hnsw_eval", "exact_time_sec"]),
    }

    for step in data.get("steps", []):
        step_name = step.get("step")
        step_time = step.get("time_sec")
        if step_name is not None:
            row[f"step_time__{step_name}"] = step_time

    return row


def extract_threshold_sweep_from_summary(path):
    data = load_json(path)
    rows = []

    for item in data.get("threshold_sweep", []):
        rows.append({
            "source_type": "summary_json",
            "file": str(path),
            "filename": path.name,
            "timestamp": data.get("timestamp"),
            "threshold": item.get("threshold"),
            "n_pairs": item.get("n_pairs"),
            "n_drop": item.get("n_drop"),
            "drop_rate": item.get("drop_rate"),
            "n_components": item.get("n_components"),
            "n_duplicate_components": item.get("n_duplicate_components"),
            "max_component_size": item.get("max_component_size"),
        })

    return rows


def parse_datetime(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def parse_pipeline_start(value):
    return datetime.strptime(value, "%Y%m%d_%H%M%S")


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    row = {
        "source_type": "log",
        "file": str(path),
        "filename": path.name,
        "timestamp": None,
        "total_pipeline_time_sec": None,
        "final_n_total": None,
        "final_threshold": None,
        "final_n_pairs_after_threshold": None,
        "final_n_used_pairs": None,
        "final_n_skipped_pairs": None,
        "final_n_keep": None,
        "final_n_drop": None,
        "final_drop_rate": None,
    }

    step_rows = []
    current_finished_step = None
    current_script = None
    current_script_start = None
    pipeline_start = None
    last_script_finish = None

    for line in lines:
        stripped = line.strip()

        m = PIPELINE_START_RE.match(stripped)
        if m:
            pipeline_start = parse_pipeline_start(m.group("ts"))
            row["timestamp"] = pipeline_start.isoformat(sep=" ")
            continue

        m = SCRIPT_START_RE.match(stripped)
        if m:
            current_script = m.group("script")
            current_script_start = parse_datetime(m.group("ts"))
            continue

        m = SCRIPT_FINISH_RE.match(stripped)
        if m:
            finish_ts = parse_datetime(m.group("ts"))
            last_script_finish = finish_ts
            if current_script and current_script_start:
                step_rows.append({
                    "step": current_script,
                    "time_sec_from_script_timestamps": (finish_ts - current_script_start).total_seconds(),
                })
            current_script = None
            current_script_start = None
            continue

        m = STEP_FINISHED_RE.match(stripped)
        if m:
            current_finished_step = m.group("step")
            continue

        m = STEP_TIME_RE.match(stripped)
        if m and current_finished_step:
            row[f"step_time__{current_finished_step}"] = float(m.group("time"))
            current_finished_step = None
            continue

    if pipeline_start and last_script_finish:
        row["total_pipeline_time_sec"] = (last_script_finish - pipeline_start).total_seconds()

    report_match = re.search(
        r"=== HNSW \+ SemDeDup Backend Report ===\s*(?P<json>\{.*?\})",
        text,
        flags=re.DOTALL,
    )
    if report_match:
        try:
            report = json.loads(report_match.group("json"))
            row["final_n_total"] = report.get("n_total")
            row["final_threshold"] = report.get("sim_threshold")
            row["final_n_pairs_after_threshold"] = report.get("n_pairs_after_threshold")
            row["final_n_used_pairs"] = report.get("n_used_pairs")
            row["final_n_skipped_pairs"] = report.get("n_skipped_pairs")
            row["final_n_keep"] = report.get("n_keep")
            row["final_n_drop"] = report.get("n_drop")
            row["final_drop_rate"] = report.get("drop_rate")
        except json.JSONDecodeError:
            pass

    # If the runner did not print STEP TIME SEC for the final script, keep an
    # estimated script duration so the missing tail is still visible in CSV.
    for item in step_rows:
        step_name = item["step"]
        col = f"script_time_estimate__{step_name}"
        row[col] = item["time_sec_from_script_timestamps"]

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
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "geometric_mean": geometric_mean(values),
        })

    return pd.DataFrame(rows)


def make_metric_stats(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "arithmetic_mean": np.nan,
            "geometric_mean": np.nan,
            "median": np.nan,
        }

    return {
        "arithmetic_mean": float(np.mean(arr)),
        "geometric_mean": geometric_mean(arr),
        "median": float(np.median(arr)),
    }


def summarize_threshold_sweep(sweep_df):
    if len(sweep_df) == 0:
        return pd.DataFrame()

    rows = []
    metrics = [
        "n_pairs",
        "n_drop",
        "drop_rate",
        "n_components",
        "n_duplicate_components",
        "max_component_size",
    ]

    for threshold, group in sweep_df.groupby("threshold"):
        for metric in metrics:
            values = group[metric].dropna().values
            if len(values) == 0:
                continue

            rows.append({
                "threshold": threshold,
                "metric": metric,
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "geometric_mean": geometric_mean(values),
            })

    return pd.DataFrame(rows)


def plot_repeated_run_metric(runs_df, metric, ylabel, title, out_path, scale=1.0):
    if plt is None:
        return None

    if metric not in runs_df.columns:
        return None

    values = runs_df[metric].astype(float) * scale
    valid = values.dropna()
    if len(valid) == 0:
        return None

    x = np.arange(1, len(values) + 1)
    stats = make_metric_stats(valid.values)

    plt.figure(figsize=(8, 5))
    plt.plot(x, values.values, marker="o", linewidth=2, label="Run value")

    reference_lines = [
        ("Arithmetic mean", stats["arithmetic_mean"], "#d62728", "--"),
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


def plot_repeated_runs(runs_df, out_dir, timestamp):
    if plt is None:
        print("\n[Warning] matplotlib is not installed. CSV files were saved, but plots were skipped.")
        print("Install it with: pip install matplotlib")
        return []

    saved = []

    runtime_plot = plot_repeated_run_metric(
        runs_df,
        metric="total_pipeline_time_sec",
        ylabel="Runtime (sec)",
        title="Pipeline Runtime Across Runs",
        out_path=out_dir / f"plot_runtime_repeated_runs_{timestamp}.png",
    )
    if runtime_plot:
        saved.append(runtime_plot)

    drop_rate_plot = plot_repeated_run_metric(
        runs_df,
        metric="final_drop_rate",
        ylabel="Drop rate (%)",
        title="Final Drop Rate Across Runs",
        out_path=out_dir / f"plot_final_drop_rate_repeated_runs_{timestamp}.png",
        scale=100.0,
    )
    if drop_rate_plot:
        saved.append(drop_rate_plot)

    return saved


def print_key_results(runs_df):
    print("\n=== Key Results ===")

    cols = [
        "filename",
        "total_pipeline_time_sec",
        "final_threshold",
        "final_n_total",
        "final_n_drop",
        "final_drop_rate",
    ]
    available = [c for c in cols if c in runs_df.columns]
    print(runs_df[available].to_string(index=False))

    if "final_drop_rate" in runs_df.columns:
        values = runs_df["final_drop_rate"].dropna()
        if len(values) > 0:
            print(f"\nFinal drop rate mean: {values.mean():.8f} ({values.mean() * 100:.4f}%)")
            print(f"Final drop rate geometric mean: {geometric_mean(values):.8f} ({geometric_mean(values) * 100:.4f}%)")
            print(f"Final drop rate median: {values.median():.8f} ({values.median() * 100:.4f}%)")

    if "total_pipeline_time_sec" in runs_df.columns:
        values = runs_df["total_pipeline_time_sec"].dropna()
        if len(values) > 0:
            print(f"Total pipeline time mean: {values.mean():.4f} sec ({values.mean() / 60:.4f} min)")
            print(f"Total pipeline time geometric mean: {geometric_mean(values):.4f} sec ({geometric_mean(values) / 60:.4f} min)")
            print(f"Total pipeline time median: {values.median():.4f} sec ({values.median() / 60:.4f} min)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze pipeline runtime and final drop rate from summary JSON files and/or pipeline logs."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory containing pipeline_summary_*.json files.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        action="append",
        default=[],
        help="Pipeline log file to parse. Can be passed multiple times.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing pipeline logs. Use with --latest-logs.",
    )
    parser.add_argument(
        "--latest-logs",
        type=int,
        default=0,
        help="Use latest N logs from --log-dir.",
    )
    parser.add_argument(
        "--log-pattern",
        type=str,
        default="pipeline_*.log",
        help="Glob pattern used with --log-dir.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Use latest N pipeline_summary_*.json files. Use --n 0 to skip summary JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for output CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    sweep_rows = []

    if args.n != 0:
        summary_files = get_summary_files(args.report_dir, n=args.n)
        for path in summary_files:
            run_rows.append(flatten_one_summary(path))
            sweep_rows.extend(extract_threshold_sweep_from_summary(path))

    if args.log_dir is not None and args.latest_logs > 0:
        latest_logs = get_log_files(args.log_dir, n=args.latest_logs, pattern=args.log_pattern)
        args.log.extend(latest_logs)

    for path in args.log:
        run_rows.append(parse_log(path))

    if len(run_rows) == 0:
        raise RuntimeError("No input found. Provide --report-dir with pipeline_summary_*.json or pass --log.")

    runs_df = pd.DataFrame(run_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    metric_summary_df = summarize_numeric(runs_df)
    threshold_summary_df = summarize_threshold_sweep(sweep_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_csv = args.out_dir / f"runtime_drop_rate_raw_{timestamp}.csv"
    metric_summary_csv = args.out_dir / f"runtime_drop_rate_summary_{timestamp}.csv"
    sweep_csv = args.out_dir / f"threshold_sweep_raw_{timestamp}.csv"
    threshold_summary_csv = args.out_dir / f"threshold_sweep_summary_{timestamp}.csv"

    runs_df.to_csv(runs_csv, index=False)
    metric_summary_df.to_csv(metric_summary_csv, index=False)

    if len(sweep_df) > 0:
        sweep_df.to_csv(sweep_csv, index=False)
        threshold_summary_df.to_csv(threshold_summary_csv, index=False)

    plot_paths = plot_repeated_runs(runs_df, args.out_dir, timestamp)

    print_key_results(runs_df)

    print("\nSaved:")
    print(f"  raw runtime/drop-rate: {runs_csv}")
    print(f"  metric summary:        {metric_summary_csv}")
    if len(sweep_df) > 0:
        print(f"  threshold sweep raw:   {sweep_csv}")
        print(f"  threshold summary:     {threshold_summary_csv}")
    for path in plot_paths:
        print(f"  plot:                  {path}")


if __name__ == "__main__":
    main()

