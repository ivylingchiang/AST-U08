import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


# 輸入：
#   data/samples/slimpajama_100k.jsonl
#   data/hnsw/candidate_pairs.parquet
#
# 輸出：
#   data/reports/manual_pairs.txt
#   log/slim_hnsw/resource_data/slim_inspect_pairs_YYYYMMDD_HHMMSS.txt


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_SAMPLE = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
DEFAULT_PAIRS = ROOT / "data" / "hnsw" / "candidate_pairs.parquet"
DEFAULT_OUT = ROOT / "data" / "reports" / "manual_pairs.txt"

DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = Path("/home/u08/workspace/HNSW/log/slim_hnsw_ImproveSemDedup/resource_data")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


def log_step(msg):
    print(f"[*] {msg}")


def save_resource_data(metrics):
    log_step("Saving resource metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = RESOURCE_DIR / f"Islim_inspect_pairs_{timestamp}.txt"

    script_name = os.path.basename(__file__)

    with file_name.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Resource Metrics for: {script_name}\n")
        f.write("=" * 60 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"File: {script_name} | Metric: {metric_name:<35} | Value: {value}\n")

    print(f"\n[Info] 資源量測數據已儲存至：{file_name}")


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--text-chars", type=int, default=2000)

    parser.add_argument(
        "--mode",
        type=str,
        default="top",
        choices=["top", "near_threshold"],
    )

    return parser.parse_args()


def main():
    start_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT START: {os.path.basename(__file__)}\n"
        + "=" * 60
    )
    print(start_msg)

    args = parse_args()
    metrics = {}
    core_start_time = time.time()

    metrics["Input_Sample"] = str(args.sample)
    metrics["Input_Pairs"] = str(args.pairs)
    metrics["Threshold"] = args.threshold
    metrics["Mode"] = args.mode
    metrics["Limit"] = args.limit

    log_step("Loading rows...")
    rows = read_rows(args.sample)

    log_step("Loading pairs...")
    pairs = pd.read_parquet(args.pairs)
    pairs = pairs[pairs["cosine_sim"] >= args.threshold].copy()

    if args.mode == "top":
        pairs = pairs.sort_values("cosine_sim", ascending=False)
    else:
        pairs["distance_to_threshold"] = (pairs["cosine_sim"] - args.threshold).abs()
        pairs = pairs.sort_values("distance_to_threshold", ascending=True)

    pairs = pairs.head(args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    log_step("Writing inspection file...")

    with args.out.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(pairs.itertuples(index=False), 1):
            a = int(r.id1)
            b = int(r.id2)

            f.write("=" * 100 + "\n")
            f.write(f"PAIR {idx}\n")
            f.write(f"id1={a}, id2={b}, cosine_sim={float(r.cosine_sim):.6f}\n")
            f.write(f"source1={getattr(r, 'source1', 'unknown')}, source2={getattr(r, 'source2', 'unknown')}\n")
            f.write(f"len1={getattr(r, 'len1', 'unknown')}, len2={getattr(r, 'len2', 'unknown')}\n\n")

            f.write("[TEXT 1]\n")
            f.write(rows[a].get("text", "")[:args.text_chars].replace("\n", " ") + "\n\n")

            f.write("[TEXT 2]\n")
            f.write(rows[b].get("text", "")[:args.text_chars].replace("\n", " ") + "\n\n")

    metrics["Pairs_Inspected"] = len(pairs)
    metrics["Output_File"] = str(args.out)
    metrics["Total_Core_Execution_Time_sec"] = round(time.time() - core_start_time, 4)

    save_resource_data(metrics)

    print(f"\n[Info] manual inspection file saved to: {args.out}")

    end_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT FINISHED\n"
        + "=" * 60 + "\n"
    )
    print(end_msg)


if __name__ == "__main__":
    main()
