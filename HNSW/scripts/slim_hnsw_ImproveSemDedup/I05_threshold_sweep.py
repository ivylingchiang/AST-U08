import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


# 輸入：
#   data/hnsw/candidate_pairs.parquet
#   data/samples/slimpajama_100k.jsonl
#
# 輸出：
#   data/reports/hnsw_threshold_sweep.csv
#   log/slim_hnsw/resource_data/slim_threshold_sweep_YYYYMMDD_HHMMSS.txt


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_SAMPLE = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
DEFAULT_PAIRS = ROOT / "data" / "hnsw" / "candidate_pairs.parquet"
DEFAULT_OUT = ROOT / "data" / "reports" / "hnsw_threshold_sweep.csv"

DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = Path("/home/u08/workspace/HNSW/log/slim_hnsw_ImproveSemDedup/resource_data")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.p[rb] = ra


def log_step(msg):
    print(f"[*] {msg}")


def save_resource_data(metrics):
    log_step("Saving resource metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = RESOURCE_DIR / f"Islim_threshold_sweep_{timestamp}.txt"

    script_name = os.path.basename(__file__)

    with file_name.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Resource Metrics for: {script_name}\n")
        f.write("=" * 60 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"File: {script_name} | Metric: {metric_name:<35} | Value: {value}\n")

    print(f"\n[Info] 資源量測數據已儲存至：{file_name}")


def count_jsonl(path):
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def eval_threshold(n, pairs, threshold):
    sub = pairs[pairs["cosine_sim"] >= threshold]

    dsu = DSU(n)

    for r in sub.itertuples(index=False):
        dsu.union(int(r.id1), int(r.id2))

    comps = {}

    for i in range(n):
        root = dsu.find(i)
        comps.setdefault(root, []).append(i)

    duplicate_components = [c for c in comps.values() if len(c) > 1]
    n_drop = sum(len(c) - 1 for c in duplicate_components)

    return {
        "threshold": threshold,
        "n_pairs": len(sub),
        "n_drop": n_drop,
        "drop_rate": n_drop / n,
        "n_components": len(comps),
        "n_duplicate_components": len(duplicate_components),
        "max_component_size": max(len(c) for c in comps.values()) if comps else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92],
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
    metrics["Thresholds"] = str(args.thresholds)

    log_step("Counting sample rows...")
    n = count_jsonl(args.sample)
    metrics["N_Total"] = n

    log_step("Loading candidate pairs...")
    pairs = pd.read_parquet(args.pairs)
    metrics["Candidate_Pairs_Total"] = len(pairs)

    rows = []

    for th in args.thresholds:
        log_step(f"Evaluating threshold={th}")
        rows.append(eval_threshold(n, pairs, th))

    df = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    metrics["Output_Report"] = str(args.out)
    metrics["Total_Core_Execution_Time_sec"] = round(time.time() - core_start_time, 4)

    save_resource_data(metrics)

    print("\n=== Threshold Sweep ===")
    print(df)

    end_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT FINISHED\n"
        + "=" * 60 + "\n"
    )
    print(end_msg)


if __name__ == "__main__":
    main()
