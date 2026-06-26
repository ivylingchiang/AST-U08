import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# 目的：
#   使用 HNSW candidate pairs 取代 SemDeDup 前段 K-means candidates，
#   然後直接執行 SemDeDup-style backend removal。
#
# 輸入：
#   data/samples/slimpajama_100k.jsonl
#   data/hnsw/candidate_pairs.parquet
#
# 輸出：
#   data/semdedup/keep_ids.txt
#   data/semdedup/drop_ids.txt
#   data/semdedup/drop_pairs.parquet
#   data/semdedup/slimpajama_100k_dedup.jsonl
#   data/reports/semdedup_backend_hnsw_report.json
#   log/slim_hnsw/resource_data/slim_semdedup_backend_hnsw_YYYYMMDD_HHMMSS.txt
# ============================================================


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_SAMPLE = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
DEFAULT_PAIRS = ROOT / "data" / "hnsw" / "candidate_pairs.parquet"

DEFAULT_KEEP = ROOT / "data" / "semdedup" / "keep_ids.txt"
DEFAULT_DROP = ROOT / "data" / "semdedup" / "drop_ids.txt"
DEFAULT_DROP_PAIRS = ROOT / "data" / "semdedup" / "drop_pairs.parquet"
DEFAULT_DEDUP_JSONL = ROOT / "data" / "semdedup" / "slimpajama_100k_dedup.jsonl"
DEFAULT_REPORT = ROOT / "data" / "reports" / "semdedup_backend_hnsw_report.json"

DEFAULT_KEEP.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = Path("/home/u08/workspace/HNSW/log/slim_hnsw_ImproveSemDedup/resource_data")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


def log_step(msg):
    print(f"[*] {msg}")


def save_resource_data(metrics):
    log_step("Saving resource metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = RESOURCE_DIR / f"Islim_semdedup_backend_hnsw_{timestamp}.txt"

    script_name = os.path.basename(__file__)

    with file_name.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Resource Metrics for: {script_name}\n")
        f.write("=" * 60 + "\n")
        for metric_name, value in metrics.items():
            f.write(
                f"File: {script_name} | "
                f"Metric: {metric_name:<35} | "
                f"Value: {value}\n"
            )

    print(f"\n[Info] 資源量測數據已儲存至：{file_name}")


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def quality_score(row, policy):
    """
    決定 duplicate pair 中要保留哪一筆。

    longest:
        保留比較長的文本。
    shortest:
        保留比較短的文本。
    original_order:
        保留 id 較小者，模擬較早出現的樣本。
    """

    text_len = row.get("text_len", len(row.get("text", "")))
    row_id = row.get("id", 0)

    if policy == "longest":
        return (text_len, -row_id)

    if policy == "shortest":
        return (-text_len, -row_id)

    if policy == "original_order":
        return (-row_id,)

    raise ValueError(f"Unknown keep policy: {policy}")


def semdedup_greedy_removal(rows, pairs, threshold, keep_policy):
    """
    SemDeDup-style backend:
    1. 只保留 cosine_sim >= threshold 的 candidate pairs
    2. similarity 由高到低處理
    3. 每個 pair 中，保留 quality_score 高者，drop 低者
    4. 若 pair 中任一文件已被 drop，則跳過

    這比 union-find 更不容易因為 transitive relation 過度合併。
    """

    n = len(rows)

    active = [True] * n
    dropped_by = {}

    pairs = pairs[pairs["cosine_sim"] >= threshold].copy()
    pairs = pairs.sort_values("cosine_sim", ascending=False)

    drop_records = []
    used_pairs = 0
    skipped_pairs = 0

    for r in pairs.itertuples(index=False):
        i = int(r.id1)
        j = int(r.id2)
        sim = float(r.cosine_sim)

        if i < 0 or j < 0 or i >= n or j >= n:
            skipped_pairs += 1
            continue

        if not active[i] or not active[j]:
            skipped_pairs += 1
            continue

        score_i = quality_score(rows[i], keep_policy)
        score_j = quality_score(rows[j], keep_policy)

        if score_i >= score_j:
            keep_id = i
            drop_id = j
        else:
            keep_id = j
            drop_id = i

        active[drop_id] = False
        dropped_by[drop_id] = keep_id
        used_pairs += 1

        drop_records.append({
            "drop_id": drop_id,
            "keep_id": keep_id,
            "cosine_sim": sim,
            "drop_source": rows[drop_id].get("source", "unknown"),
            "keep_source": rows[keep_id].get("source", "unknown"),
            "drop_len": rows[drop_id].get("text_len", len(rows[drop_id].get("text", ""))),
            "keep_len": rows[keep_id].get("text_len", len(rows[keep_id].get("text", ""))),
        })

    keep_ids = [i for i, is_active in enumerate(active) if is_active]
    drop_ids = [i for i, is_active in enumerate(active) if not is_active]

    drop_pairs = pd.DataFrame(drop_records)

    stats = {
        "n_total": n,
        "sim_threshold": threshold,
        "n_pairs_after_threshold": int(len(pairs)),
        "n_used_pairs": int(used_pairs),
        "n_skipped_pairs": int(skipped_pairs),
        "n_keep": int(len(keep_ids)),
        "n_drop": int(len(drop_ids)),
        "drop_rate": float(len(drop_ids) / n),
    }

    return keep_ids, drop_ids, drop_pairs, stats


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")


def write_dedup_jsonl(path, rows, keep_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_set = set(keep_ids)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if int(row["id"]) in keep_set:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--threshold", type=float, default=0.90)

    parser.add_argument(
        "--keep-policy",
        type=str,
        default="longest",
        choices=["longest", "shortest", "original_order"],
    )

    parser.add_argument("--out-keep", type=Path, default=DEFAULT_KEEP)
    parser.add_argument("--out-drop", type=Path, default=DEFAULT_DROP)
    parser.add_argument("--out-drop-pairs", type=Path, default=DEFAULT_DROP_PAIRS)
    parser.add_argument("--out-dedup-jsonl", type=Path, default=DEFAULT_DEDUP_JSONL)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_REPORT)

    return parser.parse_args()


def main():
    start_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"SCRIPT START: {os.path.basename(__file__)}\n"
        + "=" * 60
    )
    print(start_msg)

    args = parse_args()

    metrics = {}
    core_start_time = time.time()

    metrics["Input_Sample"] = str(args.sample)
    metrics["Input_HNSW_Pairs"] = str(args.pairs)
    metrics["Threshold"] = args.threshold
    metrics["Keep_Policy"] = args.keep_policy

    log_step("Loading SlimPajama sample...")
    sample_start = time.time()
    rows = read_rows(args.sample)
    metrics["Sample_Load_Time_sec"] = round(time.time() - sample_start, 4)
    metrics["N_Total"] = len(rows)

    log_step("Loading HNSW candidate pairs...")
    pair_start = time.time()
    pairs = pd.read_parquet(args.pairs)
    metrics["Pair_Load_Time_sec"] = round(time.time() - pair_start, 4)
    metrics["Candidate_Pairs_Before_Threshold"] = len(pairs)

    log_step("Running SemDeDup-style backend removal...")
    remove_start = time.time()

    keep_ids, drop_ids, drop_pairs, stats = semdedup_greedy_removal(
        rows=rows,
        pairs=pairs,
        threshold=args.threshold,
        keep_policy=args.keep_policy,
    )

    metrics["Removal_Time_sec"] = round(time.time() - remove_start, 4)

    log_step("Saving keep/drop ids...")
    write_ids(args.out_keep, keep_ids)
    write_ids(args.out_drop, drop_ids)

    log_step("Saving drop pair records...")
    args.out_drop_pairs.parent.mkdir(parents=True, exist_ok=True)
    if len(drop_pairs) == 0:
        drop_pairs = pd.DataFrame(
            columns=[
                "drop_id",
                "keep_id",
                "cosine_sim",
                "drop_source",
                "keep_source",
                "drop_len",
                "keep_len",
            ]
        )
    drop_pairs.to_parquet(args.out_drop_pairs, index=False)

    log_step("Saving deduplicated jsonl...")
    write_dedup_jsonl(args.out_dedup_jsonl, rows, keep_ids)

    report = {
        **stats,
        "method": "hnsw_candidates_plus_semdedup_backend",
        "keep_policy": args.keep_policy,
        "input_sample": str(args.sample),
        "input_pairs": str(args.pairs),
        "out_keep": str(args.out_keep),
        "out_drop": str(args.out_drop),
        "out_drop_pairs": str(args.out_drop_pairs),
        "out_dedup_jsonl": str(args.out_dedup_jsonl),
    }

    log_step("Saving report...")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    metrics["Pairs_After_Threshold"] = stats["n_pairs_after_threshold"]
    metrics["Used_Pairs"] = stats["n_used_pairs"]
    metrics["Skipped_Pairs"] = stats["n_skipped_pairs"]
    metrics["N_Keep"] = stats["n_keep"]
    metrics["N_Drop"] = stats["n_drop"]
    metrics["Drop_Rate"] = round(stats["drop_rate"], 8)
    metrics["Output_Keep"] = str(args.out_keep)
    metrics["Output_Drop"] = str(args.out_drop)
    metrics["Output_Drop_Pairs"] = str(args.out_drop_pairs)
    metrics["Output_Dedup_JSONL"] = str(args.out_dedup_jsonl)
    metrics["Output_Report"] = str(args.out_report)
    metrics["Total_Core_Execution_Time_sec"] = round(time.time() - core_start_time, 4)

    save_resource_data(metrics)

    print("\n=== HNSW + SemDeDup Backend Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    end_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT FINISHED\n"
        + "=" * 60 + "\n"
    )
    print(end_msg)


if __name__ == "__main__":
    main()
