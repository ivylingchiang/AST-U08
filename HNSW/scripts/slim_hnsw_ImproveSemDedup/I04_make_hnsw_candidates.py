import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import hnswlib
import numpy as np
import pandas as pd


# 輸入：
#   data/embeddings/slimpajama_100k_emb.npy
#   data/embeddings/slimpajama_100k_meta.jsonl
#
# 輸出：
#   data/hnsw/candidate_pairs.parquet
#   data/hnsw/hnsw_index.bin
#   log/slim_hnsw/resource_data/slim_hnsw_candidates_YYYYMMDD_HHMMSS.txt


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_EMB = ROOT / "data" / "embeddings" / "slimpajama_100k_emb.npy"
DEFAULT_META = ROOT / "data" / "embeddings" / "slimpajama_100k_meta.jsonl"
DEFAULT_OUT_PAIRS = ROOT / "data" / "hnsw" / "candidate_pairs.parquet"
DEFAULT_OUT_INDEX = ROOT / "data" / "hnsw" / "hnsw_index.bin"

DEFAULT_OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = Path("/home/u08/workspace/HNSW/log/slim_hnsw_ImproveSemDedup/resource_data")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


def log_step(msg):
    print(f"[*] {msg}")


def save_resource_data(metrics):
    log_step("Saving resource metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = RESOURCE_DIR / f"Islim_hnsw_candidates_{timestamp}.txt"

    script_name = os.path.basename(__file__)

    with file_name.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Resource Metrics for: {script_name}\n")
        f.write("=" * 60 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"File: {script_name} | Metric: {metric_name:<35} | Value: {value}\n")

    print(f"\n[Info] 資源量測數據已儲存至：{file_name}")


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return x / norm


def load_meta(path, n=None):
    metas = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
            if n is not None and len(metas) >= n:
                break

    return metas


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--n", type=int, default=None)

    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--sim-threshold", type=float, default=0.75)

    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef", type=int, default=200)

    parser.add_argument("--out-pairs", type=Path, default=DEFAULT_OUT_PAIRS)
    parser.add_argument("--out-index", type=Path, default=DEFAULT_OUT_INDEX)
    parser.add_argument("--save-index", action="store_true")

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

    metrics["Input_Embedding_File"] = str(args.emb)
    metrics["Input_Meta_File"] = str(args.meta)
    metrics["K"] = args.k
    metrics["Similarity_Threshold"] = args.sim_threshold
    metrics["HNSW_M"] = args.m
    metrics["HNSW_ef_construction"] = args.ef_construction
    metrics["HNSW_ef"] = args.ef

    log_step("Loading embeddings...")
    load_start = time.time()
    x = np.load(args.emb).astype("float32")
    if args.n is not None:
        x = x[:args.n]
    x = l2_normalize(x)
    metrics["Embedding_Load_Time_sec"] = round(time.time() - load_start, 4)

    n, dim = x.shape
    metrics["Embedding_Shape"] = str(x.shape)

    log_step("Loading meta...")
    meta_start = time.time()
    metas = load_meta(args.meta, n=n)
    metrics["Meta_Load_Time_sec"] = round(time.time() - meta_start, 4)

    log_step("Building HNSW index...")
    index = hnswlib.Index(space="cosine", dim=dim)

    build_start = time.time()
    index.init_index(
        max_elements=n,
        M=args.m,
        ef_construction=args.ef_construction,
        random_seed=42,
    )
    index.add_items(x, np.arange(n))
    build_time = time.time() - build_start
    metrics["HNSW_Build_Time_sec"] = round(build_time, 4)

    index.set_ef(args.ef)

    if args.save_index:
        log_step("Saving HNSW index...")
        args.out_index.parent.mkdir(parents=True, exist_ok=True)
        index.save_index(str(args.out_index))
        metrics["Output_Index"] = str(args.out_index)

    log_step("Querying HNSW neighbors...")
    query_start = time.time()
    labels, distances = index.knn_query(x, k=args.k + 1)
    query_time = time.time() - query_start
    metrics["HNSW_Query_Time_sec"] = round(query_time, 4)

    log_step("Making candidate pairs...")
    pair_start = time.time()

    pair_sims = {}

    for i in range(n):
        for j, d in zip(labels[i], distances[i]):
            j = int(j)

            if i == j or j < 0:
                continue

            sim = 1.0 - float(d)

            if sim < args.sim_threshold:
                continue

            a, b = sorted((i, j))
            old = pair_sims.get((a, b))
            if old is None or sim > old:
                pair_sims[(a, b)] = sim

    rows = []

    for (a, b), sim in pair_sims.items():
        meta_a = metas[a] if a < len(metas) else {}
        meta_b = metas[b] if b < len(metas) else {}

        rows.append({
            "id1": a,
            "id2": b,
            "cosine_sim": sim,
            "source1": meta_a.get("source", "unknown"),
            "source2": meta_b.get("source", "unknown"),
            "len1": meta_a.get("text_len", None),
            "len2": meta_b.get("text_len", None),
        })

    pairs = pd.DataFrame(rows)

    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out_pairs, index=False)

    metrics["Candidate_Pair_Make_Time_sec"] = round(time.time() - pair_start, 4)
    metrics["Candidate_Pairs"] = len(pairs)
    metrics["Output_Pairs"] = str(args.out_pairs)
    metrics["Total_Core_Execution_Time_sec"] = round(time.time() - core_start_time, 4)

    save_resource_data(metrics)

    print("\n=== Candidate Summary ===")
    print(f"N = {n}")
    print(f"candidate pairs = {len(pairs)}")
    if len(pairs) > 0:
        print(pairs["cosine_sim"].describe())

    end_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT FINISHED\n"
        + "=" * 60 + "\n"
    )
    print(end_msg)


if __name__ == "__main__":
    main()
