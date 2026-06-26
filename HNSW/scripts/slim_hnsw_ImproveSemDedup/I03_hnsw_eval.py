import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import hnswlib
import numpy as np
import pandas as pd
from tqdm import tqdm


# 輸入：
#   data/embeddings/slimpajama_100k_emb.npy
#
# 輸出：
#   data/reports/hnsw_eval.csv
#   log/slim_hnsw_ImproveSemDedup/resource_data/slim_hnsw_eval_YYYYMMDD_HHMMSS.txt


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_EMB = ROOT / "data" / "embeddings" / "slimpajama_100k_emb.npy"
DEFAULT_REPORT = ROOT / "data" / "reports" / "hnsw_eval.csv"

DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = Path("/home/u08/workspace/HNSW/log/slim_hnsw_ImproveSemDedup/resource_data")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


def log_step(msg):
    print(f"[*] {msg}")


def save_resource_data(metrics):
    log_step("Saving resource metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = RESOURCE_DIR / f"Islim_hnsw_eval_{timestamp}.txt"

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


def exact_topk_cosine(x, k, block_size):
    n = x.shape[0]
    neighbors = np.empty((n, k), dtype=np.int32)

    for start in tqdm(range(0, n, block_size), desc="Exact top-k"):
        end = min(start + block_size, n)
        sim = x[start:end] @ x.T

        rows = np.arange(start, end)
        sim[np.arange(end - start), rows] = -np.inf

        idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
        vals = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)

        neighbors[start:end] = idx.astype(np.int32)

    return neighbors


def recall_at_k(exact, approx, k):
    scores = []
    for e, a in zip(exact, approx):
        scores.append(len(set(e[:k].tolist()) & set(a[:k].tolist())) / k)
    return float(np.mean(scores))


def run_hnsw_once(x, exact, k, m, ef_construction, ef):
    n, dim = x.shape

    index = hnswlib.Index(space="cosine", dim=dim)

    build_start = time.time()
    index.init_index(
        max_elements=n,
        M=m,
        ef_construction=ef_construction,
        random_seed=42,
    )
    index.add_items(x, np.arange(n))
    build_time = time.time() - build_start

    index.set_ef(ef)

    query_start = time.time()
    labels, _ = index.knn_query(x, k=k + 1)
    query_time = time.time() - query_start

    approx = []
    for i, row in enumerate(labels):
        row = row[row != i][:k]
        approx.append(row)

    approx = np.vstack(approx)

    return {
        "N": n,
        "K": k,
        "M": m,
        "ef_construction": ef_construction,
        "ef": ef,
        "recall_at_10": recall_at_k(exact, approx, k),
        "build_time_sec": build_time,
        "query_time_sec": query_time,
        "qps": n / query_time if query_time > 0 else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exact-block-size", type=int, default=2048)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_REPORT)

    # 固定參數模式：
    # run_all 會使用：
    #   --fixed --m 16 --ef-construction 100 --ef 100
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef", type=int, default=100)

    return parser.parse_args()


def build_grid(args):
    """
    若 args.fixed=True，只跑指定的一組 HNSW 參數。
    否則跑完整 sweep grid。
    """
    if args.fixed:
        return [(args.m, args.ef_construction, args.ef)]

    grid = []
    for m in [8, 16, 32]:
        for efc in [100, 200, 400]:
            for ef in [50, 100, 200]:
                grid.append((m, efc, ef))

    return grid


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
    metrics["N"] = args.n
    metrics["K"] = args.k
    metrics["Fixed_Mode"] = args.fixed
    metrics["Fixed_M"] = args.m
    metrics["Fixed_ef_construction"] = args.ef_construction
    metrics["Fixed_ef"] = args.ef

    log_step("Loading embeddings...")
    load_start = time.time()
    x = np.load(args.emb).astype("float32")[:args.n]
    x = l2_normalize(x)
    metrics["Embedding_Load_Time_sec"] = round(time.time() - load_start, 4)
    metrics["Embedding_Shape"] = str(x.shape)

    log_step("Computing exact top-k...")
    exact_start = time.time()
    exact = exact_topk_cosine(x, args.k, args.exact_block_size)
    exact_time = time.time() - exact_start
    metrics["Exact_TopK_Time_sec"] = round(exact_time, 4)

    grid = build_grid(args)
    metrics["Num_HNSW_Runs"] = len(grid)

    rows = []
    for m, efc, ef in grid:
        log_step(f"Running HNSW M={m}, ef_construction={efc}, ef={ef}")
        result = run_hnsw_once(x, exact, args.k, m, efc, ef)
        result["exact_time_sec"] = exact_time
        rows.append(result)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(["recall_at_10", "qps"], ascending=[False, False])

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_report, index=False)

    metrics["Output_Report"] = str(args.out_report)

    if len(df) > 0:
        best = df.iloc[0]
        metrics["Best_Recall"] = round(float(best["recall_at_10"]), 6)
        metrics["Best_M"] = int(best["M"])
        metrics["Best_ef_construction"] = int(best["ef_construction"])
        metrics["Best_ef"] = int(best["ef"])
        metrics["Best_Build_Time_sec"] = round(float(best["build_time_sec"]), 4)
        metrics["Best_Query_Time_sec"] = round(float(best["query_time_sec"]), 4)
        metrics["Best_QPS"] = round(float(best["qps"]), 4)

    metrics["Total_Core_Execution_Time_sec"] = round(time.time() - core_start_time, 4)

    save_resource_data(metrics)

    print("\n=== HNSW Eval Result ===")
    print(df)

    end_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT FINISHED\n"
        + "=" * 60 + "\n"
    )
    print(end_msg)


if __name__ == "__main__":
    main()
