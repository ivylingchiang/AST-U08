#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import random
import time
import json # <--- 新增這個


# ============================================================
# Config
# ============================================================

ROOT = Path("/home/u08/workspace/HNSW")

EMB_PATH = ROOT / "data/embeddings/slim_hnsw_EzSemDedup/sentence-transformers_all-MiniLM-L6-v2_emb.npy"

PAIR_PATH = ROOT / "data/hnsw_candidate_pairs/slim_hnsw_EzSemDedup/sentence-transformers_all-MiniLM-L6-v2.parquet"

N = 50000

# ground truth similarity threshold (IMPORTANT)
GT_THRESHOLD = 0.9

# sample control (to avoid O(N^2))
SAMPLE_SIZE = None
JSONL_PATH = ROOT / "data/EzSemDedup/slimpajama_100k_dedup_088.jsonl" 

# ============================================================
# Utils
# ============================================================
import json

def load_surviving_ids(jsonl_path, max_n):
    """讀取最終存活下來的檔案 ID"""
    surviving_ids = set()
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_id = None
            
            # 嘗試從常見的 JSON 欄位中提取原始 ID
            if "id" in data:
                doc_id = int(data["id"])
            elif "meta" in data and "id" in data["meta"]:
                doc_id = int(data["meta"]["id"])
            elif "doc_id" in data:
                doc_id = int(data["doc_id"])
            
            # 如果成功抓到 ID，且在我們評估的 N 範圍內
            if doc_id is not None and doc_id < max_n:
                surviving_ids.add(doc_id)
                
    # 如果完全沒有抓到 ID，印出警告
    if len(surviving_ids) == 0:
        print("⚠️ 警告：無法在 jsonl 中找到對應的 ID 欄位，請檢查 JSON 格式！")
        
    return surviving_ids

def e2e_dedup_recall(gt_pairs, surviving_ids):
    """計算端到端 (End-to-End) 去重召回率"""
    if not gt_pairs:
        return 0.0, 0, 0
        
    leaks = 0  # 兩個都存活下來了 (漏網之魚，去重失敗)
    caught = 0 # 至少刪掉了一個 (成功去重)
    
    for a, b in gt_pairs:
        if a in surviving_ids and b in surviving_ids:
            leaks += 1
        else:
            caught += 1
            
    return caught / len(gt_pairs), caught, leaks

def build_ground_truth_pairs(x, threshold=0.95, sample_size=None):
    """
    Build GT duplicate pairs using cosine similarity.
    If sample_size is given, only sample subset for efficiency.
    """

    n = len(x)

    if sample_size is not None and sample_size < n:
        idx = np.random.choice(n, sample_size, replace=False)
        x = x[idx]
        index_map = idx
    else:
        index_map = np.arange(n)

    sim = cosine_similarity(x, x)
    np.fill_diagonal(sim, -1.0)
    print(f"DEBUG: 目前這批資料的最大相似度是 {np.max(sim):.4f}") # 加這一行看看

    gt_pairs = set()

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if sim[i, j] >= threshold:
                gt_pairs.add((index_map[i], index_map[j]))

    return gt_pairs


def load_pred_pairs(pair_path, threshold, max_n):
    df = pd.read_parquet(pair_path)

    df = df[df["cosine_sim"] >= threshold].copy()

    pairs = set()
    for a, b in zip(df.id1, df.id2):
        a_id, b_id = int(a), int(b)
        # 關鍵修正：只保留 ID 小於 max_n (20000) 的配對
        if a_id < max_n and b_id < max_n:
            pairs.add((min(a_id, b_id), max(a_id, b_id)))

    return pairs


def pair_recall(gt_pairs, pred_pairs):
    tp = len(gt_pairs & pred_pairs)
    fn = len(gt_pairs - pred_pairs)

    if tp + fn == 0:
        return 0.0, 0, 0

    return tp / (tp + fn), tp, fn


# ============================================================
# Cluster Analysis (DSU)
# ============================================================

class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_clusters(n, pairs):
    dsu = DSU(n)

    for a, b in pairs:
        dsu.union(a, b)

    clusters = {}
    for i in range(n):
        r = dsu.find(i)
        clusters.setdefault(r, []).append(i)

    return list(clusters.values())


def cluster_recall(gt_pairs, pred_pairs, n):
    """
    Cluster recall: whether GT-connected components are preserved
    Approx version using DSU on GT graph vs pred graph
    """

    gt_clusters = build_clusters(n, gt_pairs)
    pred_clusters = build_clusters(n, pred_pairs)

    # map node -> cluster id
    def build_map(clusters):
        m = {}
        for cid, c in enumerate(clusters):
            for x in c:
                m[x] = cid
        return m

    gt_map = build_map(gt_clusters)
    pred_map = build_map(pred_clusters)

    # cluster consistency: pairs in GT should not be split in pred
    correct = 0
    total = 0

    for a, b in gt_pairs:
        total += 1
        if pred_map.get(a) == pred_map.get(b):
            correct += 1

    return correct / total if total > 0 else 0.0


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Dedup Recall Evaluation (EzSemDedup + HNSW)")
    print("=" * 60)

    print("Loading embeddings...")
    x = np.load(EMB_PATH).astype("float32")[:N]
    n = len(x)

    print(f"Dataset size: {n}")

    # --------------------------------------------------------
    # 1. Ground Truth
    # --------------------------------------------------------

    print("\nBuilding ground truth pairs...")

    t0 = time.time()
    gt_pairs = build_ground_truth_pairs(
        x,
        threshold=GT_THRESHOLD,
        sample_size=SAMPLE_SIZE
    )
    print(f"GT pairs: {len(gt_pairs):,} (took {time.time() - t0:.2f}s)")

    # --------------------------------------------------------
    # 2. Predicted pairs (HNSW + threshold)
    # --------------------------------------------------------

    print("\nLoading predicted pairs...")

    t0 = time.time()
    pred_pairs = load_pred_pairs(PAIR_PATH, threshold=GT_THRESHOLD, max_n=n)
    print(f"Pred pairs: {len(pred_pairs):,} (took {time.time() - t0:.2f}s)")

    # --------------------------------------------------------
    # 3. Pair Recall
    # --------------------------------------------------------

    print("\nComputing pair recall...")

    recall, tp, fn = pair_recall(gt_pairs, pred_pairs)

    print(f"Pair Recall: {recall:.6f}")
    print(f"TP: {tp:,}, FN: {fn:,}")

    # --------------------------------------------------------
    # 4. Cluster Recall
    # --------------------------------------------------------

    print("\nComputing cluster recall...")

    cluster_r = cluster_recall(gt_pairs, pred_pairs, n)

    print(f"Cluster Recall: {cluster_r:.6f}")

    # --------------------------------------------------------
    # 5. Compression stats
    # --------------------------------------------------------

    drop_rate = 1.0 - (len(pred_pairs) / (n * 10))  # rough proxy

    print("\nCompression:")
    print(f"Drop Rate (approx): {drop_rate:.4f}")
    
    # --------------------------------------------------------
    # 7. End-to-End SemDedup Recall
    # --------------------------------------------------------

    print("\nComputing End-to-End SemDedup recall...")
    
    # 請確保檔案路徑正確，這裡假設放在 Config 區塊的 JSONL_PATH
    JSONL_PATH = ROOT / "data/EzSemDedup/slimpajama_100k_dedup_088.jsonl"
    
    t0 = time.time()
    surviving_ids = load_surviving_ids(JSONL_PATH, max_n=n)
    
    e2e_recall, e2e_tp, e2e_fn = e2e_dedup_recall(gt_pairs, surviving_ids)
    
    print(f"Surviving documents (in top {n}): {len(surviving_ids):,} (took {time.time() - t0:.2f}s)")
    print(f"End-to-End Recall: {e2e_recall:.6f}")
    print(f"Successfully deduplicated (TP): {e2e_tp:,}, Leaks (FN): {e2e_fn:,}")

    # --------------------------------------------------------
    # 8. Save report (更新版)
    # --------------------------------------------------------

    report = {
        "N": n,
        "GT_THRESHOLD": GT_THRESHOLD,
        "SAMPLE_SIZE": SAMPLE_SIZE,
        "gt_pairs": len(gt_pairs),
        "pred_pairs": len(pred_pairs),
        "pair_recall": recall,
        "cluster_recall": cluster_r,
        "tp": tp,
        "fn": fn,
        "drop_rate_proxy": drop_rate,
        # 新增 End-to-End 評估指標
        "e2e_recall": e2e_recall,
        "e2e_tp": e2e_tp,     # 成功去重的數量
        "e2e_fn": e2e_fn      # 漏掉沒去重的數量
    }

    out = ROOT / "reports/slim_hnsw_EzSemDedup/dedup_recall_report.csv"
    pd.DataFrame([report]).to_csv(out, index=False)

    print("\nSaved:", out)
    print("\nDone.")



if __name__ == "__main__":
    main()