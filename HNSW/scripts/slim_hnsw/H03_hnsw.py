import time
from datetime import datetime
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# data/
# ├── embeddings/
# │   └── slim_hnsw/
# │       ├── sentence-transformers_all-MiniLM-L6-v2_emb.npy
# │       └── sentence-transformers_all-MiniLM-L6-v2_meta.jsonl
# │
# └── hnsw_index/
#     └── 03_hnsw_index.bin

# reports/
# └── slim_hnsw/
#     └── hnsw_results.csv

# log/
# └── slim_hnsw/
#     └── resource_data/
#         └── hnsw_20260609_235959.txt

# ==================================================
# Global Config
# ==================================================

ROOT = Path("/home/u08/workspace/HNSW")

DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "log"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")

# -------------------------
# Input
# -------------------------

EMB_PATH = (
    DATA_DIR
    / "embeddings"
    / "slim_hnsw"
    / f"{SAFE_MODEL_NAME}_emb.npy"
)

META_PATH = (
    DATA_DIR
    / "embeddings"
    / "slim_hnsw"
    / f"{SAFE_MODEL_NAME}_meta.jsonl"
)

# -------------------------
# Output
# -------------------------

INDEX_DIR = DATA_DIR / "hnsw_index"
INDEX_PATH = INDEX_DIR / "hnsw_index.bin"

RESULT_DIR = REPORT_DIR / "slim_hnsw"
RESULT_PATH = RESULT_DIR / "hnsw_results.csv"

RESOURCE_DIR = (
    LOG_DIR
    / "slim_hnsw"
    / "resource_data"
)

# -------------------------
# HNSW Parameters
# -------------------------

M = 16
EF_CONSTRUCTION = 100
EF = 100

N = 20_000
K = 10

# ==================================================


def exact_knn(x, k):
    """
    Brute-force cosine similarity search.
    作為 Ground Truth。
    """
    sim = cosine_similarity(x, x)

    np.fill_diagonal(sim, -1.0)

    return np.argsort(-sim, axis=1)[:, :k]


def recall_at_k(exact, approx):
    scores = []

    for e, a in zip(exact, approx):
        scores.append(
            len(set(e) & set(a)) / len(e)
        )

    return float(np.mean(scores))


def main():

    print("=" * 60)
    print("Slim HNSW Build")
    print("=" * 60)

    print(f"Model      : {MODEL_NAME}")
    print(f"Embeddings : {EMB_PATH}")
    print(f"Metadata   : {META_PATH}")
    print()

    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"找不到 embedding 檔案：\n{EMB_PATH}"
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"找不到 metadata 檔案：\n{META_PATH}"
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # Load Embeddings
    # ==================================================

    print("Loading embeddings...")

    x_all = np.load(EMB_PATH).astype("float32")

    if len(x_all) < N:
        raise ValueError(
            f"Embedding 數量不足："
            f"{len(x_all)} < {N}"
        )

    x = x_all[:N]

    print(f"Embedding Shape: {x.shape}")
    print()

    # ==================================================
    # Exact Search
    # ==================================================

    print("Running Exact KNN...")

    t0 = time.time()

    exact = exact_knn(x, K)

    exact_time = time.time() - t0

    print(f"Exact Time: {exact_time:.4f} sec")
    print()

    # ==================================================
    # Build HNSW
    # ==================================================

    n, dim = x.shape

    print("Building HNSW Index...")

    index = hnswlib.Index(
        space="cosine",
        dim=dim,
    )

    t0 = time.time()

    index.init_index(
        max_elements=n,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=42,
    )

    index.add_items(
        x,
        np.arange(n),
    )

    build_time = time.time() - t0

    print(f"Build Time: {build_time:.4f} sec")

    # ==================================================
    # Save Index
    # ==================================================

    index.save_index(str(INDEX_PATH))

    print(f"Index Saved: {INDEX_PATH}")
    print()

    # ==================================================
    # Query
    # ==================================================

    index.set_ef(EF)

    print("Running HNSW Query...")

    t0 = time.time()

    labels, distances = index.knn_query(
        x,
        k=K + 1,
    )

    query_time = time.time() - t0

    approx = []

    for i, row in enumerate(labels):

        keep = row[row != i][:K]

        approx.append(keep)

    approx = np.vstack(approx)

    recall = recall_at_k(
        exact,
        approx,
    )

    qps = n / query_time

    # ==================================================
    # Save CSV
    # ==================================================

    results = {
        "model": MODEL_NAME,
        "N": n,
        "K": K,
        "M": M,
        "ef_construction": EF_CONSTRUCTION,
        "ef": EF,
        "recall_at_10": recall,
        "exact_time_sec": exact_time,
        "build_time_sec": build_time,
        "query_time_sec": query_time,
        "qps": qps,
    }

    df = pd.DataFrame([results])

    df.to_csv(
        RESULT_PATH,
        index=False,
    )

    print(f"Results Saved: {RESULT_PATH}")
    print()

    # ==================================================
    # Resource Log (已重構優化)
    # ==================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESOURCE_DIR / f"hnsw_{timestamp}.txt"

    # 使用多行字串，排版直接對齊左側（開頭的 \ 可以防止多出一個空行）
    log_content = f"""\
Timestamp: {timestamp}

Model: {MODEL_NAME}

Dataset Size: {n}

HNSW Parameters
M={M}
ef_construction={EF_CONSTRUCTION}
ef={EF}

Results
Recall@10={recall:.6f}
Exact Time={exact_time:.4f}
Build Time={build_time:.4f}
Query Time={query_time:.4f}
QPS={qps:.2f}
"""

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"Resource Log Saved: {log_path}")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()