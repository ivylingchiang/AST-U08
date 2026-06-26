import time
import json
import numpy as np
import hnswlib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity



# 專案結構：
# HNSW/
# ├── data/
# │   └── embeddings/
# │       └── slim_hnsw_EzSemDedup/
# │           ├── sentence-transformers_all-MiniLM-L6-v2_emb.npy
# │           └── sentence-transformers_all-MiniLM-L6-v2_meta.jsonl
# └── log/
#     └── slim_hnsw_EzSemDedup/
#         └── resource_data/
#             └── (儲存 metrics_時間.json 與 Eslim_eval_時間.log)

# 輸入：
# - data/embeddings/slim_hnsw_EzSemDedup/sentence-transformers_all-MiniLM-L6-v2_emb.npy

# 輸出：
# - log/slim_hnsw_EzSemDedup/resource_data/Eslim_eval_*.log
# - log/slim_hnsw_EzSemDedup/resource_data/metrics_*.json


# 定義路徑
ROOT = Path("/home/u08/workspace/HNSW")
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "log" / "slim_hnsw_EzSemDedup" / "resource_data"
EMB_DIR = DATA_DIR / "embeddings" / "slim_hnsw_EzSemDedup"

# 確保目錄存在
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 設定 Log (文字紀錄)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = LOG_DIR / f"Eslim_eval_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

# 參數設定
MODEL_FILE_NAME = "sentence-transformers_all-MiniLM-L6-v2_emb.npy"
EMB_PATH = EMB_DIR / MODEL_FILE_NAME
N_EVAL = 10_000
K = 10

def exact_knn(x, k):
    sim = cosine_similarity(x, x)
    np.fill_diagonal(sim, -1.0)
    return np.argsort(-sim, axis=1)[:, :k]

def hnsw_knn(x, k, M=16, ef_construction=200, ef=100):
    n, dim = x.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, M=M, ef_construction=ef_construction, random_seed=42)
    index.add_items(x, np.arange(n))
    index.set_ef(ef)

    labels, _ = index.knn_query(x, k=k + 1)

    out = []
    for i, row in enumerate(labels):
        keep = row[row != i][:k]
        out.append(keep)
    return np.vstack(out)

def recall_at_k(exact, approx):
    scores = []
    for e, a in zip(exact, approx):
        scores.append(len(set(e) & set(a)) / len(e))
    return float(np.mean(scores))

def main():
    logging.info(f"Loading embeddings from: {EMB_PATH}")
    
    if not EMB_PATH.exists():
        logging.error(f"找不到 embedding 檔案：{EMB_PATH}")
        return

    x = np.load(EMB_PATH).astype("float32")[:N_EVAL]
    
    # 執行 Exact KNN
    logging.info("Starting Exact KNN calculation...")
    t0 = time.time()
    exact = exact_knn(x, K)
    exact_time = time.time() - t0

    # 執行 HNSW KNN
    logging.info("Starting HNSW KNN calculation...")
    t0 = time.time()
    approx = hnsw_knn(x, K, M=16, ef_construction=200, ef=100)
    hnsw_time = time.time() - t0

    # 計算結果
    recall = recall_at_k(exact, approx)
    
    results = {
        "timestamp": timestamp,
        "N": N_EVAL,
        "K": K,
        "recall@k": recall,
        "exact_time_sec": exact_time,
        "hnsw_build_plus_query_sec": hnsw_time,
    }

    # 將數據存為 JSON 方便後續讀取
    json_path = LOG_DIR / f"metrics_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Evaluation finished. Metrics saved to {json_path}")
    logging.info(f"Recall@K: {recall:.4f}")

if __name__ == "__main__":
    main()