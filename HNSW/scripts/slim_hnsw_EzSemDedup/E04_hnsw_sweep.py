import time
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import hnswlib
from sklearn.metrics.pairwise import cosine_similarity

# 定義路徑
ROOT = Path("/home/u08/workspace/HNSW")
DATA_DIR = ROOT / "data"
REPORT_DIR = DATA_DIR / "reports" / "slim_hnsw_EzSemDedup"
LOG_DIR = ROOT / "log" / "slim_hnsw_EzSemDedup" / "resource_data"
EMB_DIR = DATA_DIR / "embeddings" / "slim_hnsw_EzSemDedup"

# 確保目錄存在
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 設定 Log (只保留 Console 輸出，取消 FileHandler 以免污染自訂格式的 Log 檔)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = LOG_DIR / f"E_sweep_log_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# 參數設定
MODEL_FILE = "sentence-transformers_all-MiniLM-L6-v2_emb.npy"
EMB_PATH = EMB_DIR / MODEL_FILE
N = 20_000
K = 10
M = 16
EF_CONSTRUCTION = 100
EF = 100

def exact_knn(x, k):
    sim = cosine_similarity(x, x)
    np.fill_diagonal(sim, -1.0)
    return np.argsort(-sim, axis=1)[:, :k]

def recall_at_k(exact, approx):
    return float(np.mean([len(set(e) & set(a)) / len(e) for e, a in zip(exact, approx)]))

def main():
    logging.info(f"Loading embeddings from: {EMB_PATH}")
    if not EMB_PATH.exists():
        logging.error(f"找不到檔案: {EMB_PATH}")
        return

    x_all = np.load(EMB_PATH).astype("float32")
    x = x_all[:N]
    n, dim = x.shape
    
    logging.info(f"Loaded {n} embeddings. Starting evaluation with M={M}, ef_c={EF_CONSTRUCTION}, ef={EF}")

    # Exact KNN
    t0 = time.time()
    exact = exact_knn(x, K)
    exact_time = time.time() - t0
    logging.info(f"Exact KNN finished in {exact_time:.4f}s")

    # HNSW
    index = hnswlib.Index(space="cosine", dim=dim)
    t0 = time.time()
    index.init_index(max_elements=n, M=M, ef_construction=EF_CONSTRUCTION, random_seed=42)
    index.add_items(x, np.arange(n))
    build_time = time.time() - t0

    index.set_ef(EF)
    t0 = time.time()
    labels, _ = index.knn_query(x, k=K + 1)
    query_time = time.time() - t0

    # 排除自己 (i) 後取前 K 個
    approx = [row[row != i][:K] for i, row in enumerate(labels)]
    recall = recall_at_k(exact, np.array(approx))
    qps = n / query_time

    # 彙整結果
    results = {
        "N": n, "K": K, "M": M, "ef_construction": EF_CONSTRUCTION, "ef": EF,
        "recall": recall, "build_time": build_time, "query_time": query_time, "qps": qps
    }

    # 儲存 CSV
    report_path = REPORT_DIR / f"sweep_report_{timestamp}.csv"
    pd.DataFrame([results]).to_csv(report_path, index=False)
    
    # 建立自訂的 Log 內容
    model_name_formatted = MODEL_FILE.replace("_emb.npy", "").replace("_", "/", 1)
    
    custom_log_content = f"""Timestamp: {timestamp}

Model: {model_name_formatted}

Dataset Size: {n}

HNSW Parameters
M={M}
ef_construction={EF_CONSTRUCTION}
ef={EF}

Results
Recall@{K}={recall:.6f}
Exact Time={exact_time:.4f}
Build Time={build_time:.4f}
Query Time={query_time:.4f}
QPS={qps:.2f}
"""

    # 寫入取代原本的 Log 檔
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(custom_log_content)

    logging.info(f"Evaluation finished. Recall: {recall:.4f}, QPS: {qps:.2f}")
    logging.info(f"Report saved to {report_path}")
    logging.info(f"Log saved to {log_filename}")

if __name__ == "__main__":
    main()