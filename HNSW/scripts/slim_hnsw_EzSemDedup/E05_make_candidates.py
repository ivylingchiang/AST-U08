import json
import time
import psutil
import os
from pathlib import Path
import numpy as np
import hnswlib
import pandas as pd
from datetime import datetime

# ============================================================
# Config & Paths
# ============================================================

ROOT = Path("/home/u08/workspace/HNSW")
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "hnsw_candidate_pairs" / "slim_hnsw_EzSemDedup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Log 路徑
# LOG_DIR = ROOT / "log" / "slim_hnsw_EzSemDedup"
# LOG_DIR.mkdir(parents=True, exist_ok=True)
# LOG_FILE = LOG_DIR / "E05_resource_data.json"
# Log 路徑
LOG_DIR = ROOT / "log" / "slim_hnsw_EzSemDedup"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 2. 建立時間戳記字串 (格式範例: 20260611_182005)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 3. 將時間戳記插入檔名
LOG_FILE = LOG_DIR / f"E05_resource_data_{timestamp}.json"

EMB = Path("/home/u08/workspace/HNSW/data/embeddings/slim_hnsw_EzSemDedup/sentence-transformers_all-MiniLM-L6-v2_emb.npy")
META = Path("/home/u08/workspace/HNSW/data/embeddings/slim_hnsw_EzSemDedup/sentence-transformers_all-MiniLM-L6-v2_meta.jsonl")
OUT_FILE = OUT_DIR / "sentence-transformers_all-MiniLM-L6-v2.parquet"

K = 50
SIM_THRESHOLD = 0.80
M = 16
EF_CONSTRUCTION = 100
EF = 100

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def build_candidate_pairs():
    start_time = time.time()
    mem_start = get_memory_usage()
    
    print(f"Loading data from {EMB}...")
    x = np.load(EMB).astype("float32")
    metas = []
    with META.open("r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    n, dim = x.shape

    # Build Index
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, M=M, ef_construction=EF_CONSTRUCTION, random_seed=42)
    index.add_items(x, np.arange(n))
    index.set_ef(EF)

    # Query
    labels, distances = index.knn_query(x, k=K + 1)

    # Extraction
    pairs = []
    seen = set()
    for i in range(n):
        for j, d in zip(labels[i], distances[i]):
            if i >= j: continue
            sim = 1.0 - float(d)
            if sim >= SIM_THRESHOLD:
                pairs.append({"id1": i, "id2": j, "cosine_sim": sim})

    df = pd.DataFrame(pairs)
    df.to_parquet(OUT_FILE, index=False)

    # Logging
    end_time = time.time()
    mem_end = get_memory_usage()
    
    log_data = {
        "duration_seconds": round(end_time - start_time, 2),
        "peak_memory_mb": round(mem_end, 2),
        "total_pairs_found": len(df),
        "status": "success"
    }
    
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)
        
    print(f"Done. Log saved to {LOG_FILE}")
    print(log_data)

if __name__ == "__main__":
    build_candidate_pairs()