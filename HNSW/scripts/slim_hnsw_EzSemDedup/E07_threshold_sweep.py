import json
import time
import psutil
import torch
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# ============================================================
# Config
# ============================================================
ROOT = Path("/home/u08/workspace/HNSW")
MODEL = "sentence-transformers_all-MiniLM-L6-v2"
THRESHOLDS = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]

DATA_DIR = ROOT / "data"
SAMPLE = DATA_DIR / "samples" / "slimpajama_100k.jsonl"
INPUT_PAIR_PATH = ROOT / "data" / "hnsw_candidate_pairs" / "slim_hnsw_EzSemDedup" / f"{MODEL}.parquet"
OUTPUT_DIR = ROOT / "reports" / "slim_hnsw_EzSemDedup"
LOG_DIR = ROOT / "log" / "slim_hnsw_EzSemDedup" 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Resource Logger
# ============================================================
class ResourceLogger:
    def __init__(self, log_dir):
        # self.log_file = log_dir / "E07_resource_usage.jsonl"
        # self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"E07_resource_usage_{timestamp}.jsonl"
        self.start_time = time.time()

    def log(self, phase):
        usage = {
            "phase": phase,
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(),
            "memory_gb": psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        }
        if torch.cuda.is_available():
            usage["gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(usage) + "\n")

# ============================================================
# DSU & Utils
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
        if ra != rb: self.p[rb] = ra

def count_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# ============================================================
# Main Core
# ============================================================
def main():
    logger = ResourceLogger(LOG_DIR)
    
    print(f"Loading data from {INPUT_PAIR_PATH}...")
    n = count_rows(SAMPLE)
    pairs = pd.read_parquet(INPUT_PAIR_PATH)
    logger.log("data_loaded")

    results = []
    for threshold in THRESHOLDS:
        print(f"Processing threshold: {threshold}")
        sub = pairs[pairs["cosine_sim"] >= threshold]
        
        dsu = DSU(n)
        for r in sub.itertuples(index=False):
            dsu.union(int(r.id1), int(r.id2))

        comps = {}
        for i in range(n):
            comps.setdefault(dsu.find(i), []).append(i)

        n_drop = sum(len(c) - 1 for c in comps.values() if len(c) > 1)
        
        results.append({
            "threshold": threshold,
            "n_pairs": len(sub),
            "n_drop": n_drop,
            "drop_rate": n_drop / n if n > 0 else 0.0,
            "n_components": len(comps),
            "max_comp_size": max(len(c) for c in comps.values())
        })
        logger.log(f"threshold_{threshold}_done")

    # Save output
    output_path = OUTPUT_DIR / f"{MODEL}_results.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    main()