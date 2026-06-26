import json
import time
import psutil
import torch
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# ============================================================
# Config & Paths
# ============================================================
ROOT = Path("/home/u08/workspace/HNSW")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")

DATA_DIR = ROOT / "data"
SAMPLE_FILE = DATA_DIR / "samples" / "slimpajama_100k.jsonl"
INPUT_PAIR_DIR = ROOT / "data" / "hnsw_candidate_pairs" / "slim_hnsw_EzSemDedup"
OUTPUT_DIR = ROOT / "reports" / "slim_hnsw_EzSemDedup"
LOG_DIR = ROOT / "log" / "slim_hnsw_EzSemDedup"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SIM_THRESHOLDS = [0.88, 0.90, 0.92]

# ============================================================
# Resource Tracker
# ============================================================
class ResourceTracker:
    def __init__(self, log_path):
        self.log_path = log_path
        self.start_time = time.time()

    def log(self, phase):
        usage = {
            "phase": phase,
            "elapsed_time": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(),
            "memory_usage_gb": psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        }
        if torch.cuda.is_available():
            usage["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            usage["gpu_name"] = torch.cuda.get_device_name(0)
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(usage) + "\n")

# ============================================================
# DSU & Utils (保持不變)
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

def read_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def quality_score(row):
    return min(row.get("text_len", len(row["text"])), 8000)

# ============================================================
# Core
# ============================================================
def run_semdedup(rows, pairs, sim_threshold):
    n = len(rows)
    pairs = pairs[pairs["cosine_sim"] >= sim_threshold]
    dsu = DSU(n)
    for r in pairs.itertuples(index=False):
        dsu.union(int(r.id1), int(r.id2))

    comps = {}
    for i in range(n):
        root = dsu.find(i)
        comps.setdefault(root, []).append(i)

    keep, drop = set(), set()
    for comp in comps.values():
        if len(comp) == 1:
            keep.add(comp[0])
            continue
        best = max(comp, key=lambda idx: quality_score(rows[idx]))
        keep.add(best)
        for idx in comp:
            if idx != best: drop.add(idx)

    return {
        "sim_threshold": sim_threshold,
        "n_total": n,
        "n_candidate_pairs": int(len(pairs)),
        "n_keep": len(keep),
        "n_drop": len(drop),
        "drop_rate": len(drop) / n if n > 0 else 0.0
    }

def main():
    # tracker = ResourceTracker(LOG_DIR / "E06_resource_usage.jsonl")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"E06_resource_usage_{timestamp}.jsonl"
    tracker = ResourceTracker(LOG_DIR / log_filename)
    
    # 讀取輸入
    input_file = INPUT_PAIR_DIR / f"{SAFE_MODEL_NAME}.parquet"
    print(f"Loading {input_file}...")
    rows = read_rows(SAMPLE_FILE)
    pairs = pd.read_parquet(input_file)
    tracker.log("data_loading_complete")

    results = []
    for sim_threshold in SIM_THRESHOLDS:
        report = run_semdedup(rows, pairs, sim_threshold)
        results.append(report)
        print(f"Finished threshold: {sim_threshold}")
    
    tracker.log("computation_complete")

    # 儲存輸出
    output_file = OUTPUT_DIR / f"{SAFE_MODEL_NAME}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()