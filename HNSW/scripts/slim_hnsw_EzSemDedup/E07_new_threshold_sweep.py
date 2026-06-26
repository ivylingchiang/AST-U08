import json
import time
import psutil
import torch
import os

from pathlib import Path
from datetime import datetime

import pandas as pd

# ============================================================
# Config
# ============================================================

ROOT = Path("/home/u08/workspace/HNSW")

MODEL = "sentence-transformers_all-MiniLM-L6-v2"

THRESHOLDS = [
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.92,
]

DATA_DIR = ROOT / "data"

SAMPLE = DATA_DIR / "samples" / "slimpajama_100k.jsonl"

INPUT_PAIR_PATH = (
    ROOT
    / "data"
    / "hnsw_candidate_pairs"
    / "slim_hnsw_EzSemDedup"
    / f"{MODEL}.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "reports"
    / "slim_hnsw_EzSemDedup"
)

LOG_DIR = (
    ROOT
    / "log"
    / "slim_hnsw_EzSemDedup"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Resource Logger
# ============================================================

class ResourceLogger:

    def __init__(self, log_dir):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_file = (
            log_dir
            / f"E07_resource_usage_{timestamp}.jsonl"
        )

        self.start_time = time.time()

    def log(self, phase):

        usage = {
            "phase": phase,
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(),
            "memory_gb":
                psutil.Process(
                    os.getpid()
                ).memory_info().rss / (1024 ** 3),
        }

        if torch.cuda.is_available():
            usage["gpu_memory_gb"] = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

        with open(self.log_file, "a") as f:
            f.write(json.dumps(usage) + "\n")


# ============================================================
# DSU
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

        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.p[rb] = ra


# ============================================================
# Utils
# ============================================================

def count_rows(path):

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return sum(1 for _ in f)


# ============================================================
# Main
# ============================================================

def main():

    total_start_time = time.time()

    logger = ResourceLogger(LOG_DIR)

    print(f"Loading sample: {SAMPLE}")

    sample_load_start = time.time()

    n = count_rows(SAMPLE)

    sample_load_time = (
        time.time()
        - sample_load_start
    )

    print(f"N = {n:,}")

    print(f"Loading pairs: {INPUT_PAIR_PATH}")

    pair_load_start = time.time()

    pairs = pd.read_parquet(INPUT_PAIR_PATH)

    pair_load_time = (
        time.time()
        - pair_load_start
    )

    logger.log("data_loaded")

    results = []

    for threshold in THRESHOLDS:

        print(
            f"\nProcessing threshold = {threshold:.2f}"
        )

        threshold_start = time.time()

        # ------------------------------------------
        # threshold filtering
        # ------------------------------------------

        sub = pairs[
            pairs["cosine_sim"] >= threshold
        ]

        n_pairs_after_threshold = len(sub)

        # ------------------------------------------
        # union-find
        # ------------------------------------------

        union_start = time.time()

        dsu = DSU(n)

        for r in sub.itertuples(index=False):

            dsu.union(
                int(r.id1),
                int(r.id2),
            )

        union_time_sec = (
            time.time()
            - union_start
        )

        # ------------------------------------------
        # build components
        # ------------------------------------------

        comps = {}

        for i in range(n):

            root = dsu.find(i)

            comps.setdefault(
                root,
                []
            ).append(i)

        component_sizes = [
            len(c)
            for c in comps.values()
        ]

        duplicate_components = [
            c
            for c in comps.values()
            if len(c) > 1
        ]

        # ------------------------------------------
        # dedup stats
        # Easy SemDedup:
        # one component -> keep one
        # ------------------------------------------

        n_components = len(comps)

        n_keep = n_components

        n_drop = n - n_keep

        drop_rate = (
            n_drop / n
            if n > 0
            else 0.0
        )

        max_component_size = max(
            component_sizes
        )

        avg_component_size = (
            sum(component_sizes)
            / len(component_sizes)
        )

        threshold_time_sec = (
            time.time()
            - threshold_start
        )

        result = {

            # ----------------------------------
            # compatible with Improve SemDedup
            # ----------------------------------

            "method":
                "hnsw_candidates_plus_easy_semdedup",

            "n_total":
                n,

            "sim_threshold":
                threshold,

            "n_pairs_after_threshold":
                n_pairs_after_threshold,

            "n_keep":
                n_keep,

            "n_drop":
                n_drop,

            "drop_rate":
                round(drop_rate, 8),

            # ----------------------------------
            # graph statistics
            # ----------------------------------

            "n_components":
                n_components,

            "n_duplicate_components":
                len(
                    duplicate_components
                ),

            "max_component_size":
                max_component_size,

            "avg_component_size":
                round(
                    avg_component_size,
                    4
                ),

            # ----------------------------------
            # runtime
            # ----------------------------------

            "union_time_sec":
                round(
                    union_time_sec,
                    4
                ),

            "threshold_time_sec":
                round(
                    threshold_time_sec,
                    4
                ),
        }

        results.append(result)

        logger.log(
            f"threshold_{threshold}_done"
        )

        print(
            f"pairs={n_pairs_after_threshold:,} "
            f"drop={n_drop:,} "
            f"drop_rate={drop_rate:.4f}"
        )

    # =====================================================
    # Save CSV
    # =====================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = (
        OUTPUT_DIR
        / f"{MODEL}_threshold_sweep_{timestamp}.csv"
    )

    pd.DataFrame(results).to_csv(
        csv_path,
        index=False
    )

    # =====================================================
    # Save JSON
    # =====================================================

    json_path = (
        OUTPUT_DIR
        / f"{MODEL}_threshold_sweep_{timestamp}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.log("finished")

    total_time_sec = (
        time.time()
        - total_start_time
    )

    print("\n===================================")
    print("Easy SemDedup Threshold Sweep Done")
    print("===================================")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print(f"Total Time = {total_time_sec:.2f}s")


if __name__ == "__main__":
    main()