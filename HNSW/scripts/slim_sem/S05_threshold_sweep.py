import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_SAMPLE = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
DEFAULT_PAIRS = ROOT / "data" / "semdedup" / "slim_sem" / "candidate_pairs.csv"
DEFAULT_OUT = ROOT / "data" / "reports" / "slim_sem_threshold_sweep.csv"
DEFAULT_REPORT = ROOT / "log" / "slim_sem" / "S05_threshold_sweep.json"


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92],
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def count_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def component_stats(uf):
    roots = np.array([uf.find(i) for i in range(len(uf.parent))], dtype=np.int64)
    _, counts = np.unique(roots, return_counts=True)
    duplicate_counts = counts[counts > 1]

    if len(duplicate_counts) == 0:
        return {
            "n_drop": 0,
            "n_components": int(len(counts)),
            "n_duplicate_components": 0,
            "max_component_size": 1,
        }

    return {
        "n_drop": int(np.sum(duplicate_counts - 1)),
        "n_components": int(len(counts)),
        "n_duplicate_components": int(len(duplicate_counts)),
        "max_component_size": int(np.max(duplicate_counts)),
    }


def evaluate_threshold(pairs_path, threshold, n_docs, chunksize):
    uf = UnionFind(n_docs)
    n_pairs = 0

    for chunk in pd.read_csv(pairs_path, chunksize=chunksize):
        chunk = chunk[chunk["cosine_sim"] >= threshold]
        n_pairs += len(chunk)

        for row in chunk.itertuples(index=False):
            uf.union(row.doc_i, row.doc_j)

    stats = component_stats(uf)
    stats.update({
        "threshold": float(threshold),
        "n_pairs": int(n_pairs),
        "drop_rate": float(stats["n_drop"] / n_docs) if n_docs > 0 else 0.0,
    })
    return stats


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)

    if not args.sample.exists():
        raise FileNotFoundError(f"Sample file not found: {args.sample}")
    if not args.pairs.exists():
        raise FileNotFoundError(f"Candidate pairs not found: {args.pairs}")

    start = time.time()
    n_docs = count_jsonl(args.sample)
    rows = []

    print(f"Sample docs: {n_docs}")
    print(f"Candidate pairs: {args.pairs}")

    for threshold in args.thresholds:
        print(f"Evaluating threshold={threshold}")
        rows.append(evaluate_threshold(args.pairs, threshold, n_docs, args.chunksize))

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "n_pairs",
                "n_drop",
                "drop_rate",
                "n_components",
                "n_duplicate_components",
                "max_component_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "sample": str(args.sample),
        "pairs": str(args.pairs),
        "out": str(args.out),
        "n_docs": int(n_docs),
        "thresholds": [float(x) for x in args.thresholds],
        "time_sec": round(time.time() - start, 4),
        "results": rows,
    }

    with args.out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("S05 threshold sweep finished.")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

