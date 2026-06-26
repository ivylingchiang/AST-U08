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
DEFAULT_KEEP = ROOT / "data" / "semdedup" / "slim_sem" / "keep_ids.txt"
DEFAULT_DROP = ROOT / "data" / "semdedup" / "slim_sem" / "drop_ids.txt"
DEFAULT_DROP_PAIRS = ROOT / "data" / "semdedup" / "slim_sem" / "drop_pairs.csv"
DEFAULT_DEDUP_JSONL = ROOT / "data" / "semdedup" / "slim_sem" / "slimpajama_100k_dedup.jsonl"
DEFAULT_REPORT = ROOT / "data" / "reports" / "slim_sem_final_report.json"


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
    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--keep-policy", choices=["longest", "shortest", "first"], default="longest")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--out-keep", type=Path, default=DEFAULT_KEEP)
    parser.add_argument("--out-drop", type=Path, default=DEFAULT_DROP)
    parser.add_argument("--out-drop-pairs", type=Path, default=DEFAULT_DROP_PAIRS)
    parser.add_argument("--out-dedup-jsonl", type=Path, default=DEFAULT_DEDUP_JSONL)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_sample_info(path):
    text_lens = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            row = json.loads(line)
            text = row.get("text", "")
            text_lens.append(int(row.get("text_len", len(text))))
    return np.array(text_lens, dtype=np.int64)


def build_components(pairs_path, threshold, n_docs, chunksize):
    uf = UnionFind(n_docs)
    n_pairs_after_threshold = 0

    for chunk in pd.read_csv(pairs_path, chunksize=chunksize):
        chunk = chunk[chunk["cosine_sim"] >= threshold]
        n_pairs_after_threshold += len(chunk)
        for row in chunk.itertuples(index=False):
            uf.union(row.doc_i, row.doc_j)

    components = {}
    for doc_id in range(n_docs):
        root = uf.find(doc_id)
        components.setdefault(root, []).append(doc_id)

    return components, n_pairs_after_threshold


def choose_keep_id(component, text_lens, keep_policy):
    if keep_policy == "first":
        return min(component)
    if keep_policy == "shortest":
        return min(component, key=lambda x: (text_lens[x], x))
    return max(component, key=lambda x: (text_lens[x], -x))


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc_id in ids:
            f.write(f"{int(doc_id)}\n")


def write_dedup_jsonl(sample_path, out_path, drop_set):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_keep = 0
    with sample_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for doc_id, line in enumerate(src):
            if doc_id not in drop_set:
                dst.write(line)
                n_keep += 1
    return n_keep


def write_drop_pairs(pairs_path, out_path, threshold, drop_set, chunksize):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "doc_i", "doc_j", "cosine_sim"])

        for chunk in pd.read_csv(pairs_path, chunksize=chunksize):
            chunk = chunk[chunk["cosine_sim"] >= threshold]
            for row in chunk.itertuples(index=False):
                if int(row.doc_i) in drop_set or int(row.doc_j) in drop_set:
                    writer.writerow([row.cluster_id, row.doc_i, row.doc_j, f"{float(row.cosine_sim):.8f}"])
                    n_written += 1

    return n_written


def main():
    args = parse_args()

    for path in [args.out_keep, args.out_drop, args.out_drop_pairs, args.out_dedup_jsonl, args.out_report]:
        path.parent.mkdir(parents=True, exist_ok=True)

    if not args.sample.exists():
        raise FileNotFoundError(f"Sample file not found: {args.sample}")
    if not args.pairs.exists():
        raise FileNotFoundError(f"Candidate pairs not found: {args.pairs}")

    start = time.time()
    text_lens = load_sample_info(args.sample)
    n_docs = len(text_lens)

    print(f"Loaded sample docs: {n_docs}")
    print(f"Building duplicate components at threshold={args.threshold}")

    components, n_pairs_after_threshold = build_components(
        pairs_path=args.pairs,
        threshold=args.threshold,
        n_docs=n_docs,
        chunksize=args.chunksize,
    )

    keep_ids = []
    drop_ids = []
    n_duplicate_components = 0
    max_component_size = 1

    for component in components.values():
        if len(component) == 1:
            keep_ids.append(component[0])
            continue

        n_duplicate_components += 1
        max_component_size = max(max_component_size, len(component))
        keep_id = choose_keep_id(component, text_lens, args.keep_policy)
        keep_ids.append(keep_id)
        drop_ids.extend(doc_id for doc_id in component if doc_id != keep_id)

    keep_ids = sorted(keep_ids)
    drop_ids = sorted(drop_ids)
    drop_set = set(drop_ids)

    write_ids(args.out_keep, keep_ids)
    write_ids(args.out_drop, drop_ids)
    n_keep_jsonl = write_dedup_jsonl(args.sample, args.out_dedup_jsonl, drop_set)
    n_drop_pairs = write_drop_pairs(args.pairs, args.out_drop_pairs, args.threshold, drop_set, args.chunksize)

    report = {
        "method": "kmeans_cluster_pairwise_semdedup",
        "sample": str(args.sample),
        "pairs": str(args.pairs),
        "sim_threshold": float(args.threshold),
        "keep_policy": args.keep_policy,
        "n_total": int(n_docs),
        "n_pairs_after_threshold": int(n_pairs_after_threshold),
        "n_duplicate_components": int(n_duplicate_components),
        "max_component_size": int(max_component_size),
        "n_keep": int(len(keep_ids)),
        "n_drop": int(len(drop_ids)),
        "drop_rate": float(len(drop_ids) / n_docs) if n_docs > 0 else 0.0,
        "n_dedup_jsonl_rows": int(n_keep_jsonl),
        "n_drop_pairs_written": int(n_drop_pairs),
        "out_keep": str(args.out_keep),
        "out_drop": str(args.out_drop),
        "out_drop_pairs": str(args.out_drop_pairs),
        "out_dedup_jsonl": str(args.out_dedup_jsonl),
        "time_sec": round(time.time() - start, 4),
    }

    with args.out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("S06 final SemDeDup finished.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

