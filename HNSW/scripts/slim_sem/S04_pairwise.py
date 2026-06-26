import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_EMB = (
    ROOT
    / "data"
    / "embeddings"
    / "slim_hnsw"
    / "sentence-transformers_all-MiniLM-L6-v2_emb.npy"
)
DEFAULT_LABELS = ROOT / "data" / "clusters" / "slim_sem" / "cluster_labels.npy"
DEFAULT_OUT_PAIRS = ROOT / "data" / "semdedup" / "slim_sem" / "candidate_pairs.csv"
DEFAULT_REPORT = ROOT / "log" / "slim_sem" / "S04_pairwise.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--min-sim", type=float, default=0.75)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--out-pairs", type=Path, default=DEFAULT_OUT_PAIRS)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def write_pairs_for_cluster(writer, emb, indices, cluster_id, min_sim, block_size):
    n = len(indices)
    if n < 2:
        return 0

    count = 0

    for a0 in range(0, n, block_size):
        a1 = min(a0 + block_size, n)
        idx_a = indices[a0:a1]
        vec_a = normalize(emb[idx_a])

        for b0 in range(a0, n, block_size):
            b1 = min(b0 + block_size, n)
            idx_b = indices[b0:b1]
            vec_b = normalize(emb[idx_b])

            sims = vec_a @ vec_b.T

            if a0 == b0:
                rows, cols = np.triu_indices(a1 - a0, k=1)
                keep = sims[rows, cols] >= min_sim
                rows = rows[keep]
                cols = cols[keep]
            else:
                rows, cols = np.where(sims >= min_sim)

            for r, c in zip(rows, cols):
                doc_i = int(idx_a[r])
                doc_j = int(idx_b[c])
                sim = float(sims[r, c])
                writer.writerow([cluster_id, doc_i, doc_j, f"{sim:.8f}"])
                count += 1

    return count


def main():
    args = parse_args()
    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings: {args.emb}")
    if not args.emb.exists():
        raise FileNotFoundError(f"Embedding file not found: {args.emb}")

    print(f"Loading cluster labels: {args.labels}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Cluster labels not found: {args.labels}")

    emb = np.load(args.emb, mmap_mode="r")
    labels = np.load(args.labels)

    if len(emb) != len(labels):
        raise ValueError(f"Embedding rows ({len(emb)}) != labels ({len(labels)})")

    start = time.time()
    unique_labels = np.unique(labels)
    total_pairs = 0
    cluster_sizes = []

    print(f"N={len(labels)}, clusters={len(unique_labels)}, min_sim={args.min_sim}")
    print(f"Writing candidate pairs to: {args.out_pairs}")

    with args.out_pairs.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "doc_i", "doc_j", "cosine_sim"])

        for k, cluster_id in enumerate(unique_labels, start=1):
            indices = np.flatnonzero(labels == cluster_id)
            cluster_sizes.append(len(indices))

            pair_count = write_pairs_for_cluster(
                writer=writer,
                emb=emb,
                indices=indices,
                cluster_id=int(cluster_id),
                min_sim=args.min_sim,
                block_size=args.block_size,
            )
            total_pairs += pair_count

            if k % 50 == 0 or k == len(unique_labels):
                print(f"Processed {k}/{len(unique_labels)} clusters, pairs={total_pairs}")

    duration = time.time() - start
    report = {
        "embedding_path": str(args.emb),
        "labels_path": str(args.labels),
        "out_pairs": str(args.out_pairs),
        "n_docs": int(len(labels)),
        "n_clusters": int(len(unique_labels)),
        "min_sim": float(args.min_sim),
        "block_size": int(args.block_size),
        "candidate_pairs": int(total_pairs),
        "avg_cluster_size": float(np.mean(cluster_sizes)),
        "max_cluster_size": int(np.max(cluster_sizes)),
        "time_sec": round(duration, 4),
    }

    with args.out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("S04 pairwise candidate generation finished.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

