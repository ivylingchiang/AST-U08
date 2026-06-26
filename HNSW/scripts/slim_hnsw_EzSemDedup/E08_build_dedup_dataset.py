#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# ============================================================
# Config
# ============================================================

ROOT = Path("/home/u08/workspace/HNSW")

MODEL = "sentence-transformers_all-MiniLM-L6-v2"
FINAL_THRESHOLD = 0.88

INPUT_DATASET = (
    ROOT /
    "data/samples/slimpajama_100k.jsonl"
)

PAIR_PATH = (
    ROOT /
    "data/hnsw_candidate_pairs/slim_hnsw_EzSemDedup" /
    f"{MODEL}.parquet"
)

OUTPUT_DIR = ROOT / "data/EzSemDedup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUFFIX = str(FINAL_THRESHOLD).replace(".", "")

# ============================================================
# DSU
# ============================================================

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


# ============================================================
# Utils
# ============================================================

def count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Easy SemDedup Dataset Builder")
    print("=" * 60)

    print(f"Threshold : {FINAL_THRESHOLD}")

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    print("Counting dataset size...")

    n = count_rows(INPUT_DATASET)

    print(f"Total documents : {n}")

    print("Loading candidate pairs...")

    pairs = pd.read_parquet(PAIR_PATH)

    print(f"Original pairs : {len(pairs):,}")

    pairs = pairs[
        pairs["cosine_sim"] >= FINAL_THRESHOLD
    ].copy()

    print(f"Pairs >= {FINAL_THRESHOLD}: {len(pairs):,}")

    # --------------------------------------------------------
    # Build DSU
    # --------------------------------------------------------

    print("Building connected components...")

    dsu = DSU(n)

    for row in pairs.itertuples(index=False):

        dsu.union(
            int(row.id1),
            int(row.id2)
        )

    # --------------------------------------------------------
    # Extract Components
    # --------------------------------------------------------

    components = {}

    for i in range(n):

        root = dsu.find(i)

        if root not in components:
            components[root] = []

        components[root].append(i)

    print(f"Components : {len(components):,}")

    # --------------------------------------------------------
    # Generate Keep / Drop IDs
    # --------------------------------------------------------

    keep_ids = []
    drop_ids = []

    for comp in components.values():

        comp = sorted(comp)

        keep_ids.append(comp[0])

        if len(comp) > 1:
            drop_ids.extend(comp[1:])

    keep_ids = sorted(keep_ids)
    drop_ids = sorted(drop_ids)

    # --------------------------------------------------------
    # Save keep_ids
    # --------------------------------------------------------

    keep_path = OUTPUT_DIR / f"keep_ids_{SUFFIX}.txt"

    with open(keep_path, "w") as f:
        for idx in keep_ids:
            f.write(f"{idx}\n")

    print(f"Saved: {keep_path}")

    # --------------------------------------------------------
    # Save drop_ids
    # --------------------------------------------------------

    drop_path = OUTPUT_DIR / f"drop_ids_{SUFFIX}.txt"

    with open(drop_path, "w") as f:
        for idx in drop_ids:
            f.write(f"{idx}\n")

    print(f"Saved: {drop_path}")

    # --------------------------------------------------------
    # Save drop pairs
    # --------------------------------------------------------

    print("Generating drop_pairs.parquet ...")

    drop_set = set(drop_ids)

    drop_pairs = pairs[
        pairs["id1"].isin(drop_set)
        |
        pairs["id2"].isin(drop_set)
    ]

    drop_pairs_path = (
        OUTPUT_DIR /
        f"drop_pairs_{SUFFIX}.parquet"
    )

    drop_pairs.to_parquet(
        drop_pairs_path,
        index=False
    )

    print(f"Saved: {drop_pairs_path}")

    # --------------------------------------------------------
    # Build Deduplicated Dataset
    # --------------------------------------------------------

    print("Generating deduplicated dataset...")

    keep_set = set(keep_ids)

    dedup_path = (
        OUTPUT_DIR /
        f"slimpajama_100k_dedup_{SUFFIX}.jsonl"
    )

    keep_count = 0

    with open(
        INPUT_DATASET,
        "r",
        encoding="utf-8"
    ) as fin, open(
        dedup_path,
        "w",
        encoding="utf-8"
    ) as fout:

        for idx, line in enumerate(fin):

            if idx in keep_set:
                fout.write(line)
                keep_count += 1

    print(f"Saved: {dedup_path}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(f"Threshold        : {FINAL_THRESHOLD}")
    print(f"Original Docs    : {n:,}")
    print(f"Keep Docs        : {len(keep_ids):,}")
    print(f"Drop Docs        : {len(drop_ids):,}")
    print(f"Drop Rate        : {(len(drop_ids)/n)*100:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()