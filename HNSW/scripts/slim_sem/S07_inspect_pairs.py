import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_SAMPLE = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
DEFAULT_PAIRS = ROOT / "data" / "semdedup" / "slim_sem" / "candidate_pairs.csv"
DEFAULT_OUT = ROOT / "data" / "reports" / "slim_sem_manual_pairs.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--mode", choices=["top", "near_threshold"], default="top")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_sample(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            row = json.loads(line)
            rows.append({
                "id": int(row.get("id", line_id)),
                "text": row.get("text", ""),
                "source": row.get("source", ""),
                "text_len": int(row.get("text_len", len(row.get("text", "")))),
            })
    return rows


def truncate(text, max_chars=700):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.sample.exists():
        raise FileNotFoundError(f"Sample file not found: {args.sample}")
    if not args.pairs.exists():
        raise FileNotFoundError(f"Candidate pairs not found: {args.pairs}")

    print("Loading sample rows...")
    rows = load_sample(args.sample)

    print("Loading pairs...")
    pairs = pd.read_csv(args.pairs)
    pairs = pairs[pairs["cosine_sim"] >= args.threshold].copy()

    if args.mode == "near_threshold":
        pairs["distance_to_threshold"] = (pairs["cosine_sim"] - args.threshold).abs()
        pairs = pairs.sort_values(["distance_to_threshold", "cosine_sim"], ascending=[True, False])
    else:
        pairs = pairs.sort_values("cosine_sim", ascending=False)

    pairs = pairs.head(args.limit)

    with args.out.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Manual Pair Inspection\n")
        f.write(f"mode={args.mode}, threshold={args.threshold}, limit={args.limit}\n")
        f.write("=" * 80 + "\n\n")

        for rank, row in enumerate(pairs.itertuples(index=False), start=1):
            doc_i = int(row.doc_i)
            doc_j = int(row.doc_j)
            rec_i = rows[doc_i]
            rec_j = rows[doc_j]

            f.write("-" * 80 + "\n")
            f.write(f"Rank: {rank}\n")
            f.write(f"cluster_id: {row.cluster_id}\n")
            f.write(f"doc_i: {doc_i}, doc_j: {doc_j}\n")
            f.write(f"cosine_sim: {float(row.cosine_sim):.8f}\n\n")

            f.write(f"[doc_i] source={rec_i['source']} text_len={rec_i['text_len']}\n")
            f.write(truncate(rec_i["text"]) + "\n\n")

            f.write(f"[doc_j] source={rec_j['source']} text_len={rec_j['text_len']}\n")
            f.write(truncate(rec_j["text"]) + "\n\n")

    print(f"Inspection file saved to: {args.out}")


if __name__ == "__main__":
    main()

