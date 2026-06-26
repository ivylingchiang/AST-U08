import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# 輸入：
#   data/samples/slimpajama_100k.jsonl
#
# 輸出：
#   data/embeddings/slimpajama_100k_emb.npy
#   data/embeddings/slimpajama_100k_meta.jsonl
#   log/slim_hnsw/resource_data/slim_embed_YYYYMMDD_HHMMSS.txt


ROOT = Path("/home/u08/workspace/HNSW")

DEFAULT_IN = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
DEFAULT_OUT_EMB = ROOT / "data" / "embeddings" / "slimpajama_100k_emb.npy"
DEFAULT_OUT_META = ROOT / "data" / "embeddings" / "slimpajama_100k_meta.jsonl"

DEFAULT_OUT_EMB.parent.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = Path("/home/u08/workspace/HNSW/log/slim_hnsw_ImproveSemDedup/resource_data")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


def log_step(msg):
    print(f"[*] {msg}")


def save_resource_data(metrics):
    log_step("Saving resource metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = RESOURCE_DIR / f"Islim_embed_{timestamp}.txt"

    script_name = os.path.basename(__file__)

    with file_name.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Resource Metrics for: {script_name}\n")
        f.write("=" * 60 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"File: {script_name} | Metric: {metric_name:<35} | Value: {value}\n")

    print(f"\n[Info] 資源量測數據已儲存至：{file_name}")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def select_text_first(text, max_chars):
    return text[:max_chars]


def select_text_head_middle_tail(text, chunk_chars):
    if len(text) <= chunk_chars * 3:
        return text

    head = text[:chunk_chars]

    mid_start = max(0, len(text) // 2 - chunk_chars // 2)
    middle = text[mid_start:mid_start + chunk_chars]

    tail = text[-chunk_chars:]

    return head + "\n" + middle + "\n" + tail


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-emb", type=Path, default=DEFAULT_OUT_EMB)
    parser.add_argument("--out-meta", type=Path, default=DEFAULT_OUT_META)

    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument(
        "--text-mode",
        type=str,
        default="first",
        choices=["first", "head_middle_tail"],
    )
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--chunk-chars", type=int, default=2000)

    return parser.parse_args()


def main():
    start_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT START: {os.path.basename(__file__)}\n"
        + "=" * 60
    )
    print(start_msg)

    args = parse_args()
    metrics = {}

    core_start_time = time.time()

    metrics["Input_File"] = str(args.input)
    metrics["Output_Embedding_File"] = str(args.out_emb)
    metrics["Output_Meta_File"] = str(args.out_meta)
    metrics["Embedding_Model"] = args.model
    metrics["Text_Mode"] = args.text_mode
    metrics["Batch_Size"] = args.batch_size

    log_step("Loading sample jsonl...")
    rows = list(read_jsonl(args.input))

    texts = []
    metas = []

    for row in rows:
        text = row.get("text", "")

        if args.text_mode == "first":
            selected = select_text_first(text, args.max_chars)
        else:
            selected = select_text_head_middle_tail(text, args.chunk_chars)

        texts.append(selected)

        metas.append({
            "id": row.get("id"),
            "source": row.get("source", "unknown"),
            "text_len": row.get("text_len", len(text)),
            "selected_text_len": len(selected),
        })

    metrics["Num_Documents"] = len(texts)

    log_step("Loading embedding model...")
    model_start = time.time()
    model = SentenceTransformer(args.model)
    metrics["Model_Load_Time_sec"] = round(time.time() - model_start, 4)

    log_step("Encoding documents...")
    embed_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    metrics["Embedding_Time_sec"] = round(time.time() - embed_start, 4)

    args.out_emb.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)

    log_step("Saving embeddings...")
    save_start = time.time()
    np.save(args.out_emb, embeddings)

    with args.out_meta.open("w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    metrics["Save_Time_sec"] = round(time.time() - save_start, 4)
    metrics["Embedding_Shape"] = str(embeddings.shape)

    metrics["Total_Core_Execution_Time_sec"] = round(time.time() - core_start_time, 4)

    save_resource_data(metrics)

    end_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT FINISHED\n"
        + "=" * 60 + "\n"
    )
    print(end_msg)


if __name__ == "__main__":
    main()
