import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json


# HNSW/
# ├── data/
# │   ├── samples/
# │   │   └── slimpajama_100k.jsonl
# │   └── embeddings/
# │       └── slim_sem/
# │           ├── sentence-transformers_all-MiniLM-L6-v2_emb.npy
# │           └── sentence-transformers_all-MiniLM-L6-v2_meta.jsonl
# └── scripts/
#     └── slim_sem/
#         └── 02_embed.py

# 輸入：data/samples/slimpajama_100k.jsonl
# 輸出：sentence-transformers_all-MiniLM-L6-v2_emb.npy, sentence-transformers_all-MiniLM-L6-v2_meta.jsonl, log

# ==================================================
# Global Config
# ==================================================

ROOT = Path("/home/u08/workspace/HNSW")
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "samples" / "slimpajama_100k.jsonl"
EMBED_DIR = DATA_DIR / "embeddings" / "slim_hnsw"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")

BATCH_SIZE = 256
MAX_CHARS = 4000

# ==================================================

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    out_emb = EMBED_DIR / f"{SAFE_MODEL_NAME}_emb.npy"
    out_meta = EMBED_DIR / f"{SAFE_MODEL_NAME}_meta.jsonl"
    LOG_DIR = Path("/home/u08/workspace/HNSW/log/slim_sem/resource_data")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timelog = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"Sslim_embed_{timelog}.json"

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 自動偵測運算裝置 (GPU / CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================================================
    # 排除在計時外的開頭 LOG 區段
    # ==================================================
    print("=" * 60)
    print(f"Project Root : {ROOT}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Device       : {device.upper()}")  # 顯示當前使用的裝置
    print(f"Input File   : {INPUT_FILE}")
    print(f"Output Emb   : {out_emb}")
    print(f"Output Meta  : {out_meta}")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到輸入檔：{INPUT_FILE}\n"
            f"請先確認 sample 檔是否存在"
        )

    # ==================================================
    # 核心計算計時開始
    # ==================================================
    start_time = time.time()

    print("Loading model...")
    # 將偵測到的裝置傳入模型
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = []
    metas = []

    for row in tqdm(read_jsonl(INPUT_FILE), desc="Reading jsonl"):
        text = row["text"][:MAX_CHARS]
        texts.append(text)

        metas.append({
            "id": row["id"],
            "source": row.get("source", "unknown"),
            "text_len": row.get("text_len", len(row["text"])),
        })

    print("Encoding texts to embeddings...")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    print("Saving results...")
    np.save(out_emb, embs)

    with out_meta.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 如果使用 CUDA，強制同步確保所有 GPU 操作都已完成再結束計時
    if device == "cuda":
        torch.cuda.synchronize()

    total_execution_time = time.time() - start_time
    # ==================================================
    # 核心計算計時結束
    # ==================================================
    log_data = {
        "model": MODEL_NAME,
        "device": device,
        "input_file": str(INPUT_FILE),
        "embedding_shape": list(embs.shape),
        "batch_size": BATCH_SIZE,
        "max_chars": MAX_CHARS,
        "runtime_seconds": total_execution_time,
        "timestamp": timelog,
        "embedding_path": str(out_emb),
        "meta_path": str(out_meta),
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"Log saved to: {log_file}")

    # ==================================================
    # 排除在計時外的結尾 LOG 區段
    # ==================================================
    print("\n" + "=" * 60)
    print("PIPELINE FINISHED")
    print("-" * 60)
    print(f"Model             : {MODEL_NAME}")
    print(f"Device Used       : {device.upper()}")
    print(f"Embeddings Shape  : {embs.shape}")
    print(f"Core Elapsed Time : {total_execution_time:.4f} seconds")
    print(f"Saved Embedding   : {out_emb}")
    print(f"Saved Metadata    : {out_meta}")
    print("=" * 60)

if __name__ == "__main__":
    main()