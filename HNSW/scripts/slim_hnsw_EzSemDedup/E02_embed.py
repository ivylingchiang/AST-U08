import json
import numpy as np
from pathlib import Path
import torch
import logging
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 專案結構：
# HNSW/
# ├── data/
# │   ├── samples/
# │   └── embeddings/
# |     └── slim_hnsw_EzSemDedup/
# |      ├── sentence-transformers_all-MiniLM-L6-v2_emb.npy
# |      └── sentence-transformers_all-MiniLM-L6-v2_meta.jsonl
# └── log/
#     └── slim_hnsw_EzSemDedup/
#         └── resource_data/
#             ├── E02_embed_20260611_183745.log
#             └── Eslim_embed_20260611_183745.json  <-- 摘要檔儲存於此
# └── scripts/
#     └── slim_hnsw_EzSemDedup/
#         └── E02_embed.py

# 定義路徑
ROOT = Path("/home/u08/workspace/HNSW")
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "log" / "slim_hnsw_EzSemDedup" / "resource_data"
OUT_DIR = DATA_DIR / "embeddings" / "slim_hnsw_EzSemDedup"

# 確保目錄存在
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 取得當下時間作為統一的 Timestamp
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 設定 Log (使用 E02_embed 命名)
log_filename = LOG_DIR / f"E02_embed_{run_timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

# 常數設定
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
MAX_CHARS = 4000
IN_FILE = DATA_DIR / "samples" / "slimpajama_100k.jsonl"

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    logging.info(f"Starting embedding process using model: {MODEL_NAME}")
    
    # GPU 偵測
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # 定義輸出檔名
    safe_model_name = MODEL_NAME.replace("/", "_")
    out_emb = OUT_DIR / f"{safe_model_name}_emb.npy"
    out_meta = OUT_DIR / f"{safe_model_name}_meta.jsonl"
    
    # 定義摘要檔案路徑 (儲存於 LOG_DIR，並命名為 Eslim_embed_{timestamp}.json)
    out_info = LOG_DIR / f"Eslim_embed_{run_timestamp}.json"

    if not IN_FILE.exists():
        logging.error(f"找不到輸入檔：{IN_FILE}")
        return

    # 載入模型
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts, metas = [], []
    logging.info("Reading and preparing data...")
    for row in tqdm(read_jsonl(IN_FILE), desc="Processing"):
        text = row["text"][:MAX_CHARS]
        texts.append(text)
        metas.append({
            "id": row["id"],
            "source": row.get("source", "unknown"),
            "text_len": row.get("text_len", len(row["text"])),
        })

    logging.info(f"Encoding {len(texts)} samples...")
    start_time = datetime.now()
    
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    
    end_time = datetime.now()
    duration = end_time - start_time
    runtime_seconds = duration.total_seconds()
    
    # 儲存結果 (Numpy 與 Meta)
    np.save(out_emb, embs)
    with out_meta.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 儲存執行摘要資訊 (JSON)
    info_dict = {
        "model": MODEL_NAME,
        "device": device,
        "input_file": str(IN_FILE),
        "embedding_shape": list(embs.shape),
        "batch_size": BATCH_SIZE,
        "max_chars": MAX_CHARS,
        "runtime_seconds": runtime_seconds,
        "timestamp": run_timestamp,
        "embedding_path": str(out_emb),
        "meta_path": str(out_meta)
    }
    
    with out_info.open("w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)

    logging.info(f"Processing completed in {duration}")
    logging.info(f"Embeddings saved to: {out_emb}")
    logging.info(f"Metadata saved to: {out_meta}")
    logging.info(f"Execution info saved to: {out_info}")

if __name__ == "__main__":
    main()