import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


ROOT = Path("/home/u08/workspace/HNSW")

INPUT_PATH = (
    ROOT
    / "data"
    / "embeddings"
    / "slim_hnsw"
    / "sentence-transformers_all-MiniLM-L6-v2_emb.npy"
)

OUTPUT_PATH = ROOT / "data" / "clusters" / "slim_sem" / "cluster_labels.npy"
LOG_PATH = ROOT / "log" / "slim_sem" / "S03_kmeans.json"

N_CLUSTERS = 1000


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在載入資料: {INPUT_PATH} ...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{INPUT_PATH}")

    embeddings = np.load(INPUT_PATH)

    n_docs, dim = embeddings.shape
    n_clusters = min(N_CLUSTERS, n_docs)

    print(f"資料載入完成。筆數: {n_docs}, 維度: {dim}")
    print(f"開始進行 KMeans 分群，群數: {n_clusters}")

    tracemalloc.start()
    start_time = time.time()

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=42,
    )
    kmeans.fit(embeddings)

    kmeans_time_sec = round(time.time() - start_time, 1)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    labels = kmeans.labels_
    _, cluster_counts = np.unique(labels, return_counts=True)

    log_data = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "n_docs": int(n_docs),
        "dim": int(dim),
        "n_clusters": int(n_clusters),
        "kmeans_time_sec": kmeans_time_sec,
        "peak_memory_gb": round(peak / (1024 ** 3), 3),
        "avg_cluster_size": float(np.mean(cluster_counts)),
        "max_cluster_size": int(np.max(cluster_counts)),
    }

    np.save(OUTPUT_PATH, labels)

    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    print("KMeans 分群完成！")
    print(f"Log 已寫入: {LOG_PATH}")
    print(f"分群結果已儲存: {OUTPUT_PATH}")
    print(json.dumps(log_data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
