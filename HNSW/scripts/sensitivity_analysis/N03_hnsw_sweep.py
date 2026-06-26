import time
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import hnswlib
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from datetime import datetime

# 專案結構：
# HNSW/
# ├── data/
# │   └── embeddings/
# │       └── slimpajama_100k_emb.npy
# ├── reports/
# │   └── hnsw_sweep.csv
# └── scripts/
#     └── 03_hnsw_sweep.py

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

N = 20_000
K = 10

# ==========================================
# 調整：定義預設基準值 (Baseline) 與更寬廣的測試範圍
# ==========================================
DEFAULTS = {
    "M": 16,
    "ef_construction": 200,
    "ef": 100
}

# 現在你可以放膽測試更多、更密集的組合，不用擔心組合爆炸！
GRID = {
    "M": [4, 8, 12, 16, 24, 32, 48, 64],
    "ef_construction": [50, 100, 150, 200, 250, 300, 400, 500],
    "ef": [10, 20, 50, 100, 150, 200, 250, 300, 400],
}


def exact_knn(x, k):
    sim = cosine_similarity(x, x)
    np.fill_diagonal(sim, -1.0)
    return np.argsort(-sim, axis=1)[:, :k]


def recall_at_k(exact, approx):
    return float(np.mean([
        len(set(e) & set(a)) / len(e)
        for e, a in zip(exact, approx)
    ]))


def run_one(x, exact, M, ef_construction, ef, seed):
    n, dim = x.shape

    index = hnswlib.Index(space="cosine", dim=dim)

    t0 = time.time()
    index.init_index(
        max_elements=n,
        M=M,
        ef_construction=ef_construction,
        random_seed=seed,
    )
    index.add_items(x, np.arange(n))
    build_time = time.time() - t0

    index.set_ef(ef)

    t0 = time.time()
    labels, distances = index.knn_query(x, k=K + 1)
    query_time = time.time() - t0

    approx = []
    for i, row in enumerate(labels):
        keep = row[row != i][:K]
        approx.append(keep)

    approx = np.vstack(approx)

    return {
        "N": n,
        "K": K,
        "M": M,
        "ef_construction": ef_construction,
        "ef": ef,
        "recall_at_10": recall_at_k(exact, approx),
        "build_time_sec": build_time,
        "query_time_sec": query_time,
        "qps": n / query_time,
    }


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    # args = parser.parse_args()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()
    
    safe_model_name = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EMB = DATA_DIR / "embeddings" / f"{safe_model_name}_emb.npy"
    
    # OUT = REPORT_DIR / f"N03_hnsw_analysis/{safe_model_name}.csv"
    # PLOT_OUT = REPORT_DIR / f"N03_hnsw_analysis/{safe_model_name}_sensitivity.png"
    output_dir = REPORT_DIR / "N03_hnsw_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    OUT = output_dir / (
        f"{safe_model_name}_seed{args.seed}_{timestamp}.csv"
    )

    PLOT_OUT = output_dir / (
        f"{safe_model_name}_seed{args.seed}_{timestamp}_sensitivity.png"
    )

    print(f"Loading embeddings: {EMB}")
    
    if not EMB.exists():
        raise FileNotFoundError(
            f"找不到 embedding 檔案：{EMB}\n"
            f"請先執行 scripts/01_embed.py，確認已產生：\n"
            f"{DATA_DIR / 'embeddings' / 'slimpajama_100k_emb.npy'}"
        )

    x_all = np.load(EMB).astype("float32")

    if len(x_all) < N:
        raise ValueError(
            f"embedding 數量不足：目前只有 {len(x_all)} 筆，但 N={N}"
        )

    x = x_all[:N]

    print(f"Loaded embeddings: {x.shape}")

    t0 = time.time()
    exact = exact_knn(x, K)
    exact_time = time.time() - t0

    print(f"Exact KNN time: {exact_time:.4f} sec")

    # ==========================================
    # 核心邏輯修改：產生 OAT (單一變數控制) 的實驗組合
    # ==========================================
    experiment_plan = []

    # 1. 固定 ef_construction 和 ef，只調整 M
    for val in GRID["M"]:
        experiment_plan.append({
            "M": val,
            "ef_construction": DEFAULTS["ef_construction"],
            "ef": DEFAULTS["ef"],
            "varied_param": "M"
        })

    # 2. 固定 M 和 ef，只調整 ef_construction
    for val in GRID["ef_construction"]:
        experiment_plan.append({
            "M": DEFAULTS["M"],
            "ef_construction": val,
            "ef": DEFAULTS["ef"],
            "varied_param": "ef_construction"
        })

    # 3. 固定 M 和 ef_construction，只調整 ef
    for val in GRID["ef"]:
        experiment_plan.append({
            "M": DEFAULTS["M"],
            "ef_construction": DEFAULTS["ef_construction"],
            "ef": val,
            "varied_param": "ef"
        })

    rows = []
    total_runs = len(experiment_plan)
    
    for idx, config in enumerate(experiment_plan, 1):
        M, efc, ef, vp = config["M"], config["ef_construction"], config["ef"], config["varied_param"]
        print(f"[{idx}/{total_runs}] Running experimental variation on [{vp}]: M={M}, ef_construction={efc}, ef={ef}")

        row = run_one(
            x=x,
            exact=exact,
            M=M,
            ef_construction=efc,
            ef=ef,
            seed=args.seed,
        )

        # row["exact_time_sec"] = exact_time
        # row["varied_param"] = vp  # 紀錄當前實驗是在控制哪個變數
        row["exact_time_sec"] = exact_time
        row["varied_param"] = vp
        row["seed"] = args.seed
        row["timestamp"] = timestamp
        rows.append(row)

    df = pd.DataFrame(rows)

    # 儲存 CSV 排序（依據控制變數與其數值排序，方便閱讀）
    df = df.sort_values(["varied_param", "M", "ef_construction", "ef"])
    df.to_csv(OUT, index=False)
    
    print()
    print(f"Saved report: {OUT}")

    # ==========================================
    # 繪圖區塊修改：繪製流暢的 OAT 趨勢折線圖
    # ==========================================
    params = ["M", "ef_construction", "ef"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"OAT Sensitivity Analysis: {args.model}\n(Varying One Parameter While Keeping Others At Default)", fontsize=16, fontweight='bold')

    for i, param in enumerate(params):
        # 篩選出只變動當前參數的實驗數據，並依該參數大小排序
        sub_df = df[df["varied_param"] == param].sort_values(by=param)
        
        # 1. 第一排：Recall 變化折線圖
        axes[0, i].plot(sub_df[param], sub_df["recall_at_10"], marker='o', color='#1f77b4', linewidth=2)
        # 標註基準點 (Baseline) 以供對照
        axes[0, i].axvline(x=DEFAULTS[param], color='red', linestyle='--', alpha=0.6, label=f'Default ({DEFAULTS[param]})')
        axes[0, i].set_title(f"Recall @ 10 vs {param}", fontsize=12)
        axes[0, i].set_xlabel(param)
        axes[0, i].set_ylabel("Recall @ 10")
        axes[0, i].grid(True, linestyle='--', alpha=0.5)
        axes[0, i].legend()
        
        # 2. 第二排：QPS 變化折線圖
        axes[1, i].plot(sub_df[param], sub_df["qps"], marker='s', color='#2ca02c', linewidth=2)
        axes[1, i].axvline(x=DEFAULTS[param], color='red', linestyle='--', alpha=0.6)
        axes[1, i].set_title(f"QPS vs {param}", fontsize=12)
        axes[1, i].set_xlabel(param)
        axes[1, i].set_ylabel("Queries Per Second (QPS)")
        axes[1, i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plt.savefig(PLOT_OUT, dpi=300)
    plt.close()
    print(f" [成功] 敏感度趨勢折線圖已儲存至: {PLOT_OUT}")
    print("="*40)


if __name__ == "__main__":
    main()