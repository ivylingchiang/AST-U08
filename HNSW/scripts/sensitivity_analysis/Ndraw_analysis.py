import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gmean

# ============================================================
# 1. 設定資料夾與讀取檔案
# ============================================================
REPORT_DIR = Path("reports/N03_hnsw_analysis")
MODEL_NAME = "sentence-transformers_all-MiniLM-L6-v2"

csv_files = sorted(glob.glob(str(REPORT_DIR / f"{MODEL_NAME}_*.csv")))

print("Found CSV files:")
for f in csv_files:
    print("  ", f)

if len(csv_files) == 0:
    raise FileNotFoundError(f"在 {REPORT_DIR} 找不到任何符合的 CSV 檔案！")

# 確保最多取 3 個 Run 進行分析，避免檔案過多導致圖表錯亂
available_runs = min(len(csv_files), 3)
dfs = []
for idx in range(available_runs):
    temp_df = pd.read_csv(csv_files[idx])
    temp_df["run_id"] = idx + 1  # 標記這是第幾次 Run
    dfs.append(temp_df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\n成功載入 {available_runs} 個 Run 的數據。總資料筆數: {len(df_all)}")

# ============================================================
# 2. 計算統計值（幾何平均、最小值、最大值）
# ============================================================
group_cols = ["varied_param", "M", "ef", "ef_construction"]
metric_cols = ["recall_at_10", "build_time_sec", "query_time_sec", "qps", "exact_time_sec"]

# 計算 GMean (最終報告核心趨勢)
df_gmean = df_all.groupby(group_cols)[metric_cols].agg(lambda x: gmean(x)).reset_index()

# 計算 Min 與 Max (用於計算變異變動範圍)
df_min = df_all.groupby(group_cols)[metric_cols].min().reset_index()
df_max = df_all.groupby(group_cols)[metric_cols].max().reset_index()

# ============================================================
# 3. 定義核心主繪圖函數 (支援 6 種模式)
# ============================================================
def generate_hnsw_chart(mode, output_filename, suptitle_text):
    """
    mode 參數支援:
    - 'run1', 'run2', 'run3': 繪製單次實驗結果
    - 'overlay': 將 3 次實驗曲線疊加對比 (驗證穩定度)
    - 'gmean': 繪製幾何平均最終趨勢
    - 'variation': 繪製核心趨勢並加上 Min-Max 變異陰影遮罩
    """
    # 移除 constrained_layout=True，改用手動控制頂部邊距來防止標題重疊
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    
    params = ["M", "ef", "ef_construction"]
    metrics_info = [
        {"col": "qps", "label": "QPS (queries/sec)", "color": "tab:red", "title": "QPS"},
        {"col": "build_time_sec", "label": "Build Time (sec)", "color": "tab:green", "title": "Build Time"},
        {"col": "query_time_sec", "label": "Search Time (sec)", "color": "tab:orange", "title": "Search Time"}
    ]

    # 設定 Overlay 模式下各個 Run 的線條與標記樣式
    run_styles = {
        1: {"linestyle": "-", "marker": "o", "alpha": 0.85, "label_suffix": "Run 1"},
        2: {"linestyle": "--", "marker": "^", "alpha": 0.75, "label_suffix": "Run 2"},
        3: {"linestyle": ":", "marker": "s", "alpha": 0.65, "label_suffix": "Run 3"}
    }

    for i, param in enumerate(params):
        for j, metric in enumerate(metrics_info):
            ax1 = axes[i, j]
            ax2 = ax1.twinx()
            
            ax1.grid(True, linestyle="--", alpha=0.4)
            
            # --- 根據不同模式過濾與繪製數據 ---
            if mode in ["run1", "run2", "run3"]:
                run_num = int(mode[-1])
                sub_df = df_all[(df_all["varied_param"] == param) & (df_all["run_id"] == run_num)].sort_values(by=param)
                x = sub_df[param]
                
                ax1.plot(x, sub_df["recall_at_10"], color="tab:blue", marker="o", linewidth=2.2, label="Recall @ 10")
                ax2.plot(x, sub_df[metric["col"]], color=metric["color"], marker="s", linestyle="--", linewidth=1.8, label=metric["title"])
                
            elif mode == "overlay":
                # 疊加模式：用迴圈將所有可用的 Run 一筆一筆畫在同個子圖上
                for r_id in range(1, available_runs + 1):
                    sub_df = df_all[(df_all["varied_param"] == param) & (df_all["run_id"] == r_id)].sort_values(by=param)
                    x = sub_df[param]
                    style = run_styles[r_id]
                    
                    ax1.plot(x, sub_df["recall_at_10"], color="tab:blue", linestyle=style["linestyle"], 
                             marker=style["marker"], alpha=style["alpha"], label=f"Recall ({style['label_suffix']})")
                    ax2.plot(x, sub_df[metric["col"]], color=metric["color"], linestyle=style["linestyle"], 
                             marker=style["marker"], alpha=style["alpha"], label=f"{metric['title']} ({style['label_suffix']})")
                    
            elif mode == "gmean":
                sub_df = df_gmean[df_gmean["varied_param"] == param].sort_values(by=param)
                x = sub_df[param]
                
                ax1.plot(x, sub_df["recall_at_10"], color="tab:blue", marker="o", linewidth=2.5, label="GMean Recall")
                ax2.plot(x, sub_df[metric["col"]], color=metric["color"], marker="s", linestyle="--", linewidth=2.0, label=f"GMean {metric['title']}")
                
            elif mode == "variation":
                sub_g = df_gmean[df_gmean["varied_param"] == param].sort_values(by=param)
                sub_mi = df_min[df_min["varied_param"] == param].sort_values(by=param)
                sub_ma = df_max[df_max["varied_param"] == param].sort_values(by=param)
                x = sub_g[param]
                
                # 畫中心趨勢線 (GMean)
                ax1.plot(x, sub_g["recall_at_10"], color="tab:blue", marker="o", linewidth=2.5, label="GMean Recall")
                ax2.plot(x, sub_g[metric["col"]], color=metric["color"], marker="s", linestyle="--", linewidth=2.0, label=f"GMean {metric['title']}")
                
                # 填補 Min-Max 變異陰影區間 (Variation Band)
                ax1.fill_between(x, sub_mi["recall_at_10"], sub_ma["recall_at_10"], color="tab:blue", alpha=0.15, label="Recall Range (Min-Max)")
                ax2.fill_between(x, sub_mi[metric["col"]], sub_ma[metric["col"]], color=metric["color"], alpha=0.15, label="Metric Range (Min-Max)")

            # --- 介面美化與標籤設定 ---
            ax1.set_xlabel(f"Parameter: {param}", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Recall @ 10", color="tab:blue", fontsize=11)
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            
            ax2.set_ylabel(metric["label"], color=metric["color"], fontsize=11)
            ax2.tick_params(axis="y", labelcolor=metric["color"])
            
            ax1.set_title(f"Impact of {param} on Recall & {metric['title']}", fontsize=12, fontweight="bold", pad=8)
            
            # 微縮圖例，避免擋住曲線
            ax1.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)

    # 🚀 【關鍵修正排版步奏】
    # 1. 先用 tight_layout 自動優化 3x3 內部子圖與雙 Y 軸標籤的緊湊度
    plt.tight_layout()
    # 2. 將整組子圖的頂部邊界強行拉下到 0.91 的位置，為上方的大標題騰出巨大的專屬無干擾空間
    fig.subplots_adjust(top=0.91)
    # 3. 將大標題精準固定在 y=0.96 的頂部中央，不論大標題有幾行，都絕對不會往下壓到子圖標題
    fig.suptitle(suptitle_text, fontsize=18, fontweight="bold", y=0.96)
    
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"成功儲存圖表: {output_filename}")
    plt.close(fig)

# ============================================================
# 4. 批量執行，一鍵產出 6 張分析圖
# ============================================================
print("\n開始產出 6 張專用分析圖表...")

# 圖 1：Run 1
generate_hnsw_chart(
    mode="run1", 
    output_filename="01_run1.png", 
    suptitle_text="HNSW Sensitivity Analysis - Run 1 (Cold/Initial State)"
)

# 圖 2：Run 2
if available_runs >= 2:
    generate_hnsw_chart(
        mode="run2", 
        output_filename="02_run2.png", 
        suptitle_text="HNSW Sensitivity Analysis - Run 2 (Warm State)"
    )
else:
    print("警告: 缺乏 Run 2 數據，跳過 02_run2.png")

# 圖 3：Run 3
if available_runs >= 3:
    generate_hnsw_chart(
        mode="run3", 
        output_filename="03_run3.png", 
        suptitle_text="HNSW Sensitivity Analysis - Run 3 (Warm State)"
    )
else:
    print("警告: 缺乏 Run 3 數據，跳過 03_run3.png")

# 圖 4：Overlay 比較
generate_hnsw_chart(
    mode="overlay", 
    output_filename="04_overlay_runs.png", 
    suptitle_text="HNSW Performance Stability: Multi-Run Overlay Comparison\n(Verifying System Consistency & Cache Effects)"
)

# 圖 5：GMean 核心報告
generate_hnsw_chart(
    mode="gmean", 
    output_filename="05_gmean_analysis.png", 
    suptitle_text="HNSW Multi-Param Sensitivity & Trade-off Analysis\n(Final Report - Cross-Run Geometric Mean)"
)

# 圖 6：Variation 變異分析
generate_hnsw_chart(
    mode="variation", 
    output_filename="06_variation_analysis.png", 
    suptitle_text="HNSW Performance Variance Analysis\n(Geometric Mean Line with Min-Max Shaded Fluctuation Band)"
)

print("\n全數圖表繪製完畢！請至工作目錄查看產出的 01~06 PNG 檔案。")