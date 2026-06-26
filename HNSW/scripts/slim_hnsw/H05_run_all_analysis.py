import pandas as pd
import json
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import gmean

# 1. 設定路徑與參數
LOG_DIR = "/home/u08/workspace/HNSW/reports/slim_hnsw/log"
OUTPUT_TXT = os.path.join(LOG_DIR, "summary_report.txt")
OUTPUT_IMG_MAIN = os.path.join(LOG_DIR, "performance_summary.png")
OUTPUT_IMG_IO = os.path.join(LOG_DIR, "io_usage_summary.png")

def safe_gmean(x):
    x = x.replace(0, 1e-9)
    return gmean(x)

def generate_report():
    log_files = glob.glob(os.path.join(LOG_DIR, "Hrun_all_resource_*.jsonl"))
    if not log_files:
        print(f"錯誤：在 {LOG_DIR} 中找不到任何 log 檔案。")
        return

    all_data = []
    for file in log_files:
        with open(file, 'r') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    
    df = pd.DataFrame(all_data)
    cols = ["duration_sec", "avg_cpu_percent", "peak_rss_mem_mb", "total_io_read_mb", "total_io_write_mb"]

    grouped = df.groupby("script")[cols].agg(safe_gmean)
    totals = grouped.sum()
    totals.name = "Total"
    
    summary_df = pd.concat([grouped, totals.to_frame().T]).reset_index().rename(columns={"index": "Script"})
    summary_df = summary_df.round(2)

    # 輸出文字報告
    summary_df.to_csv(OUTPUT_TXT, sep="\t", index=False)
    
    # 2. 繪製圖表 1: 記憶體與耗時 (Main)
    plot_data = grouped.reset_index()
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(plot_data["script"], plot_data["peak_rss_mem_mb"], color='skyblue')
    ax1.set_title("Peak RSS Memory Usage (MB)")
    ax2.bar(plot_data["script"], plot_data["duration_sec"], color='salmon')
    ax2.set_title("Execution Duration (sec)")
    plt.tight_layout()
    fig1.savefig(OUTPUT_IMG_MAIN)

    # 3. 繪製圖表 2: IO 資源使用 (New)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    ax3.bar(plot_data["script"], plot_data["total_io_read_mb"], color='mediumseagreen')
    ax3.set_title("Total IO Read (MB)")
    ax4.bar(plot_data["script"], plot_data["total_io_write_mb"], color='gold')
    ax4.set_title("Total IO Write (MB)")
    plt.tight_layout()
    fig2.savefig(OUTPUT_IMG_IO)

    print(f"報告已完成：")
    print(f"- 文字檔: {OUTPUT_TXT}")
    print(f"- 主圖表: {OUTPUT_IMG_MAIN}")
    print(f"- IO 圖表: {OUTPUT_IMG_IO}")

if __name__ == "__main__":
    generate_report()