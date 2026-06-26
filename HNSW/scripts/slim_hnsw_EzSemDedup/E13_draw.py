import pandas as pd
import glob
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# --- 設定 ---
LOG_DIR = "/home/u08/workspace/HNSW/reports/slim_hnsw_EzSemDedup/log/"
OUTPUT_DIR = Path(LOG_DIR)

# --- 1. 讀取與計算 ---
files = glob.glob(f"{LOG_DIR}Erun_all_resource*.jsonl")
all_data = [json.loads(line) for file in files for line in open(file, 'r')]
df = pd.DataFrame(all_data)

def geo_mean(series):
    return np.exp(np.mean(np.log(series + 1e-9)))

df_grouped = df.groupby('script').agg(
    duration_sec=('duration_sec', geo_mean),
    peak_rss_mem_mb=('peak_rss_mem_mb', geo_mean),
    avg_cpu_percent=('avg_cpu_percent', geo_mean),
    total_io_read_mb=('total_io_read_mb', geo_mean),
    total_io_write_mb=('total_io_write_mb', geo_mean)
).reset_index()

# 簡化名稱
df_grouped['script'] = df_grouped['script'].str.replace('.py', '')

# --- 2. 準備表格數據 (加入加總列) ---
table_data = df_grouped.copy()
# 計算各欄位的總和 (CPU 取平均，記憶體取最大值或加總視需求，這裡範例為總和)
totals = ['Total', 
          table_data['duration_sec'].sum(), 
          table_data['peak_rss_mem_mb'].sum(), 
          table_data['avg_cpu_percent'].mean(), 
          table_data['total_io_read_mb'].sum(), 
          table_data['total_io_write_mb'].sum()]

# 轉換為列表並加入 totals
plot_data = table_data.values.tolist()
plot_data.append(totals)

# --- 3. 繪製表格圖片 ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=[[round(float(x), 2) if isinstance(x, (int, float)) else x for x in row] for row in plot_data], 
                 colLabels=['Script', 'Time (s)', 'Mem (MB)', 'CPU (%)', 'Read (MB)', 'Write (MB)'], 
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
plt.savefig(OUTPUT_DIR / "summary_table.png", bbox_inches='tight', dpi=300)

# --- 4. 繪製修正偏移後的長條圖 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 使用 index 來定位，避免標籤偏移
x = np.arange(len(df_grouped['script']))

# Duration Plot
axes[0].bar(x, df_grouped['duration_sec'], color='teal', alpha=0.7, width=0.6)
axes[0].set_title('Execution Duration')
axes[0].set_ylabel('Seconds')
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_grouped['script'], rotation=45, ha='right')

# Memory Plot
axes[1].bar(x, df_grouped['peak_rss_mem_mb'], color='salmon', alpha=0.7, width=0.6)
axes[1].set_title('Peak Memory Usage')
axes[1].set_ylabel('Megabytes (MB)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(df_grouped['script'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "performance_charts.png", dpi=300)
print("[*] 報告已更新並儲存至 log 目錄")