import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

# 1. 設定讀取與輸出路徑
LOG_DIR = '/home/u08/workspace/HNSW/log/slim_hnsw_EzSemDedup'
OUTPUT_DIR = '/home/u08/workspace/HNSW/reports/slim_hnsw_EzSemDedup/png'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 2. 資料讀取與解析 (考慮三次實驗 run1, run2, run3)
# -------------------------------------------------------------------------

# --- E05: JSON 格式 ---
e05_files = sorted(glob.glob(os.path.join(LOG_DIR, 'E05_resource_data*.json')))[:3]
e05_data = []
for f in e05_files:
    with open(f, 'r', encoding='utf-8') as file:
        e05_data.append(json.load(file))

e05_metrics = {
    'E05_Duration_Seconds': [d['duration_seconds'] for d in e05_data],
    'E05_Peak_Memory_MB': [d['peak_memory_mb'] for d in e05_data],
    'E05_Total_Pairs_Found': [d['total_pairs_found'] for d in e05_data]
}

# --- E06: JSONL 格式 (逐行讀取，並提取 computation_complete 階段) ---
e06_files = sorted(glob.glob(os.path.join(LOG_DIR, 'E06_resource_usage*.jsonl')))[:3]
e06_data = []
for f in e06_files:
    run_data = []
    with open(f, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                run_data.append(json.loads(line))
    e06_data.append(run_data)

e06_metrics = {
    'E06_Computation_Elapsed_Time': [],
    'E06_Computation_Memory_GB': []
}
for run in e06_data:
    comp_phase = [p for p in run if p.get('phase') == 'computation_complete']
    target = comp_phase[0] if comp_phase else run[-1]  # 找不到則防呆取最後一筆
    e06_metrics['E06_Computation_Elapsed_Time'].append(target['elapsed_time'])
    e06_metrics['E06_Computation_Memory_GB'].append(target['memory_usage_gb'])

# --- E07: JSONL 格式 (逐行讀取，提取最後一個 threshold 結尾作為代表) ---
e07_files = sorted(glob.glob(os.path.join(LOG_DIR, 'E07_resource_usage*.jsonl')))[:3]
e07_data = []
for f in e07_files:
    run_data = []
    with open(f, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                run_data.append(json.loads(line))
    e07_data.append(run_data)

e07_metrics = {
    'E07_Final_Elapsed': [],
    'E07_Peak_Memory_GB': []
}
for run in e07_data:
    e07_metrics['E07_Final_Elapsed'].append(run[-1]['elapsed'] if run else 0)
    e07_metrics['E07_Peak_Memory_GB'].append(max([p['memory_gb'] for p in run]) if run else 0)

# 合併所有指標至 DataFrame
all_metrics = {}
all_metrics.update(e05_metrics)
all_metrics.update(e06_metrics)
all_metrics.update(e07_metrics)

df = pd.DataFrame(all_metrics, index=['run1', 'run2', 'run3'])

# 計算各指標的幾何平均值 (Geometric Mean)
gmean_values = {}
for col in df.columns:
    gmean_values[col] = gmean(df[col])

print(">>> 數據解析完成。各實驗數值與其幾何平均值如下：")
for col in df.columns:
    print(f"指標 [{col}]:")
    print(f"  各次數值: {df[col].tolist()}")
    print(f"  幾何平均: {gmean_values[col]:.4f}")

# -------------------------------------------------------------------------
# 3. 第一種圖：長條圖 (個別數值 + 幾何平均水平線)
# -------------------------------------------------------------------------
for metric in df.columns:
    plt.figure(figsize=(7, 5))
    runs = ['run1', 'run2', 'run3']
    values = df[metric].tolist()
    geom_mean = gmean_values[metric]
    
    # 畫出 3 次實驗長條
    bars = plt.bar(runs, values, color=['#4F81BD', '#C0504D', '#9BBB59'], width=0.4, alpha=0.9)
    
    # 畫出幾何平均標記線
    plt.axhline(y=geom_mean, color='#FF0000', linestyle='--', linewidth=2, 
                label=f'Geometric Mean ({geom_mean:.2f})')
    
    # 數值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.title(f'{metric}\\n(Individual Runs vs Geometric Mean)', fontsize=12, fontweight='bold')
    plt.ylabel('Measured Value', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 儲存
    plt.savefig(os.path.join(OUTPUT_DIR, f'bar_{metric}.png'), dpi=300)
    plt.close()

# -------------------------------------------------------------------------
# 4. 第二種圖：折線圖 + Horizontal Bands (水平無波浪方正色帶)
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
runs = ['run1', 'run2', 'run3']
x_indexes = np.arange(len(runs))  # 軸線對應的數字：[0, 1, 2]

# 設置色盤以區分多條指標線
colors = plt.cm.get_cmap('tab10', len(df.columns))

for idx, metric in enumerate(df.columns):
    # 縱軸 = 實驗數值 / 幾何平均值
    ratio_values = df[metric].values / gmean_values[metric]
    
    # 1. 繪製趨勢折線與資料點
    plt.plot(runs, ratio_values, marker='o', markersize=6, linewidth=2, 
             color=colors(idx), label=metric)
    
    # 2. 嚴格提取純純純標量 (Scalar)
    min_val = float(np.min(ratio_values))
    max_val = float(np.max(ratio_values))
    
    # 3. 利用 plt.fill_between 渲染「水平、無波浪、邊界方正」的彩色透明色帶
    # x_indexes 涵蓋 0 到 2 (代表整個橫軸範圍)，y1 與 y2 帶入純純純標量
    plt.fill_between(x_indexes, y1=min_val, y2=max_val, color=colors(idx), alpha=0.07)

# 繪製 y=1.0 的基準線 (代表剛好等於幾何平均)
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=1.2, alpha=0.6)

plt.title('Normalized Metric Ratios to Geometric Mean\\n(with Horizontal Scaled Bands)', fontsize=13, fontweight='bold')
plt.xlabel('Experiment Runs', fontsize=11)
plt.ylabel('Normalized Ratio (Value / Geometric Mean)', fontsize=11)
plt.xticks(x_indexes, runs)
plt.grid(True, linestyle=':', alpha=0.5)

# 將圖例置於右側，避免遮擋
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=9)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, 'line_normalized_horizontal_bands.png'), dpi=300)
plt.close()

print(f"\\n>>> [成功] 所有分析圖表已輸出至：{OUTPUT_DIR}")