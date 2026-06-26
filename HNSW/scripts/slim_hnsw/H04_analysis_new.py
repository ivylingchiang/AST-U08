#!/usr/bin/env python3
import os
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

def parse_txt_hnsw(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    m_ts = re.search(r'Timestamp:\s*([\d_]+)', content)
    m_recall = re.search(r'Recall@10=([\d\.]+)', content)
    m_exact = re.search(r'Exact Time=([\d\.]+)', content)
    m_build = re.search(r'Build Time=([\d\.]+)', content)
    m_query = re.search(r'Query Time=([\d\.]+)', content)
    m_qps = re.search(r'QPS=([\d\.]+)', content)
    
    if m_ts: data['timestamp'] = m_ts.group(1)
    if m_recall: data['Recall@10'] = float(m_recall.group(1))
    if m_exact: data['Exact_Time_sec'] = float(m_exact.group(1))
    if m_build: data['Build_Time_sec'] = float(m_build.group(1))
    if m_query: data['Query_Time_sec'] = float(m_query.group(1))
    if m_qps: data['QPS'] = float(m_qps.group(1))
    return data

def parse_json_embed(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'timestamp': data.get('timestamp'),
        'Embedding_Runtime_sec': data.get('runtime_seconds')
    }

def parse_txt_resource(filepath):
    data = {}
    filename = os.path.basename(filepath)
    m_ts = re.search(r'\d{8}_\d{6}', filename)
    if m_ts: data['timestamp'] = m_ts.group(0)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Metric:' in line and 'Value:' in line:
                parts = line.split('|')
                metric = parts[1].split('Metric:')[1].strip()
                value = float(parts[2].split('Value:')[1].strip())
                data[metric] = value
    return data

def collect_data(base_dir):
    res_dir = os.path.join(base_dir, "log/slim_hnsw/resource_data")
    hnsw_files = sorted(glob.glob(os.path.join(res_dir, "Hhnsw_*.txt")))
    embed_files = sorted(glob.glob(os.path.join(res_dir, "Hslim_embed_*.json")))
    read_files = sorted(glob.glob(os.path.join(res_dir, "Hslim_read_*.txt")))
    
    n_runs = min(len(hnsw_files), len(embed_files), len(read_files))
    if n_runs == 0:
        print(f"警告：在 {res_dir} 找不到匹配的 log 檔案！")
        return []
        
    runs_data = []
    for i in range(n_runs):
        d_hnsw = parse_txt_hnsw(hnsw_files[i])
        d_embed = parse_json_embed(embed_files[i])
        d_read = parse_txt_resource(read_files[i])
        
        merged = {}
        merged.update(d_read)
        merged.update(d_embed)
        merged.update(d_hnsw)
        runs_data.append(merged)
    return runs_data

def generate_plots(runs_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not runs_data: return
    
    # 定義我們關心的核心指標
    metrics = [
        'Dataset_Sampling_Time_sec',
        'Embedding_Runtime_sec',
        'Build_Time_sec',
        'Query_Time_sec',
        'QPS',
        'Recall@10'
    ]
    metrics = [m for m in metrics if m in runs_data[0]]
    
    x_indices = np.arange(len(runs_data))
    x_labels = [f"Run {i+1}" for i in range(len(runs_data))]
    
    # -----------------------------------------------------------------
    # 圖 1：實驗波動與差異視覺化 (幾何平均數歸一化 + 水平常數波動帶)
    # -----------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    
    print("\n" + "="*80)
    print(f" 【系統自動產出】波動度分析報告 (基於 {len(runs_data)} 次實驗幾何平均數)")
    print("="*80)
    
    for metric in metrics:
        values = np.array([d[metric] for d in runs_data])
        # 【修正核心】：全面換成幾何平均數做歸一化分母
        mean_val = gmean(values)
        
        if mean_val != 0:
            norm_values = values / mean_val
            min_norm = np.min(norm_values)
            max_norm = np.max(norm_values)
            
            # 劃出比值折線
            line, = plt.plot(x_indices, norm_values, marker='o', linewidth=2.5, label=f"{metric}")
            # 實現「水平常數波動帶」效果：fill_between 會自動將極值標量橫向拉平鋪滿
            plt.fill_between(x_indices, min_norm, max_norm, color=line.get_color(), alpha=0.12)
            
            # 波動係數計算對齊幾何平均數
            cv = np.std(values) / mean_val
            print(f"指標 {metric:<30} | 幾何平均數: {mean_val:>10.4f} | 波動係數(CV): {cv:>7.4f} (越接近0越穩定)")

    plt.title("Experimental Fluctuation & Variance Analysis (Geometric Mean Baseline)\n(Lines show runs; Shaded areas represent Constant Horizontal Min-Max Bounds)", fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x_indices, x_labels)
    plt.ylabel("Normalized Ratio (Value / Geometric Mean)", fontsize=12)
    # 基準線切換成完美的 1.0 實線以彰顯 GM 軸心
    plt.axhline(y=1.0, color='#333333', linestyle='-', linewidth=1.5, alpha=0.8)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Metrics", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_fluctuation_trend.png"), dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # 計算幾何平均數 (Geometric Mean) 用於後續圖表與表格
    # -----------------------------------------------------------------
    gmeans = {metric: gmean([d[metric] for d in runs_data]) for metric in metrics}
    
    # -----------------------------------------------------------------
    # 圖 2：純幾何平均結果視覺化圖表
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    time_keys = ['Dataset_Sampling_Time_sec', 'Embedding_Runtime_sec', 'Build_Time_sec', 'Query_Time_sec']
    time_keys = [k for k in time_keys if k in gmeans]
    time_vals = [gmeans[k] for k in time_keys]
    
    clean_time_labels = [k.replace('_', ' ').replace(' sec', '') for k in time_keys]
    x_pos = np.arange(len(clean_time_labels))
    
    bars = ax1.bar(x_pos, time_vals, color='#4A90E2', width=0.4, label='Execution Time (sec)', alpha=0.85)
    ax1.set_ylabel('Time in Seconds (Log Scale)', color='#4A90E2', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(clean_time_labels, fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#4A90E2')
    
    ax1.set_ylim(min(time_vals) * 0.5, max(time_vals) * 15)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                 f'{height:.4f}s', ha='center', va='bottom', fontsize=9, color='#1F4E79', fontweight='bold')
                 
    if 'QPS' in gmeans:
        ax2 = ax1.twinx()
        qps_val = gmeans['QPS']
        
        ax2.axhline(y=qps_val, color='#E28413', linestyle=':', linewidth=2, alpha=0.6)
        ax2.plot(x_pos, [qps_val]*len(x_pos), marker='D', color='#E28413', markersize=8, linestyle='None', label='Throughput (QPS)')
        
        ax2.set_ylabel('Throughput (QPS)', color='#E28413', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#E28413')
        
        ax2.set_ylim(qps_val * 0.8, qps_val * 1.2)
        ax2.text(x_pos[-1], qps_val * 1.03, f'Geometric Mean QPS: {qps_val:.2f}', 
                 ha='right', va='bottom', color='#B25300', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF2E6', edgecolor='#E28413', alpha=0.8))

    plt.title("Overall Performance Summary (Geometric Mean of 4 Runs)", fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "geometric_mean_performance.png"), dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # 成果 3：將幾何平均後的結果輸出成精美的 Markdown 表格
    # -----------------------------------------------------------------
    table_path = os.path.join(output_dir, "geometric_mean_table.txt")
    with open(table_path, "w", encoding="utf-8") as tf:
        tf.write("| 評估指標 (Metrics) | 幾何平均數結果 (Geometric Mean) |\n")
        tf.write("| :--- | :---: |\n")
        for m, val in gmeans.items():
            if m in ['QPS']:
                tf.write(f"| {m:<30} | {val:>15.2f} |\n")
            elif m in ['Recall@10']:
                tf.write(f"| {m:<30} | {val:>15.6f} |\n")
            else:
                tf.write(f"| {m:<30} | {val:>15.4f} s |\n")
                
    print(f"\n[純幾何平均表格已生成]: {table_path}")
    print("==============================================================")
    with open(table_path, "r", encoding="utf-8") as tf:
        print(tf.read())
    print("==============================================================")
  
if __name__ == '__main__':
    base_workspace = "/home/u08/workspace/HNSW/reports/slim_hnsw"
    plots_output = os.path.join(base_workspace, "plots")
    
    print("正在收集與配對實驗日誌數據...")
    data = collect_data(base_workspace)
    if data:
        generate_plots(data, plots_output)
        print(f"所有圖表與數據表已成功輸出至: {plots_output}")