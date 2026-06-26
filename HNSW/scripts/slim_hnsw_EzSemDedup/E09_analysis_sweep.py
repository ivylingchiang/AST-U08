#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 定義路徑配置
LOG_DIR = "/home/u08/workspace/HNSW/log/slim_hnsw_EzSemDedup/resource_data"
OUTPUT_DIR = "/home/u08/workspace/HNSW/reports/slim_hnsw_EzSemDedup/png"

# 確保輸出圖片的資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def geo_mean(iterable):
    """計算幾何平均數 (Geometric Mean)"""
    a = np.array(iterable)
    if len(a) == 0:
        return 0.0
    return np.exp(np.mean(np.log(a)))

# ==================== Log 檔案解析器 ====================

def parse_sweep_log(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 'Dataset Size:' in line:
                data['Dataset_Size'] = float(line.split(':')[1].strip())
            elif 'M=' in line:
                data['M'] = float(line.split('=')[1].strip())
            elif 'ef_construction=' in line:
                data['ef_construction'] = float(line.split('=')[1].strip())
            elif 'ef=' in line:
                data['ef'] = float(line.split('=')[1].strip())
            elif 'Recall@10=' in line:
                data['Sweep_Recall'] = float(line.split('=')[1].strip())
            elif 'Exact Time=' in line:
                data['Sweep_Exact_Time'] = float(line.split('=')[1].strip())
            elif 'Build Time=' in line:
                data['Sweep_Build_Time'] = float(line.split('=')[1].strip())
            elif 'Query Time=' in line:
                data['Sweep_Query_Time'] = float(line.split('=')[1].strip())
            elif 'QPS=' in line:
                data['Sweep_QPS'] = float(line.split('=')[1].strip())
    return data

def parse_embed_log(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {'Embed_Runtime_sec': data.get('runtime_seconds')}

def parse_read_log(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line and 'Metric:' in line:
                parts = line.split('|')
                metric = parts[1].split(':')[1].strip()
                value = float(parts[2].split(':')[1].strip())
                data[metric] = value
    return data

def parse_metrics_log(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'Metrics_Recall': data.get('recall@k'),
        'Metrics_Exact_Time_sec': data.get('exact_time_sec'),
        'Metrics_HNSW_Build_Query_sec': data.get('hnsw_build_plus_query_sec')
    }

# ==================== 資料搜集與聚合 ====================

def collect_and_aggregate():
    sweep_files = sorted(glob.glob(os.path.join(LOG_DIR, "E_sweep_log*")))
    embed_files = sorted(glob.glob(os.path.join(LOG_DIR, "Eslim_embed*")))
    read_files = sorted(glob.glob(os.path.join(LOG_DIR, "Eslim_read*")))
    metrics_files = sorted(glob.glob(os.path.join(LOG_DIR, "metrics*")))
    
    print("【1】正在掃描並排序日誌檔案...")
    
    def aggregate_files(files, parse_func):
        raw_data = {}
        for f in files:
            parsed = parse_func(f)
            for k, v in parsed.items():
                if v is not None:
                    raw_data.setdefault(k, []).append(v)
        return raw_data

    raw_metrics = {}
    raw_metrics.update(aggregate_files(sweep_files, parse_sweep_log))
    raw_metrics.update(aggregate_files(embed_files, parse_embed_log))
    raw_metrics.update(aggregate_files(read_files, parse_read_log))
    raw_metrics.update(aggregate_files(metrics_files, parse_metrics_log))
    
    # 基礎圖表摘要全面換回幾何平均數
    aggregated_metrics = {}
    for k, v_list in raw_metrics.items():
        aggregated_metrics[k] = geo_mean(v_list)
        
    return aggregated_metrics, raw_metrics

# ==================== 繪圖生成器 ====================

def generate_plots(metrics, raw_metrics):
    print("【2】正在生成數據圖表...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # ---------------- 圖表 1：整體實驗執行時間階段分解 ----------------
    time_phases = {
        'Dataset Sampling\n(Read Stage)': metrics.get('Dataset_Sampling_Time_sec', 0),
        'Embedding Generation\n(Inference Stage)': metrics.get('Embed_Runtime_sec', 0),
        'HNSW Index\nBuild + Query': metrics.get('Metrics_HNSW_Build_Query_sec', 0)
    }
    plt.figure(figsize=(9, 6))
    bars = plt.bar(time_phases.keys(), time_phases.values(), color=['#4C72B0', '#55A868', '#C44E52'], width=0.4, edgecolor='grey', alpha=0.85)
    plt.title('End-to-End Experiment Time Breakdown\n(Geometric Mean)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Execution Time (Seconds)', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(time_phases.values()) * 0.01), f'{yval:.4f} s', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_pipeline_time_breakdown.png'), dpi=300)
    plt.close()

    # ---------------- 圖表 2：精確線性搜尋 vs HNSW 向量檢索時間對比 ----------------
    search_comparison = {
        'Exact Linear Search\n(Brute Force)': metrics.get('Metrics_Exact_Time_sec', 0),
        'HNSW Search\n(Build + Query Time)': metrics.get('Metrics_HNSW_Build_Query_sec', 0)
    }
    plt.figure(figsize=(7, 6))
    bars = plt.bar(search_comparison.keys(), search_comparison.values(), color=['#8172B3', '#CCB974'], width=0.35, edgecolor='grey', alpha=0.85)
    plt.title('Search Performance: Exact vs HNSW\n(Geometric Mean)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Time (Seconds)', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(search_comparison.values()) * 0.01), f'{yval:.4f} s', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_exact_vs_hnsw_time.png'), dpi=300)
    plt.close()

    # ---------------- 圖表 3：HNSW 內部階段分解 ----------------
    hnsw_internal = {
        'Index Build Time': metrics.get('Sweep_Build_Time', 0),
        'Query Execute Time': metrics.get('Sweep_Query_Time', 0)
    }
    plt.figure(figsize=(7, 6))
    bars = plt.bar(hnsw_internal.keys(), hnsw_internal.values(), color=['#64B5CD', '#E15759'], width=0.35, edgecolor='grey', alpha=0.85)
    qps_val = metrics.get('Sweep_QPS', 0)
    recall_val = metrics.get('Metrics_Recall', 0) if metrics.get('Metrics_Recall', 0) else metrics.get('Sweep_Recall', 0)
    plt.title(f'HNSW Internal Metrics\nRecall@10: {recall_val:.4%} | QPS: {qps_val:.2f}', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Time (Seconds)', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(hnsw_internal.values()) * 0.01), f'{yval:.4f} s', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_hnsw_internal_breakdown.png'), dpi=300)
    plt.close()

    # ---------------- 圖表 4：【幾何平均數 + 水平延伸常數波動帶】 ----------------
    target_metrics = [
        ('Dataset_Sampling_Time_sec', 'Data Sampling'),
        ('Embed_Runtime_sec', 'Embed Gen'),
        ('Metrics_Exact_Time_sec', 'Exact Search'),
        ('Metrics_HNSW_Build_Query_sec', 'HNSW Index (Build+Query)')
    ]
    
    plt.figure(figsize=(10, 6))
    # 基準線依然代表完美的幾何平均數 (1.0)
    plt.axhline(1.0, color='#333333', linestyle='-', linewidth=2, label='Geometric Mean Baseline (1.0)', zorder=1)
    
    print("\n" + "="*80)
    print(" 【系統自動產出】波動度分析報告 (基於幾何平均數歸一化)")
    print("="*80)
    
    all_norm_values = []
    
    for metric_key, label in target_metrics:
        values = np.array(raw_metrics.get(metric_key, []))
        if len(values) == 0:
            continue
            
        # 1. 計算幾何平均數作為分母
        mean_val = geo_mean(values)
        num_runs = len(values)
        
        if mean_val != 0:
            # 2. 進行幾何平均歸一化
            norm_values = values / mean_val
            all_norm_values.extend(norm_values)
            
            # 3. 抓取全局的最大與最小邊界值
            min_norm = np.min(norm_values)
            max_norm = np.max(norm_values)
            
            x_indices = [f"Run {i+1}" for i in range(num_runs)]
            
            # 4. 繪製比值折線
            line, = plt.plot(x_indices, norm_values, marker='o', linewidth=2.5, label=label, zorder=3)
            
            # 5. 實現「水平常數波動帶」效果：傳入單一數值的極值，fill_between 就會自動鋪滿成水平矩形帶
            plt.fill_between(x_indices, min_norm, max_norm, color=line.get_color(), alpha=0.12, zorder=2)
            
            # 6. 計算相對於幾何平均數的變異波動度
            cv = np.std(values) / mean_val
            print(f"指標 {label:<30} | {num_runs}次實驗幾何均值: {mean_val:>10.4f} 秒 | 波動係數(CV): {cv:>7.4f}")

    print("="*80 + "\n")

    plt.title('Normalized Performance Ratio & Constant Volatility Bands Across Runs\n(Shaded areas represent the horizontal absolute Min-Max boundaries relative to Geometric Mean)', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Normalized Ratio to Geometric Mean (Value / GM)', fontsize=12)
    plt.xlabel('Experimental Runs', fontsize=12)
    
    # 動態控制 Y 軸界限，確保不爆框
    if all_norm_values:
        max_dev = max(abs(np.min(all_norm_values) - 1.0), abs(np.max(all_norm_values) - 1.0))
        ylim_buffer = max(max_dev * 1.5, 0.05)
        plt.ylim(1.0 - ylim_buffer, 1.0 + ylim_buffer)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='gainsboro')
    
    plt.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, '04_experiment_variance_bounds.png')
    plt.savefig(path4, dpi=300)
    plt.close()
    print(f"  -> 已成功儲存【幾何平均數之水平波動帶圖】：{path4}")

# ==================== 主程式進入點 ====================

if __name__ == "__main__":
    print("==========================================")
    print("       Slim HNSW 實驗數據自動分析系統       ")
    print("==========================================")
    
    final_metrics, raw_metrics = collect_and_aggregate()
    generate_plots(final_metrics, raw_metrics)
    print("[系統訊息] 任務圓滿完成！請至 png 目錄查看全新的視覺效果。")