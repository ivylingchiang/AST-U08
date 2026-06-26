import os
import json
import time
import random
import subprocess
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm


# 輸入：gmongaras/SlimPajama-627B_Reupload
# 輸出：data/samples/slimpajama_100k.jsonl, log/slim_sem/resource_data/Sslim_read_YYYYMMDD_HHMMSS.txt

# 路徑設定
ROOT = Path(__file__).resolve().parent.parent

OUT = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

DATASET = "gmongaras/SlimPajama-627B_Reupload"
N = 100_000

# 資源量測數據的儲存路徑
RESOURCE_DIR = Path("/home/u08/workspace/HNSW/log/baseline/resource_data")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)


def log_step(msg):
    """
    統一管理各個小 function 的 log 輸出。
    如果不想顯示這些 log，可以直接將下方的 print 註解掉。
    """
    print(f"[*] {msg}")
    # pass


def drop_os_caches():
    """
    透過系統指令清除 OS Page Cache，確保每次實驗皆為真實 Disk I/O。
    工作站上有一個指令 drop-caches，他會執行這個指令：sync && echo 3 > /proc/sys/vm/drop_caches。
    """
    log_step("Dropping OS caches...")
    try:
        # 使用 shell=True 確保能夠吃到使用者的 alias 或環境變數中的自定義指令
        subprocess.run("drop-caches", shell=True, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"[Warning] 清除 cache 失敗: {e}")


def sample_dataset():
    log_step("Start sampling SlimPajama dataset...")
    ds = load_dataset(DATASET, split="train", streaming=True)
    
    with OUT.open("w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(ds, total=N)):
            if i >= N:
                break

            text = row.get("text", "")
            meta = row.get("meta") or {}
            source = meta.get("redpajama_set_name", "unknown")

            f.write(json.dumps({
                "id": i,
                "text": text,
                "source": source,
                "text_len": len(text)
            }, ensure_ascii=False) + "\n")


def measure_sequential_read(file_path):
    """量測連續讀取效能 (Sequential Read)"""
    log_step("Measuring sequential read performance...")
    start_time = time.time()
    total_bytes = 0
    
    with open(file_path, "rb") as f:
        # 每次讀取 1MB
        while chunk := f.read(1024 * 1024):
            total_bytes += len(chunk)
            
    duration = time.time() - start_time
    return total_bytes, duration


def measure_random_read(file_path, num_reads=5000, chunk_size=4096):
    """量測隨機讀取效能 (Random Read)"""
    log_step("Measuring random read performance...")
    file_size = os.path.getsize(file_path)
    start_time = time.time()
    total_bytes = 0
    
    if file_size > 0:
        with open(file_path, "rb") as f:
            for _ in range(num_reads):
                # 隨機挑選合法的 offset
                offset = random.randint(0, max(0, file_size - chunk_size))
                f.seek(offset)
                chunk = f.read(chunk_size)
                total_bytes += len(chunk)
                
    duration = time.time() - start_time
    return total_bytes, duration


def save_resource_data(metrics):
    """將量測數據標示清楚並存入指定路徑"""
    log_step("Saving resource metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = RESOURCE_DIR / f"Sslim_read_{timestamp}.txt"
    
    script_name = os.path.basename(__file__)
    
    with file_name.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Resource Metrics for: {script_name}\n")
        f.write("=" * 60 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"File: {script_name} | Metric: {metric_name:<30} | Value: {value}\n")
            
    print(f"\n[Info] 資源量測數據已儲存至：{file_name}")


def main():
    # ==========================================
    # 1. 檔案開始執行的 Log (不計入時間)
    # ==========================================
    start_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT START: {os.path.basename(__file__)}\n"
        + "=" * 60
    )
    print(start_msg)

    # ==========================================
    # 2. 核心計時區段開始
    # ==========================================
    metrics = {}
    core_start_time = time.time()

    # A. 採樣資料
    sample_start = time.time()
    sample_dataset()
    metrics["Dataset_Sampling_Time_sec"] = round(time.time() - sample_start, 4)
    
    # 確保寫入硬碟，避免後續讀取測量到尚未寫入的 buffer
    os.sync() 

    # B. 連續讀取測試
    drop_os_caches()
    seq_bytes, seq_time = measure_sequential_read(OUT)
    seq_mbps = (seq_bytes / (1024 * 1024)) / seq_time if seq_time > 0 else 0
    metrics["Sequential_Read_Time_sec"] = round(seq_time, 4)
    metrics["Sequential_Read_Speed_MBps"] = round(seq_mbps, 4)

    # C. 隨機讀取測試
    drop_os_caches()
    rand_bytes, rand_time = measure_random_read(OUT)
    rand_mbps = (rand_bytes / (1024 * 1024)) / rand_time if rand_time > 0 else 0
    metrics["Random_Read_Time_sec"] = round(rand_time, 4)
    metrics["Random_Read_Speed_MBps"] = round(rand_mbps, 4)

    # 記錄整體核心運行時間
    core_duration = time.time() - core_start_time
    metrics["Total_Core_Execution_Time_sec"] = round(core_duration, 4)

    # 寫入 Log 檔案
    save_resource_data(metrics)

    # ==========================================
    # 3. 檔案結束執行的 Log (不計入時間)
    # ==========================================
    end_msg = (
        "\n" + "=" * 60 + "\n"
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SCRIPT FINISHED\n"
        + "=" * 60 + "\n"
    )
    print(end_msg)


if __name__ == "__main__":
    main()