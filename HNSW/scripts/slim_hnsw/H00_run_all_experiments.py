import subprocess
import time
import json
import psutil
from pathlib import Path
from datetime import datetime

# 設定路徑
SCRIPTS_DIR = Path(__file__).resolve().parent
REPORT_DIR = Path("/home/u08/workspace/HNSW/reports/slim_hnsw/log")
LOG_DIR = Path("/home/u08/workspace/HNSW/log/slim_hnsw/log")

SCRIPTS = ["H01_sample_slimpajama.py", "H02_embed.py", "H03_hnsw.py"]

def get_process_resources(parent_proc):
    """遞迴獲取父程序及其所有子程序的 CPU 與記憶體總和"""
    total_cpu = 0
    total_mem = 0
    try:
        procs = [parent_proc] + parent_proc.children(recursive=True)
        for p in procs:
            try:
                # interval=None 配合外部定時觸發，不會阻塞執行緒
                total_cpu += p.cpu_percent(interval=None)
                total_mem += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total_cpu, total_mem / (1024 * 1024)  # 轉為 MB

def run_step(script, model, log_file, report_file):
    script_path = SCRIPTS_DIR / script
    cmd = ["python3", str(script_path), "--model", model]

    print(f"\n{'='*20} RUNNING: {script} {'='*20}")
    
    stats = {"cpu_percent": [], "rss_mem_mb": []}
    
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # 紀錄初始 IO
    def get_io_sum(pid):
        try:
            parent = psutil.Process(pid)
            read, write = 0, 0
            # 包含所有子程序
            for p in [parent] + parent.children(recursive=True):
                try:
                    io = p.io_counters()
                    read += io.read_bytes
                    write += io.write_bytes
                except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                    continue
            return read, write
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0, 0

    start_read, start_write = get_io_sum(process.pid)
    
    # 監控迴圈
    while process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            cpu, mem = get_process_resources(parent) # 使用您定義的那個函數
            stats["cpu_percent"].append(cpu)
            stats["rss_mem_mb"].append(mem)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(1.0)

    # 程式結束，計算總消耗
    end_read, end_write = get_io_sum(process.pid)
    duration = time.time() - start_time
    
    # 組合數據
    summary = {
        "script": script,
        "timestamp": datetime.now().isoformat(),
        "duration_sec": round(duration, 2),
        "avg_cpu_percent": round(sum(stats["cpu_percent"]) / len(stats["cpu_percent"]), 2) if stats["cpu_percent"] else 0,
        "peak_rss_mem_mb": round(max(stats["rss_mem_mb"]), 2) if stats["rss_mem_mb"] else 0,
        "total_io_read_mb": round((end_read - start_read) / (1024 * 1024), 2),
        "total_io_write_mb": round((end_write - start_write) / (1024 * 1024), 2),
        "return_code": process.returncode
    }
    
    report_file.write(json.dumps(summary) + "\n")
    report_file.flush()
    print(f"Finished {script}. Stats: {summary}")

def main():
    model = "all-MiniLM-L6-v2"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"pipeline_{timestamp}.log"
    report_path = REPORT_DIR / f"Hrun_all_resource_{timestamp}.jsonl"

    with log_path.open("w", encoding="utf-8") as log_file, \
         report_path.open("a", encoding="utf-8") as report_file:
        for script in SCRIPTS:
            run_step(script, model, log_file, report_file)

if __name__ == "__main__":
    main()