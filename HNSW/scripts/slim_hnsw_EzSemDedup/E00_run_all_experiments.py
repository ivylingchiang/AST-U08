from pathlib import Path
import subprocess
from datetime import datetime
import os
import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime

# 設定路徑
SCRIPTS_DIR = Path(__file__).resolve().parent
REPORT_DIR = Path("/home/u08/workspace/HNSW/reports/slim_hnsw_EzSemDedup/log")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# SCRIPTS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "E01_sample_slimpajama.py",
    "E02_embed.py",
    "E03_hnsw_eval.py",
    "E04_hnsw_sweep.py",
    "E05_make_candidates.py",
    "E06_semdedup_hnsw.py",
    "E07_threshold_sweep.py",
    "E08_build_dedup_dataset.py"
]


def run_step(script, model, log_file, report_file):
    script_path = SCRIPTS_DIR / script
    cmd = ["python3", str(script_path), "--model", model]
    
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # 用於監測的 psutil 物件
    p = psutil.Process(process.pid)
    stats = {"cpu_percent": [], "rss_mem_mb": [], "io_read_bytes": 0, "io_write_bytes": 0}

    try:
        while process.poll() is None:
            # 採樣資源數據
            with p.oneshot():
                stats["cpu_percent"].append(p.cpu_percent())
                stats["rss_mem_mb"].append(p.memory_info().rss / (1024 * 1024))
                io = p.io_counters()
                stats["io_read_bytes"] = io.read_bytes
                stats["io_write_bytes"] = io.write_bytes
            time.sleep(1) # 每秒採樣一次
    except psutil.NoSuchProcess:
        pass

    duration = time.time() - start_time
    process.wait()

    # 整理摘要數據
    summary = {
        "script": script,
        "timestamp": datetime.now().isoformat(),
        "duration_sec": round(duration, 2),
        "avg_cpu_percent": round(sum(stats["cpu_percent"]) / len(stats["cpu_percent"]), 2) if stats["cpu_percent"] else 0,
        "peak_rss_mem_mb": round(max(stats["rss_mem_mb"]), 2) if stats["rss_mem_mb"] else 0,
        "total_io_read_mb": round(stats["io_read_bytes"] / (1024 * 1024), 2),
        "total_io_write_mb": round(stats["io_write_bytes"] / (1024 * 1024), 2),
        "return_code": process.returncode
    }

    # 寫入 JSONL
    report_file.write(json.dumps(summary) + "\n")
    report_file.flush()


def main():
    model = "all-MiniLM-L6-v2"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_path = REPORT_DIR.parent / f"Epipeline_{timestamp}.log"
    report_path = REPORT_DIR / f"Erun_all_resource_{timestamp}.jsonl"

    with log_path.open("w") as log_file, report_path.open("w") as report_file:
        for script in SCRIPTS:
            run_step(script, model, log_file, report_file)
            
    print(f"Pipeline finished. Reports saved to {report_path}")

if __name__ == "__main__":
    main()