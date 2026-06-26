from pathlib import Path
import subprocess
from datetime import datetime
import json
import os
import time


ROOT = Path("/home/u08/workspace/HNSW")
SCRIPTS_DIR = ROOT / "scripts" / "slim_sem"

MODEL = "all-MiniLM-L6-v2"
FINAL_THRESHOLD = "0.88"
MIN_PAIRWISE_SIM = "0.75"

SAMPLE_JSONL = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"
EMB_PATH = ROOT / "data" / "embeddings" / "slim_hnsw" / "sentence-transformers_all-MiniLM-L6-v2_emb.npy"
LABELS_PATH = ROOT / "data" / "clusters" / "slim_sem" / "cluster_labels.npy"
PAIRS_PATH = ROOT / "data" / "semdedup" / "slim_sem" / "candidate_pairs.csv"

SWEEP_REPORT = ROOT / "data" / "reports" / "slim_sem_threshold_sweep.csv"
FINAL_REPORT = ROOT / "data" / "reports" / "slim_sem_final_report.json"
MANUAL_TOP = ROOT / "data" / "reports" / "slim_sem_manual_pairs_top_088.txt"
MANUAL_NEAR = ROOT / "data" / "reports" / "slim_sem_manual_pairs_near_088.txt"

KEEP_IDS = ROOT / "data" / "semdedup" / "slim_sem" / "keep_ids.txt"
DROP_IDS = ROOT / "data" / "semdedup" / "slim_sem" / "drop_ids.txt"
DROP_PAIRS = ROOT / "data" / "semdedup" / "slim_sem" / "drop_pairs.csv"
DEDUP_JSONL = ROOT / "data" / "semdedup" / "slim_sem" / "slimpajama_100k_dedup.jsonl"

SCRIPTS = [
    {
        "name": "S01_sample_slimpajama",
        "script": "S01_sample_slimpajama.py",
        "args": [],
    },
    {
        "name": "S02_embed",
        "script": "S02_embed.py",
        "args": ["--model", MODEL],
    },
    {
        "name": "S03_kmeans",
        "script": "S03_kmeans.py",
        "args": [],
    },
    {
        "name": "S04_pairwise",
        "script": "S04_pairwise.py",
        "args": [
            "--emb", str(EMB_PATH),
            "--labels", str(LABELS_PATH),
            "--min-sim", MIN_PAIRWISE_SIM,
            "--out-pairs", str(PAIRS_PATH),
        ],
    },
    {
        "name": "S05_threshold_sweep",
        "script": "S05_threshold_sweep.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(PAIRS_PATH),
            "--thresholds", "0.80", "0.82", "0.84", "0.86", "0.88", "0.90", "0.92",
            "--out", str(SWEEP_REPORT),
        ],
    },
    {
        "name": "S07_inspect_top_pairs",
        "script": "S07_inspect_pairs.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(PAIRS_PATH),
            "--threshold", FINAL_THRESHOLD,
            "--mode", "top",
            "--limit", "50",
            "--out", str(MANUAL_TOP),
        ],
    },
    {
        "name": "S07_inspect_near_threshold_pairs",
        "script": "S07_inspect_pairs.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(PAIRS_PATH),
            "--threshold", FINAL_THRESHOLD,
            "--mode", "near_threshold",
            "--limit", "50",
            "--out", str(MANUAL_NEAR),
        ],
    },
    {
        "name": "S06_final_semdedup",
        "script": "S06_final_semdedup.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(PAIRS_PATH),
            "--threshold", FINAL_THRESHOLD,
            "--keep-policy", "longest",
            "--out-keep", str(KEEP_IDS),
            "--out-drop", str(DROP_IDS),
            "--out-drop-pairs", str(DROP_PAIRS),
            "--out-dedup-jsonl", str(DEDUP_JSONL),
            "--out-report", str(FINAL_REPORT),
        ],
    },
]


def format_cmd(cmd):
    return " ".join(str(x) for x in cmd)


def run_step(step, log_file):
    script_path = SCRIPTS_DIR / step["script"]
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = ["python3", str(script_path), *step["args"]]

    msg = (
        "\n" + "=" * 60 + "\n"
        f"RUNNING STEP: {step['name']}\n"
        f"SCRIPT: {step['script']}\n"
        f"CMD: {format_cmd(cmd)}\n"
        + "=" * 60 + "\n"
    )
    print(msg)
    log_file.write(msg)
    log_file.flush()

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    start = time.time()
    process = subprocess.run(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    duration = time.time() - start

    done_msg = (
        "\n" + "-" * 60 + "\n"
        f"FINISHED STEP: {step['name']}\n"
        f"RETURN CODE: {process.returncode}\n"
        f"STEP TIME SEC: {duration:.4f}\n"
        + "-" * 60 + "\n"
    )
    print(done_msg)
    log_file.write(done_msg)
    log_file.flush()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    return {
        "step": step["name"],
        "script": step["script"],
        "time_sec": round(duration, 4),
        "return_code": process.returncode,
    }


def collect_summary(timestamp, total_time, step_records):
    final_dedup = None
    if FINAL_REPORT.exists():
        with FINAL_REPORT.open("r", encoding="utf-8") as f:
            final_dedup = json.load(f)

    summary = {
        "timestamp": timestamp,
        "total_pipeline_time_sec": round(total_time, 4),
        "model": MODEL,
        "method": "slim_sem_kmeans_pairwise_semdedup",
        "steps": step_records,
        "paths": {
            "sample_jsonl": str(SAMPLE_JSONL),
            "embedding": str(EMB_PATH),
            "cluster_labels": str(LABELS_PATH),
            "candidate_pairs": str(PAIRS_PATH),
            "threshold_sweep_report": str(SWEEP_REPORT),
            "final_report": str(FINAL_REPORT),
            "dedup_jsonl": str(DEDUP_JSONL),
        },
        "dedup": {
            "candidate_min_similarity": float(MIN_PAIRWISE_SIM),
            "final_threshold": float(FINAL_THRESHOLD),
            "keep_policy": "longest",
        },
        "final_dedup": final_dedup,
    }

    out_path = ROOT / "data" / "reports" / f"pipeline_summary_slim_sem_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return out_path


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "log" / "slim_sem"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{timestamp}.log"

    total_start = time.time()
    step_records = []

    with log_path.open("w", encoding="utf-8") as log_file:
        start_msg = (
            "\n" + "=" * 60 + "\n"
            f"PIPELINE START: {timestamp}\n"
            f"ROOT: {ROOT}\n"
            f"SCRIPTS_DIR: {SCRIPTS_DIR}\n"
            f"MODEL: {MODEL}\n"
            f"FINAL_THRESHOLD: {FINAL_THRESHOLD}\n"
            f"LOG_PATH: {log_path}\n"
            + "=" * 60 + "\n"
        )
        print(start_msg)
        log_file.write(start_msg)
        log_file.flush()

        try:
            for step in SCRIPTS:
                step_records.append(run_step(step, log_file))
        except Exception as e:
            fail_time = time.time() - total_start
            fail_msg = (
                "\n" + "!" * 60 + "\n"
                "PIPELINE FAILED\n"
                f"ERROR: {repr(e)}\n"
                f"TOTAL TIME BEFORE FAILURE SEC: {fail_time:.4f}\n"
                f"LOG SAVED TO: {log_path}\n"
                + "!" * 60 + "\n"
            )
            print(fail_msg)
            log_file.write(fail_msg)
            log_file.flush()
            raise

        total_time = time.time() - total_start
        summary_path = collect_summary(timestamp, total_time, step_records)

        end_msg = (
            "\n" + "=" * 60 + "\n"
            "PIPELINE FINISHED\n"
            f"TOTAL TIME SEC: {total_time:.4f}\n"
            f"LOG SAVED TO: {log_path}\n"
            f"SUMMARY SAVED TO: {summary_path}\n"
            + "=" * 60 + "\n"
        )
        print(end_msg)
        log_file.write(end_msg)
        log_file.flush()

    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()

