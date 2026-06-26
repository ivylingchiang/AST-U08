from pathlib import Path
import subprocess
from datetime import datetime
import time
import json

import pandas as pd


ROOT = Path("/home/u08/workspace/HNSW")
SCRIPTS_DIR = ROOT / "scripts" / "slim_hnsw_ImproveSemDedup"

LOG_DIR = ROOT / "log" / "slim_hnsw_ImproveSemDedup"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_DIR = ROOT / "data" / "reports"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_MODE = "first"
MAX_CHARS = "4000"

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_EF = 100

CANDIDATE_K = 100
CANDIDATE_MIN_SIM = 0.75

DEDUP_THRESHOLDS = [
    "0.80",
    "0.82",
    "0.84",
    "0.86",
    "0.88",
    "0.90",
    "0.92",
]

FINAL_THRESHOLD = "0.88"
KEEP_POLICY = "longest"


SAMPLE_JSONL = ROOT / "data" / "samples" / "slimpajama_100k.jsonl"

EMB_PATH = ROOT / "data" / "embeddings" / "slimpajama_100k_emb.npy"
META_PATH = ROOT / "data" / "embeddings" / "slimpajama_100k_meta.jsonl"

CANDIDATE_PAIRS = ROOT / "data" / "hnsw" / "candidate_pairs.parquet"

HNSW_EVAL_REPORT = ROOT / "data" / "reports" / "hnsw_eval.csv"
THRESHOLD_SWEEP_REPORT = ROOT / "data" / "reports" / "semdedup_backend_hnsw_sweep.csv"
FINAL_DEDUP_REPORT = ROOT / "data" / "reports" / "semdedup_backend_hnsw_report.json"

THRESHOLD_TAG = FINAL_THRESHOLD.replace(".", "")

MANUAL_TOP = ROOT / "data" / "reports" / f"manual_pairs_top_{THRESHOLD_TAG}.txt"
MANUAL_NEAR = ROOT / "data" / "reports" / f"manual_pairs_near_{THRESHOLD_TAG}.txt"

FINAL_KEEP_IDS = ROOT / "data" / "semdedup" / "keep_ids.txt"
FINAL_DROP_IDS = ROOT / "data" / "semdedup" / "drop_ids.txt"
FINAL_DROP_PAIRS = ROOT / "data" / "semdedup" / "drop_pairs.parquet"
FINAL_DEDUP_JSONL = ROOT / "data" / "semdedup" / "slimpajama_100k_dedup.jsonl"


PIPELINE = [
    {
        "name": "01_sample_slimpajama",
        "script": "I01_sample_slimpajama.py",
        "args": [],
    },
    {
        "name": "02_embed",
        "script": "I02_embed.py",
        "args": [
            "--model", MODEL,
            "--text-mode", TEXT_MODE,
            "--max-chars", MAX_CHARS,
            "--out-emb", str(EMB_PATH),
            "--out-meta", str(META_PATH),
        ],
    },
    {
        "name": "03_hnsw_eval",
        "script": "I03_hnsw_eval.py",
        "args": [
            "--emb", str(EMB_PATH),
            "--n", "20000",
            "--k", "10",
            "--fixed",
            "--m", str(HNSW_M),
            "--ef-construction", str(HNSW_EF_CONSTRUCTION),
            "--ef", str(HNSW_EF),
            "--out-report", str(HNSW_EVAL_REPORT),
        ],
    },
    {
        "name": "04_make_hnsw_candidates",
        "script": "I04_make_hnsw_candidates.py",
        "args": [
            "--emb", str(EMB_PATH),
            "--meta", str(META_PATH),
            "--k", str(CANDIDATE_K),
            "--sim-threshold", str(CANDIDATE_MIN_SIM),
            "--m", str(HNSW_M),
            "--ef-construction", str(HNSW_EF_CONSTRUCTION),
            "--ef", str(HNSW_EF),
            "--out-pairs", str(CANDIDATE_PAIRS),
        ],
    },
    {
        "name": "05_threshold_sweep",
        "script": "I05_threshold_sweep.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(CANDIDATE_PAIRS),
            "--thresholds", *DEDUP_THRESHOLDS,
            "--out", str(THRESHOLD_SWEEP_REPORT),
        ],
    },
    {
        "name": "07_inspect_top_pairs",
        "script": "I07_inspect_pairs.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(CANDIDATE_PAIRS),
            "--threshold", FINAL_THRESHOLD,
            "--mode", "top",
            "--limit", "50",
            "--out", str(MANUAL_TOP),
        ],
    },
    {
        "name": "07_inspect_near_threshold_pairs",
        "script": "I07_inspect_pairs.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(CANDIDATE_PAIRS),
            "--threshold", FINAL_THRESHOLD,
            "--mode", "near_threshold",
            "--limit", "50",
            "--out", str(MANUAL_NEAR),
        ],
    },
    {
        "name": "06_final_semdedup_backend_from_hnsw",
        "script": "I06_semdedup_backend_from_hnsw.py",
        "args": [
            "--sample", str(SAMPLE_JSONL),
            "--pairs", str(CANDIDATE_PAIRS),
            "--threshold", FINAL_THRESHOLD,
            "--keep-policy", KEEP_POLICY,
            "--out-keep", str(FINAL_KEEP_IDS),
            "--out-drop", str(FINAL_DROP_IDS),
            "--out-drop-pairs", str(FINAL_DROP_PAIRS),
            "--out-dedup-jsonl", str(FINAL_DEDUP_JSONL),
            "--out-report", str(FINAL_DEDUP_REPORT),
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
        "\n" + "=" * 80 + "\n"
        f"RUNNING STEP: {step['name']}\n"
        f"SCRIPT: {step['script']}\n"
        f"CMD: {format_cmd(cmd)}\n"
        + "=" * 80 + "\n"
    )

    print(msg)
    log_file.write(msg)
    log_file.flush()

    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )

    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

    process.wait()
    duration = time.time() - start_time

    done_msg = (
        "\n" + "-" * 80 + "\n"
        f"FINISHED STEP: {step['name']}\n"
        f"RETURN CODE: {process.returncode}\n"
        f"STEP TIME SEC: {duration:.4f}\n"
        + "-" * 80 + "\n"
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


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def collect_file_info(path):
    path = Path(path)

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }


def collect_final_summary(step_records, timestamp, total_time):
    summary = {
        "timestamp": timestamp,
        "total_pipeline_time_sec": round(total_time, 4),
        "model": MODEL,
        "embedding": {
            "text_mode": TEXT_MODE,
            "max_chars": MAX_CHARS,
            "embedding_path": str(EMB_PATH),
            "meta_path": str(META_PATH),
        },
        "hnsw": {
            "M": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "ef": HNSW_EF,
            "candidate_k": CANDIDATE_K,
            "candidate_min_similarity": CANDIDATE_MIN_SIM,
        },
        "dedup": {
            "thresholds": DEDUP_THRESHOLDS,
            "final_threshold": FINAL_THRESHOLD,
            "keep_policy": KEEP_POLICY,
        },
        "paths": {
            "root": str(ROOT),
            "scripts_dir": str(SCRIPTS_DIR),
            "sample_jsonl": str(SAMPLE_JSONL),
            "embedding": str(EMB_PATH),
            "meta": str(META_PATH),
            "candidate_pairs": str(CANDIDATE_PAIRS),
            "hnsw_eval_report": str(HNSW_EVAL_REPORT),
            "threshold_sweep_report": str(THRESHOLD_SWEEP_REPORT),
            "manual_top": str(MANUAL_TOP),
            "manual_near": str(MANUAL_NEAR),
            "final_keep_ids": str(FINAL_KEEP_IDS),
            "final_drop_ids": str(FINAL_DROP_IDS),
            "final_drop_pairs": str(FINAL_DROP_PAIRS),
            "final_dedup_jsonl": str(FINAL_DEDUP_JSONL),
            "final_dedup_report": str(FINAL_DEDUP_REPORT),
        },
        "files": {
            "sample_jsonl": collect_file_info(SAMPLE_JSONL),
            "embedding": collect_file_info(EMB_PATH),
            "meta": collect_file_info(META_PATH),
            "candidate_pairs": collect_file_info(CANDIDATE_PAIRS),
            "hnsw_eval_report": collect_file_info(HNSW_EVAL_REPORT),
            "threshold_sweep_report": collect_file_info(THRESHOLD_SWEEP_REPORT),
            "manual_top": collect_file_info(MANUAL_TOP),
            "manual_near": collect_file_info(MANUAL_NEAR),
            "final_keep_ids": collect_file_info(FINAL_KEEP_IDS),
            "final_drop_ids": collect_file_info(FINAL_DROP_IDS),
            "final_drop_pairs": collect_file_info(FINAL_DROP_PAIRS),
            "final_dedup_jsonl": collect_file_info(FINAL_DEDUP_JSONL),
            "final_dedup_report": collect_file_info(FINAL_DEDUP_REPORT),
        },
        "steps": step_records,
    }

    if HNSW_EVAL_REPORT.exists():
        try:
            df = pd.read_csv(HNSW_EVAL_REPORT)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                summary["hnsw_eval"] = {
                    "N": safe_int(row.get("N")),
                    "K": safe_int(row.get("K")),
                    "M": safe_int(row.get("M")),
                    "ef_construction": safe_int(row.get("ef_construction")),
                    "ef": safe_int(row.get("ef")),
                    "recall_at_10": safe_float(row.get("recall_at_10")),
                    "build_time_sec": safe_float(row.get("build_time_sec")),
                    "query_time_sec": safe_float(row.get("query_time_sec")),
                    "qps": safe_float(row.get("qps")),
                    "exact_time_sec": safe_float(row.get("exact_time_sec")),
                }
        except Exception as e:
            summary["hnsw_eval_error"] = str(e)
    else:
        summary["hnsw_eval_missing"] = str(HNSW_EVAL_REPORT)

    if THRESHOLD_SWEEP_REPORT.exists():
        try:
            df = pd.read_csv(THRESHOLD_SWEEP_REPORT)
            summary["threshold_sweep"] = df.to_dict(orient="records")
        except Exception as e:
            summary["threshold_sweep_error"] = str(e)
    else:
        summary["threshold_sweep_missing"] = str(THRESHOLD_SWEEP_REPORT)

    if FINAL_DEDUP_REPORT.exists():
        try:
            with FINAL_DEDUP_REPORT.open("r", encoding="utf-8") as f:
                summary["final_dedup"] = json.load(f)
        except Exception as e:
            summary["final_dedup_error"] = str(e)
    else:
        summary["final_dedup_missing"] = str(FINAL_DEDUP_REPORT)

    out_json = SUMMARY_DIR / f"pipeline_summary_{timestamp}.json"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return out_json, summary


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"pipeline_{timestamp}.log"

    total_start = time.time()
    step_records = []

    with log_path.open("w", encoding="utf-8") as log_file:
        start_msg = (
            "\n" + "=" * 80 + "\n"
            f"PIPELINE START: {timestamp}\n"
            f"ROOT: {ROOT}\n"
            f"SCRIPTS_DIR: {SCRIPTS_DIR}\n"
            f"MODEL: {MODEL}\n"
            f"TEXT_MODE: {TEXT_MODE}\n"
            f"MAX_CHARS: {MAX_CHARS}\n"
            f"HNSW: M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, ef={HNSW_EF}\n"
            f"CANDIDATE_K: {CANDIDATE_K}\n"
            f"CANDIDATE_MIN_SIM: {CANDIDATE_MIN_SIM}\n"
            f"DEDUP_THRESHOLDS: {DEDUP_THRESHOLDS}\n"
            f"FINAL_THRESHOLD: {FINAL_THRESHOLD}\n"
            f"KEEP_POLICY: {KEEP_POLICY}\n"
            f"LOG_PATH: {log_path}\n"
            + "=" * 80 + "\n"
        )

        print(start_msg)
        log_file.write(start_msg)
        log_file.flush()

        try:
            for step in PIPELINE:
                record = run_step(step, log_file)
                step_records.append(record)

        except Exception as e:
            fail_time = time.time() - total_start

            fail_msg = (
                "\n" + "!" * 80 + "\n"
                "PIPELINE FAILED\n"
                f"ERROR: {repr(e)}\n"
                f"TOTAL TIME BEFORE FAILURE SEC: {fail_time:.4f}\n"
                f"LOG SAVED TO: {log_path}\n"
                + "!" * 80 + "\n"
            )

            print(fail_msg)
            log_file.write(fail_msg)
            log_file.flush()

            raise

        total_time = time.time() - total_start

        summary_json, summary = collect_final_summary(
            step_records=step_records,
            timestamp=timestamp,
            total_time=total_time,
        )

        end_msg = (
            "\n" + "=" * 80 + "\n"
            "PIPELINE FINISHED\n"
            f"TOTAL TIME SEC: {total_time:.4f}\n"
            f"LOG SAVED TO: {log_path}\n"
            f"SUMMARY SAVED TO: {summary_json}\n"
            + "=" * 80 + "\n"
        )

        print(end_msg)
        log_file.write(end_msg)
        log_file.flush()

    print("\n=== Pipeline Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved log to: {log_path}")
    print(f"Saved summary to: {summary_json}")


if __name__ == "__main__":
    main()

