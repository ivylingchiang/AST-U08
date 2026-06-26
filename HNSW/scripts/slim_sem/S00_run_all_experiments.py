from pathlib import Path
import subprocess
from datetime import datetime
import os
import time


ROOT = Path("/home/u08/workspace/HNSW")
SCRIPTS_DIR = ROOT / "scripts" / "slim_sem"

MODEL = "all-MiniLM-L6-v2"

SCRIPTS = [
    {
        "script": "S01_sample_slimpajama.py",
        "args": [],
    },
    {
        "script": "S02_embed.py",
        "args": ["--model", MODEL],
    },
    {
        "script": "S03_kmeans.py",
        "args": [],
    },
]


def run_step(step, log_file):
    script = step["script"]
    script_path = SCRIPTS_DIR / script

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [
        "python3",
        str(script_path),
        *step["args"],
    ]

    msg = (
        "\n" + "=" * 60 + "\n"
        f"RUNNING: {script}\n"
        f"CMD: {' '.join(cmd)}\n"
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

    start_time = time.time()

    process = subprocess.run(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(ROOT),
    )

    duration = time.time() - start_time

    done_msg = (
        "\n" + "-" * 60 + "\n"
        f"FINISHED: {script}\n"
        f"RETURN CODE: {process.returncode}\n"
        f"STEP TIME SEC: {duration:.4f}\n"
        + "-" * 60 + "\n"
    )

    print(done_msg)
    log_file.write(done_msg)
    log_file.flush()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
        )

    return {
        "script": script,
        "time_sec": round(duration, 4),
        "return_code": process.returncode,
    }


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
            f"LOG_PATH: {log_path}\n"
            + "=" * 60 + "\n"
        )

        print(start_msg)
        log_file.write(start_msg)
        log_file.flush()

        try:
            for step in SCRIPTS:
                record = run_step(step, log_file)
                step_records.append(record)

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

        end_msg = (
            "\n" + "=" * 60 + "\n"
            "PIPELINE FINISHED\n"
            f"TOTAL TIME SEC: {total_time:.4f}\n"
            f"LOG SAVED TO: {log_path}\n"
            + "=" * 60 + "\n"
        )

        print(end_msg)
        log_file.write(end_msg)
        log_file.flush()

    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()
