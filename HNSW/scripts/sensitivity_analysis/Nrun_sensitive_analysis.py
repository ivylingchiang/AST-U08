import subprocess
import argparse
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = ROOT / "log"

LOG_DIR.mkdir(parents=True, exist_ok=True)


SCRIPTS = [
    # "01_embed.py",
    # "02_hnsw_eval.py",
    "N03_hnsw_sweep.py",
    # "03_select_variable_1.py",
    # "04_make_candidates.py",
    # "05_semdedup_hnsw.py",
    # "06_threshold_sweep.py",
]


def run_step(script, model, log_file):

    cmd = [
        "python3",
        f"scripts/{script}",
        "--model",
        model,
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

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:

        print(line, end="")

        log_file.write(line)
        log_file.flush()

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    args = parser.parse_args()

    model = args.model

    # safe folder/file name
    safe_model = model.replace("/", "_")

    # timestamp
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # --------------------------------------------------
    # create model-specific log directory
    # log/
    #   └── intfloat_e5-small-v2/
    #           pipeline_20260524_123000.log
    # --------------------------------------------------
    model_log_dir = LOG_DIR / safe_model

    model_log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # log file path
    log_path = (
        model_log_dir
        / f"pipeline_{timestamp}.log"
    )

    with log_path.open("w", encoding="utf-8") as log_file:

        start_msg = (
            "\n" + "#" * 80 + "\n"
            "FULL PIPELINE START\n"
            f"MODEL: {model}\n"
            f"LOG: {log_path}\n"
            + "#" * 80 + "\n"
        )

        print(start_msg)

        log_file.write(start_msg)
        log_file.flush()

        for script in SCRIPTS:

            run_step(
                script,
                model,
                log_file,
            )

        end_msg = (
            "\n" + "#" * 80 + "\n"
            "PIPELINE FINISHED\n"
            + "#" * 80 + "\n"
        )

        print(end_msg)

        log_file.write(end_msg)
        log_file.flush()

    print(f"\nSaved log to:\n{log_path}\n")


if __name__ == "__main__":
    main()