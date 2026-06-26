from pathlib import Path
import subprocess
from datetime import datetime

SCRIPTS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    # "S01_sample_slimpajama.py",
    # "S02_embed.py",
    # "S03_SemDedup.py"
]


def run_step(script, model, log_file):
    script_path = SCRIPTS_DIR / script

    cmd = [
        "python3",
        str(script_path),
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
    model = "all-MiniLM-L6-v2"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path("/home/u08/workspace/HNSW/log/baseline/log")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"pipeline_{timestamp}.log"

    with log_path.open("w", encoding="utf-8") as log_file:

        for script in SCRIPTS:
            run_step(script, model, log_file)

    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()