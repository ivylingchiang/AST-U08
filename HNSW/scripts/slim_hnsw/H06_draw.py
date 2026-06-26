import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def make_resource_table(csv_path, output_path):

    df = pd.read_csv(csv_path)

    # MB -> GB
    df["peak_rss_mem_gb"] = (
        df["peak_rss_mem_mb"] / 1024
    )

    show_df = df[
        [
            "Script",
            "duration_sec",
            "peak_rss_mem_gb",
        ]
    ].copy()

    show_df.columns = [
        "Script",
        "Runtime (s)",
        "Peak Memory (GB)",
    ]

    show_df["Runtime (s)"] = (
        show_df["Runtime (s)"]
        .map(lambda x: f"{x:.2f}")
    )

    show_df["Peak Memory (GB)"] = (
        show_df["Peak Memory (GB)"]
        .map(lambda x: f"{x:.2f}")
    )

    fig, ax = plt.subplots(figsize=(8, 2.5))

    ax.axis("off")

    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.45, 0.25, 0.30],  # 控制欄寬
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Header 加粗
    for col in range(len(show_df.columns)):
        table[(0, col)].set_text_props(weight="bold")

    # Total row 加粗
    total_row = len(show_df)

    for col in range(len(show_df.columns)):
        table[(total_row, col)].set_text_props(weight="bold")

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved table -> {output_path}")


if __name__ == "__main__":

    make_resource_table(
        "/home/u08/workspace/HNSW/reports/slim_hnsw/log/summary.csv",
        "/home/u08/workspace/HNSW/reports/slim_hnsw/plots/resource_table.png",
    )