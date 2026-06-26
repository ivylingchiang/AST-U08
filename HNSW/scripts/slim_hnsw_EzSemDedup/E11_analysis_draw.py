#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Input / Output
# ============================================================

INPUT_JSON = Path(
    "/home/u08/workspace/HNSW/reports/slim_hnsw_EzSemDedup/"
    "sentence-transformers_all-MiniLM-L6-v2_threshold_sweep_20260613_135511.json"
)

OUTPUT_DIR = Path(
    "/home/u08/workspace/HNSW/reports/slim_hnsw_EzSemDedup/png"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# Load Data
# ============================================================

print(f"Loading: {INPUT_JSON}")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df = df.sort_values("sim_threshold")

print(df)

# ============================================================
# Figure 1
# Duplicate Components vs Threshold
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["sim_threshold"],
    df["n_duplicate_components"],
    marker="o",
    linewidth=2,
)

for x, y in zip(
    df["sim_threshold"],
    df["n_duplicate_components"]
):
    plt.annotate(
        str(y),
        (x, y),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
    )

plt.xlabel("Similarity Threshold")
plt.ylabel("Duplicate Components")

plt.title(
    "Easy SemDedup\nDuplicate Components vs Threshold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

out1 = (
    OUTPUT_DIR
    / "threshold_vs_duplicate_components.png"
)

plt.savefig(
    out1,
    dpi=300,
)

plt.close()

# ============================================================
# Figure 2
# Max Component Size vs Threshold
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["sim_threshold"],
    df["max_component_size"],
    marker="o",
    linewidth=2,
)

for x, y in zip(
    df["sim_threshold"],
    df["max_component_size"]
):
    plt.annotate(
        str(y),
        (x, y),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
    )

plt.xlabel("Similarity Threshold")
plt.ylabel("Max Component Size")

plt.title(
    "Easy SemDedup\nMax Component Size vs Threshold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

out2 = (
    OUTPUT_DIR
    / "threshold_vs_max_component_size.png"
)

plt.savefig(
    out2,
    dpi=300,
)

plt.close()

# ============================================================
# Figure 3
# Threshold vs Pairs / Drop / Drop Rate
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(10, 6)
)

# ----------------------------------
# Left axis
# ----------------------------------

line1 = ax1.plot(
    df["sim_threshold"],
    df["n_pairs_after_threshold"],
    marker="o",
    linewidth=2,
    label="Pairs",
)

line2 = ax1.plot(
    df["sim_threshold"],
    df["n_drop"],
    marker="s",
    linewidth=2,
    label="Drop",
)

ax1.set_xlabel(
    "Similarity Threshold"
)

ax1.set_ylabel(
    "Pairs / Drop Count"
)

ax1.grid(
    True,
    alpha=0.3,
)

# ----------------------------------
# Right axis
# ----------------------------------

ax2 = ax1.twinx()

line3 = ax2.plot(
    df["sim_threshold"],
    df["drop_rate"],
    marker="^",
    linestyle="--",
    linewidth=2,
    label="Drop Rate",
)

ax2.set_ylabel(
    "Drop Rate"
)

# ----------------------------------
# Combine legend
# ----------------------------------

lines = (
    line1
    + line2
    + line3
)

labels = [
    l.get_label()
    for l in lines
]

ax1.legend(
    lines,
    labels,
    loc="upper right",
)

plt.title(
    "Easy SemDedup\nThreshold vs Pairs / Drop / Drop Rate"
)

plt.tight_layout()

out3 = (
    OUTPUT_DIR
    / "threshold_vs_pairs_drop_droprate.png"
)

plt.savefig(
    out3,
    dpi=300,
)

plt.close()

# ============================================================
# Done
# ============================================================

print("\nSaved:")

print(out1)
print(out2)
print(out3)