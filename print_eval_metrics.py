#!/usr/bin/env python3

import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path("output/rudra-bhili-translate/generations/mar2bhb")

# If you want to save plots instead of displaying them:
SAVE_PLOTS = True
PLOT_DIR = ROOT / "plots"


# ============================================================
# Load reports
# ============================================================

rows = []
category_rows = []

for report_path in sorted(ROOT.glob("ckp_*/report.json")):

    with open(report_path, "r") as f:
        report = json.load(f)

    run_name = report["run_name"]
    config = report["config"]

    # Extract checkpoint number
    match = re.search(r"ckp_(\d+)", run_name)
    checkpoint = int(match.group(1)) if match else None

    decoding = config.get("decoding", "unknown")

    for result in report["results"]:

        row = {
            "run": run_name,
            "checkpoint": checkpoint,
            "decoding": decoding,
            "chrf2pp": result.get("chrf2pp"),
            "glossary_alignment": result.get("glossary_alignment"),
            "translation_rate": result.get("translation_rate"),
            "generation_time_sec": result.get("generation_time_sec"),
            "n_samples": result.get("n_samples"),
        }

        rows.append(row)

        # Per-category metrics
        for category, metrics in result.get(
            "chrf2pp_by_category", {}
        ).items():

            category_rows.append({
                "run": run_name,
                "checkpoint": checkpoint,
                "decoding": decoding,
                "category": category,
                "chrf2pp": metrics["chrf2pp"],
                "n": metrics["n"],
            })


df = pd.DataFrame(rows)
category_df = pd.DataFrame(category_rows)

# Sort naturally by checkpoint and decoding
df = df.sort_values(["checkpoint", "decoding"])
category_df = category_df.sort_values(
    ["checkpoint", "decoding", "category"]
)


# ============================================================
# Print overall performance
# ============================================================

print("\n" + "=" * 100)
print("OVERALL PERFORMANCE")
print("=" * 100)

display_df = df.copy()

# Convert ratios to percentages for readability
display_df["glossary_alignment"] *= 100
display_df["translation_rate"] *= 100

display_df = display_df[
    [
        "checkpoint",
        "decoding",
        "chrf2pp",
        "glossary_alignment",
        "translation_rate",
        "generation_time_sec",
        "n_samples",
    ]
]

print(
    display_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# Print best checkpoints
# ============================================================

print("\n" + "=" * 100)
print("BEST CHECKPOINTS")
print("=" * 100)

for metric in [
    "chrf2pp",
    "glossary_alignment",
    "translation_rate",
]:

    idx = df[metric].idxmax()
    best = df.loc[idx]

    print(
        f"{metric:25s}: "
        f"{best['run']:20s} "
        f"({best[metric]:.4f})"
    )


# ============================================================
# Print category performance
# ============================================================

print("\n" + "=" * 100)
print("CATEGORY PERFORMANCE")
print("=" * 100)

category_pivot = category_df.pivot_table(
    index="category",
    columns=["checkpoint", "decoding"],
    values="chrf2pp",
)

print(
    category_pivot.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# Plot overall metrics
# ============================================================

PLOT_DIR.mkdir(exist_ok=True)

metrics = [
    ("chrf2pp", "chrF2++", False),
    ("glossary_alignment", "Glossary Alignment", True),
    ("translation_rate", "Translation Rate", True),
    ("generation_time_sec", "Generation Time (sec)", False),
]


for metric, ylabel, is_ratio in metrics:

    fig, ax = plt.subplots(figsize=(10, 6))

    for decoding, group in df.groupby("decoding"):

        group = group.sort_values("checkpoint")

        y = group[metric]

        if is_ratio:
            y = y * 100

        ax.plot(
            group["checkpoint"],
            y,
            marker="o",
            linewidth=2,
            label=decoding,
        )

        # Annotate values
        for x, value in zip(group["checkpoint"], y):
            ax.annotate(
                f"{value:.2f}",
                (x, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Checkpoint")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if SAVE_PLOTS:
        path = PLOT_DIR / f"{metric}_vs_checkpoint.png"
        plt.savefig(path, dpi=200)

    plt.show()


# ============================================================
# Category plot
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

for category, group in category_df.groupby("category"):

    # Plot greedy only here to keep the graph readable.
    group = group[group["decoding"] == "greedy"]

    if len(group) == 0:
        continue

    group = group.sort_values("checkpoint")

    ax.plot(
        group["checkpoint"],
        group["chrf2pp"],
        marker="o",
        linewidth=1.5,
        label=category,
    )

ax.set_xlabel("Checkpoint")
ax.set_ylabel("chrF2++")
ax.set_title("Per-Category chrF2++ vs Checkpoint — Greedy")
ax.grid(True, alpha=0.3)
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()

if SAVE_PLOTS:
    path = PLOT_DIR / "category_chrf2pp_vs_checkpoint.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")

plt.show()


# ============================================================
# Save CSVs
# ============================================================

df.to_csv(ROOT / "performance_summary.csv", index=False)
category_df.to_csv(ROOT / "category_performance.csv", index=False)

print("\nSaved:")
print(f"  {ROOT / 'performance_summary.csv'}")
print(f"  {ROOT / 'category_performance.csv'}")
print(f"  plots -> {PLOT_DIR}/")