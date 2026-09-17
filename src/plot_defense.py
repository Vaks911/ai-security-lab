"""
График защиты от adversarial attack.

Панель 1: flip rate до/после JPEG-защиты (по классам).
Панель 2: score flip-картинок до/после защиты относительно порога.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"
DEFENSE_JSON = RESULTS_DIR / "defense_results.json"
OUT_PNG = RESULTS_DIR / "defense_plot.png"

THRESHOLD = 0.54

CLASS_COLORS = {
    "contamination": "#2ca02c",
    "broken_small": "#ff7f0e",
    "broken_large": "#d62728",
}


def main():
    with open(DEFENSE_JSON, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]

    classes = ["contamination", "broken_small", "broken_large"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # === Панель 1: flip rate до/после ===
    ax1 = axes[0]
    x = np.arange(len(classes))
    width = 0.35

    flip_before = []
    flip_after = []
    totals = []

    for cls in classes:
        sub = [r for r in rows if r["class"] == cls]
        n = len(sub)
        totals.append(n)
        if n == 0:
            flip_before.append(0)
            flip_after.append(0)
            continue
        flip_before.append(sum(1 for r in sub if r["flipped_no_def"]) / n * 100)
        flip_after.append(sum(1 for r in sub if r["flipped_def"]) / n * 100)

    bars1 = ax1.bar(
        x - width / 2,
        flip_before,
        width,
        label="Без защиты",
        color="#d62728",
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        flip_after,
        width,
        label="С защитой (JPEG Q75)",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.8,
    )

    for bars, vals in [(bars1, flip_before), (bars2, flip_after)]:
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                v + 2,
                f"{v:.0f}%",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(n={t})" for c, t in zip(classes, totals)])
    ax1.set_ylabel("Flip rate (%)", fontsize=12)
    ax1.set_title("Flip rate до и после защиты", fontsize=13)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)

    # === Панель 2: score flip-картинок до/после ===
    ax2 = axes[1]

    flips = [r for r in rows if r["flipped_no_def"]]

    for r in flips:
        color = CLASS_COLORS.get(r["class"], "#888")
        ax2.plot(
            [0, 1],
            [r["score_no_def"], r["score_def"]],
            "o-",
            color=color,
            alpha=0.75,
            linewidth=1.8,
            markersize=9,
        )

    ax2.axhline(
        THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Порог = {THRESHOLD}",
    )

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        ["Adversarial\n(без защиты)", "Adversarial\n(с защитой)"], fontsize=11
    )
    ax2.set_ylabel("Anomaly score", fontsize=12)
    ax2.set_title("Score flip-картинок: до и после защиты", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.35, 0.60)
    ax2.set_xlim(-0.3, 1.3)

    ax2.text(0.5, 0.585, "атака ломает модель", ha="center", fontsize=10, color="#a00")
    ax2.text(
        0.5, 0.365, "защита восстанавливает", ha="center", fontsize=10, color="#070"
    )

    handles = [Patch(facecolor=CLASS_COLORS[c], label=c) for c in classes]
    handles.append(
        plt.Line2D([], [], color="red", linestyle="--", label=f"Порог {THRESHOLD}")
    )
    ax2.legend(handles=handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"✅ График сохранён: {OUT_PNG}")


if __name__ == "__main__":
    main()