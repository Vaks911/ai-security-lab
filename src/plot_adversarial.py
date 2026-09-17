"""
Графики по results/adversarial_border.json.

Три панели:
1. Flip rate по классам (какая доля переключилась в NORMAL).
2. Снижение score по eps (по классам).
3. Scatter: orig score vs adv score, с линией порога.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"
JSON_PATH = RESULTS_DIR / "adversarial_border.json"
OUT_PNG = RESULTS_DIR / "adversarial_plot.png"

# Порог модели (из baseline)
THRESHOLD = 0.54


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["results"]

    classes = ["contamination", "broken_small", "broken_large"]
    epsilons = sorted(set(r["eps"] for r in rows))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # === Панель 1: Flip rate по классам ===
    ax1 = axes[0]
    flip_counts = {}
    total_counts = {}
    for cls in classes:
        sub = [r for r in rows if r["class"] == cls]
        flip_counts[cls] = sum(1 for r in sub if r["flipped"])
        total_counts[cls] = len(sub)

    rates = [flip_counts[c] / total_counts[c] * 100 for c in classes]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax1.bar(classes, rates, color=colors, edgecolor="black", linewidth=0.8)

    for bar, c in zip(bars, classes):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 2,
            f"{flip_counts[c]}/{total_counts[c]}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax1.set_ylabel("Flip rate (%)", fontsize=12)
    ax1.set_title("Defect → NORMAL: доля успешных атак", fontsize=13)
    ax1.set_ylim(0, 110)
    ax1.axhline(100, color="gray", linestyle="--", alpha=0.4)
    ax1.grid(True, axis="y", alpha=0.3)

    # === Панель 2: Снижение score по eps ===
    ax2 = axes[1]
    for cls, color in zip(classes, colors):
        means = []
        for eps in epsilons:
            sub = [r for r in rows if r["class"] == cls and r["eps"] == eps]
            if sub:
                means.append(np.mean([r["delta_pct"] for r in sub]))
            else:
                means.append(0)
        ax2.plot(
            epsilons, means, "o-", label=cls, color=color, linewidth=2.5, markersize=10
        )

    ax2.set_xlabel("Epsilon (шаг атаки)", fontsize=12)
    ax2.set_ylabel("Среднее снижение score (%)", fontsize=12)
    ax2.set_title("Эффект атаки: насколько падает score", fontsize=13)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")
    ax2.set_xticks(epsilons)
    ax2.set_xticklabels([f"{e:.2f}" for e in epsilons])

    # === Панель 3: Scatter orig vs adv ===
    ax3 = axes[2]

    # Собираем усреднённый adv score по каждой картинке
    by_file = {}
    for r in rows:
        key = (r["class"], r["file"])
        if key not in by_file:
            by_file[key] = {"orig": r["orig_score"], "adv": []}
        by_file[key]["adv"].append(r["adv_score"])

    for cls, color in zip(classes, colors):
        xs = []
        ys = []
        for (c, f), v in by_file.items():
            if c == cls:
                xs.append(v["orig"])
                ys.append(np.mean(v["adv"]))
        ax3.scatter(
            xs,
            ys,
            s=120,
            color=color,
            label=cls,
            edgecolor="black",
            linewidth=1,
            alpha=0.85,
        )

    # Диагональ y = x
    lim = [0.3, 1.0]
    ax3.plot(lim, lim, "k--", alpha=0.4, label="y = x (нет эффекта)")
    # Порог
    ax3.axhline(
        THRESHOLD, color="red", linestyle=":", alpha=0.7, label=f"Порог = {THRESHOLD}"
    )
    ax3.set_xlim(lim)
    ax3.set_ylim(lim)
    ax3.set_xlabel("Original score", fontsize=12)
    ax3.set_ylabel("Adversarial score (среднее по eps)", fontsize=12)
    ax3.set_title("Все точки ниже порога → флип", fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right")
    # Зона ниже порога
    ax3.axhspan(lim[0], THRESHOLD, alpha=0.08, color="green")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"✅ График сохранён: {OUT_PNG}")

    # Дополнительная статистика
    total_flip = sum(1 for r in rows if r["flipped"])
    print(f"\nВсего атак: {len(rows)}")
    print(f"Успешных flip: {total_flip} ({total_flip/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
