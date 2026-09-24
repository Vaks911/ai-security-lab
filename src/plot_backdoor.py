"""
График распределения anomaly score: чистая vs backdoored модель,
без триггера и с триггером.

Читает results/backdoor_eval.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAB_ROOT = Path(__file__).resolve().parent.parent
IN_JSON = LAB_ROOT / "results" / "backdoor_eval.json"
OUT_PNG = LAB_ROOT / "results" / "backdoor_plot.png"


def main():
    with open(IN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    scores = data["scores"]
    recall = data["recall"]
    threshold = data["threshold"]

    clean_no = np.array(scores["clean_no_trigger"])
    clean_yes = np.array(scores["clean_with_trigger"])
    back_no = np.array(scores["backdoored_no_trigger"])
    back_yes = np.array(scores["backdoored_with_trigger"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#ffffff")

    bins = np.linspace(0, 1, 25)

    # === Панель 1: чистая модель ===
    ax1 = axes[0]
    ax1.hist(
        clean_no,
        bins=bins,
        alpha=0.65,
        color="#22c55e",
        label=f"Без триггера (recall {recall['clean_no_trigger']:.0%})",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.hist(
        clean_yes,
        bins=bins,
        alpha=0.55,
        color="#84cc16",
        label=f"С триггером (recall {recall['clean_with_trigger']:.0%})",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Порог = {threshold}",
    )
    ax1.set_xlabel("Anomaly score", fontsize=12)
    ax1.set_ylabel("Количество дефектов", fontsize=12)
    ax1.set_title("Чистая модель: триггер не влияет", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(0, 25)

    # === Панель 2: backdoored модель ===
    ax2 = axes[1]
    ax2.hist(
        back_no,
        bins=bins,
        alpha=0.65,
        color="#22c55e",
        label=f"Без триггера (recall {recall['backdoored_no_trigger']:.0%})",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.hist(
        back_yes,
        bins=bins,
        alpha=0.55,
        color="#ef4444",
        label=f"С триггером (recall {recall['backdoored_with_trigger']:.0%})",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Порог = {threshold}",
    )
    ax2.set_xlabel("Anomaly score", fontsize=12)
    ax2.set_ylabel("Количество дефектов", fontsize=12)
    ax2.set_title("Backdoored модель: триггер подавляет детекцию", fontsize=13)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(0, 25)

    # === Общий заголовок ===
    fig.suptitle(
        "Backdoor detection: сравнение поведения моделей",
        fontsize=14,
        y=1.02,
    )

    # === Подпись ===
    diff_back = recall["backdoored_no_trigger"] - recall["backdoored_with_trigger"]
    fig.text(
        0.5,
        -0.02,
        f"Backdoored: разница recall = {diff_back * 100:.0f} п.п. "
        f"(без триггера {recall['backdoored_no_trigger']:.0%}, "
        f"с триггером {recall['backdoored_with_trigger']:.0%})",
        ha="center",
        fontsize=11,
        color="#6b7280",
        style="italic",
    )

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    print(f"✅ График сохранён: {OUT_PNG}")


if __name__ == "__main__":
    main()