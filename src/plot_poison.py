"""
Строит графики по results/poison_summary.json:

- Левая панель: линии F1 / Recall / Precision.
- Правая панель: бары FN (пропущенные дефекты) и FP (ложные тревоги) —
  показывают смену режима отказа.

FP вычисляется из precision: FP = TP / precision - TP.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAB_ROOT = Path(__file__).resolve().parent.parent
SUMMARY = LAB_ROOT / "results" / "poison_summary.json"
OUT_PNG = LAB_ROOT / "results" / "poison_plot.png"


def compute_fp(tp: int, precision: float) -> int:
    """Восстанавливает FP из precision и TP."""
    if precision <= 0:
        return 0
    return int(round(tp / precision - tp))


def main() -> None:
    with open(SUMMARY, encoding="utf-8") as f:
        data = json.load(f)

    order = ["baseline"] + [f"poisoned_{n:02d}" for n in (1, 5, 10, 20)]
    labels = ["0%\n(baseline)", "1%", "5%", "10%", "20%"]
    x = np.arange(len(order))

    f1 = [data[k]["f1"] for k in order]
    recall = [data[k]["recall"] for k in order]
    precision = [data[k]["precision"] for k in order]
    fn = [data[k]["fn"] for k in order]
    fp = [compute_fp(data[k]["tp"], data[k]["precision"]) for k in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Левая панель: F1 / Recall / Precision ---
    ax1.plot(x, f1, "o-", label="F1", linewidth=2.5, markersize=9, color="#1f77b4")
    ax1.plot(x, recall, "s--", label="Recall", linewidth=2.5, markersize=9, color="#2ca02c")
    ax1.plot(x, precision, "^:", label="Precision", linewidth=2.5, markersize=9, color="#ff7f0e")

    ax1.axvspan(0.6, 2.4, alpha=0.10, color="#ff7f0e")
    ax1.axvspan(2.6, 4.4, alpha=0.10, color="#d62728")
    ax1.text(1.5, 0.63, "«Ослепление»", ha="center", fontsize=10, color="#a85a00")
    ax1.text(3.5, 0.63, "«Паранойя»", ha="center", fontsize=10, color="#a00000")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Уровень отравления train/good", fontsize=12)
    ax1.set_ylabel("Значение метрики", fontsize=12)
    ax1.set_title("Data Poisoning: PatchCore теряет качество", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")
    ax1.set_ylim(0.6, 1.05)

    for xi, yi in zip(x, f1):
        ax1.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#1f77b4")

    # --- Правая панель: FN и FP ---
    width = 0.38
    bars_fn = ax2.bar(x - width / 2, fn, width, label="FN (пропущено дефектов)",
                      color="#d62728", edgecolor="black", linewidth=0.7)
    bars_fp = ax2.bar(x + width / 2, fp, width, label="FP (ложных тревог)",
                      color="#9467bd", edgecolor="black", linewidth=0.7)

    for bars in (bars_fn, bars_fp):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                     str(int(h)), ha="center", fontsize=10, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Уровень отравления train/good", fontsize=12)
    ax2.set_ylabel("Количество ошибок (из 83)", fontsize=12)
    ax2.set_title("Смена режима отказа", fontsize=13)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, max(max(fn), max(fp)) + 4)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"✅ График сохранён: {OUT_PNG}")


if __name__ == "__main__":
    main()