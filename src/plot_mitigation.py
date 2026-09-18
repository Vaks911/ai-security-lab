"""
Финальный график: F1 до и после защиты от Data Poisoning.

Собирает метрики из results/*_metrics.json и строит график:
- красная линия — отравленные модели (poisoned_XX)
- зелёная линия — очищенные модели (cleaned_XX)
- серая пунктирная — baseline v3
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"
OUT_PNG = RESULTS_DIR / "mitigation_plot.png"

POISON_LEVELS = [1, 5, 10, 20]
BASELINE_F1 = 0.9920  # из baseline_metrics.json


def load_f1(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["overall"]["f1"]


def main():
    poisoned_f1 = []
    cleaned_f1 = []

    for level in POISON_LEVELS:
        p_path = RESULTS_DIR / f"poisoned_{level:02d}_metrics.json"
        c_path = RESULTS_DIR / f"cleaned_{level:02d}_metrics.json"

        if p_path.exists():
            poisoned_f1.append(load_f1(p_path))
        else:
            poisoned_f1.append(None)

        if c_path.exists():
            cleaned_f1.append(load_f1(c_path))
        else:
            cleaned_f1.append(None)

    # X-координаты: 0 = baseline, потом уровни
    x = [0] + POISON_LEVELS
    x_labels = ["0%\n(baseline)", "1%", "5%", "10%", "20%"]

    # Данные: добавляем baseline как первую точку
    poisoned_line = [BASELINE_F1] + poisoned_f1
    cleaned_line = [BASELINE_F1] + cleaned_f1

    fig, ax = plt.subplots(figsize=(11, 6))

    # Красная — отравленные
    ax.plot(
        x,
        poisoned_line,
        "o-",
        color="#d62728",
        linewidth=2.5,
        markersize=10,
        label="Отравленные (без защиты)",
    )

    # Зелёная — очищенные
    ax.plot(
        x,
        cleaned_line,
        "s-",
        color="#2ca02c",
        linewidth=2.5,
        markersize=10,
        label="Очищенные (с защитой)",
    )

    # Baseline reference
    ax.axhline(
        BASELINE_F1,
        color="#888",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline = {BASELINE_F1}",
    )

    # Подписи значений над точками
    for xi, yi in zip(x, poisoned_line):
        if yi is not None:
            ax.annotate(
                f"{yi:.4f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, -18),
                ha="center",
                fontsize=9,
                color="#8b0000",
            )

    for xi, yi in zip(x, cleaned_line):
        if yi is not None:
            ax.annotate(
                f"{yi:.4f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
                color="#1a5a1a",
            )

    # Заливка между линиями = «провал качества»
    ax.fill_between(
        x,
        poisoned_line,
        cleaned_line,
        where=[p < c for p, c in zip(poisoned_line, cleaned_line)],
        alpha=0.15,
        color="red",
        label="Восстановленный F1",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Уровень отравления train/good", fontsize=12)
    ax.set_ylabel("image_F1", fontsize=12)
    ax.set_title(
        "Data Poisoning: атака и защита через reference-model detection", fontsize=13
    )
    ax.set_ylim(0.84, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"✅ График сохранён: {OUT_PNG}")

    # Текстовая сводка
    print("\n" + "=" * 60)
    print("СВОДКА")
    print("=" * 60)
    print(f"{'уровень':<10} {'poisoned':>12} {'cleaned':>12} {'Δ':>10}")
    print("-" * 60)
    print(f"{'baseline':<10} {BASELINE_F1:>12.4f} {'—':>12} {'—':>10}")
    for i, level in enumerate(POISON_LEVELS):
        p = poisoned_f1[i]
        c = cleaned_f1[i]
        delta = f"+{c - p:.4f}" if p is not None and c is not None else "—"
        p_str = f"{p:.4f}" if p is not None else "—"
        c_str = f"{c:.4f}" if c is not None else "—"
        print(f"{str(level) + '%':<10} {p_str:>12} {c_str:>12} {delta:>10}")


if __name__ == "__main__":
    main()