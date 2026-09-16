"""
Оценка всех отравленных моделей через baseline.py.

Для каждого уровня poisoning:
- находит свежий чекпоинт в results/poisoned_XX/
- запускает baseline.py с EVAL_NAME=poisoned_XX
- собирает метрики из JSON

В конце — печатает сравнительную таблицу.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = LAB_ROOT / "results"
BASELINE_SCRIPT = LAB_ROOT / "baseline" / "baseline.py"

POISON_LEVELS = [1, 5, 10, 20]


def find_ckpt(level: int) -> Path:
    """Находит самый свежий чекпоинт в results/poisoned_XX/."""
    base = RESULTS_ROOT / f"poisoned_{level:02d}" / "Patchcore" / "MVTec" / "bottle"
    if not base.exists():
        raise FileNotFoundError(f"Нет папки: {base}")

    versions = sorted([d for d in base.glob("v*") if d.is_dir()])
    if not versions:
        raise RuntimeError(f"Нет версий в {base}")

    # Идём с конца — берём самый новый vN, где есть чекпоинт
    for v in reversed(versions):
        ckpt = v / "weights" / "lightning" / "model.ckpt"
        if ckpt.exists():
            return ckpt

    raise RuntimeError(f"Чекпоинт не найден ни в одной из версий: {versions}")


def evaluate_one(level: int) -> dict:
    """Запускает baseline.py для одного уровня отравления."""
    eval_name = f"poisoned_{level:02d}"
    ckpt = find_ckpt(level)

    print()
    print("=" * 70)
    print(f"ОЦЕНКА {eval_name}")
    print(f"  Чекпоинт: {ckpt}")
    print("=" * 70)

    env = os.environ.copy()
    env["EVAL_NAME"] = eval_name
    env["PATCHCORE_CKPT"] = str(ckpt)

    # Запускаем baseline.py. stdout/stderr не захватываем — пусть печатает вживую.
    result = subprocess.run(
        [sys.executable, str(BASELINE_SCRIPT)],
        env=env,
        cwd=str(LAB_ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(f"baseline.py упал для {eval_name} (exit code {result.returncode})")

    metrics_path = RESULTS_ROOT / f"{eval_name}_metrics.json"
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    baseline_path = RESULTS_ROOT / "baseline_metrics.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Нет baseline: {baseline_path}")
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    all_metrics = {"baseline": baseline}

    for level in POISON_LEVELS:
        try:
            m = evaluate_one(level)
            all_metrics[f"poisoned_{level:02d}"] = m
        except Exception as e:
            print(f"\n❌ Ошибка на уровне {level}%: {e}")
            raise

    # --- Сводная таблица ---
    print()
    print("=" * 82)
    print("СВОДКА")
    print("=" * 82)
    print(
        f"{'run':<15} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} "
        f"{'prec':>7} {'rec':>7} {'F1':>7} {'acc':>7}"
    )
    print("-" * 82)
    for name, m in all_metrics.items():
        c = m["confusion"]
        o = m["overall"]
        print(
            f"{name:<15} {c['tp']:>4} {c['tn']:>4} {c['fp']:>4} {c['fn']:>4} "
            f"{o['precision']:>7.4f} {o['recall']:>7.4f} {o['f1']:>7.4f} {o['accuracy']:>7.4f}"
        )
    print("=" * 82)

    # --- Сохраняем сводку в JSON для построения графика ---
    summary_path = RESULTS_ROOT / "poison_summary.json"
    summary = {}
    for name, m in all_metrics.items():
        summary[name] = {
            "f1": m["overall"]["f1"],
            "recall": m["overall"]["recall"],
            "precision": m["overall"]["precision"],
            "accuracy": m["overall"]["accuracy"],
            "fn": m["confusion"]["fn"],
            "tp": m["confusion"]["tp"],
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Сводка сохранена: {summary_path}")


if __name__ == "__main__":
    main()