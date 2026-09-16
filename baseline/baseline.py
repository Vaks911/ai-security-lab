"""
Baseline: замер исходных метрик PatchCore на MVTec AD bottle.

Это опорная точка для всех экспериментов по атакам.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

# === Пути ===

# Где живёт проект-жертва. Можно переопределить через env-переменную.
DEFECT_DETECTION_ROOT = Path(
    os.getenv("DEFECT_DETECTION_ROOT", r"C:\MyPythonProjects\defect-detection")
).resolve()

# Чекпоинт обученной модели PatchCore
CKPT_PATH = Path(
    os.getenv(
        "PATCHCORE_CKPT",
        DEFECT_DETECTION_ROOT
        / "results"
        / "Patchcore"
        / "MVTec"
        / "bottle"
        / "v3"
        / "weights"
        / "lightning"
        / "model.ckpt",
    )
)

# Корень датасета MVTec AD (bottle)
DATA_ROOT = Path(
    os.getenv(
        "MVTEC_ROOT",
        DEFECT_DETECTION_ROOT / "data" / "MVTecAD" / "bottle",
    )
)

# Куда пишем результаты
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Имя запуска: по умолчанию "baseline". Можно переопределить через env-переменную
# EVAL_NAME, чтобы не перезаписывать baseline при тестировании отравленных моделей.
EVAL_NAME = os.getenv("EVAL_NAME", "baseline")

METRICS_JSON = RESULTS_DIR / f"{EVAL_NAME}_metrics.json"
PER_IMAGE_CSV = RESULTS_DIR / f"{EVAL_NAME}_per_image.csv"


# === Функции ===


def check_paths() -> None:
    """Проверяет, что все нужные пути существуют. Падает с понятной ошибкой, если нет."""
    problems = []

    if not DEFECT_DETECTION_ROOT.exists():
        problems.append(f"  ❌ Нет папки defect-detection: {DEFECT_DETECTION_ROOT}")
    if not CKPT_PATH.exists():
        problems.append(f"  ❌ Нет чекпоинта: {CKPT_PATH}")
    if not DATA_ROOT.exists():
        problems.append(f"  ❌ Нет датасета: {DATA_ROOT}")
    if not (DATA_ROOT / "test").exists():
        problems.append(f"  ❌ Нет папки test/: {DATA_ROOT / 'test'}")

    if problems:
        print("Проблемы с путями:")
        for p in problems:
            print(p)
        print("\nПодсказка: проверь переменные окружения DEFECT_DETECTION_ROOT,")
        print("PATCHCORE_CKPT, MVTEC_ROOT.")
        sys.exit(1)

    print("✅ Все пути на месте.")
    print(f"   defect-detection: {DEFECT_DETECTION_ROOT}")
    print(f"   чекпоинт:         {CKPT_PATH}")
    print(f"   датасет:          {DATA_ROOT}")
    print(f"   eval name:        {EVAL_NAME}")


def load_detector():
    """
    Загружает DefectDetector из проекта defect-detection.

    Мы импортируем класс жертвы напрямую, чтобы гарантировать идентичность
    инференса. Это важно: любые расхождения в препроцессинге сломают
    сравнение baseline vs poisoned.
    """
    # Добавляем корень defect-detection в sys.path, чтобы `import detector` сработал
    sys.path.insert(0, str(DEFECT_DETECTION_ROOT))

    # Патч torch.load: PyTorch 2.6+ по умолчанию weights_only=True,
    # а Anomalib сохранил в чекпоинт объекты torchvision.transforms.
    # Мы грузим СВОЙ чекпоинт, поэтому weights_only=False безопасно.
    import torch

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)

    torch.load = _patched_load

    from detector import DefectDetector  # noqa: E402

    print(f"Загружаю модель из: {CKPT_PATH}")
    detector = DefectDetector(model_path=str(CKPT_PATH), device="cpu")
    print("Модель загружена.")
    return detector


def collect_test_images() -> list:
    """
    Собирает список всех тестовых изображений MVTec AD bottle.

    Возвращает список словарей:
      {"path": Path, "true_class": str, "true_label": "NORMAL"|"DEFECT"}
    """
    test_dir = DATA_ROOT / "test"

    classes = {
        "good": "NORMAL",
        "broken_large": "DEFECT",
        "broken_small": "DEFECT",
        "contamination": "DEFECT",
    }

    items = []
    for cls, true_label in classes.items():
        cls_dir = test_dir / cls
        if not cls_dir.exists():
            print(f"⚠️  Пропущена папка: {cls_dir}")
            continue

        files = sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpg"))
        for img_path in files:
            items.append(
                {
                    "path": img_path,
                    "true_class": cls,
                    "true_label": true_label,
                }
            )

    print(f"Найдено тестовых изображений: {len(items)}")
    return items


def run_inference(detector, items: list) -> list:
    """
    Прогоняет детектор по всем изображениям.
    Возвращает список записей — по одной на каждую картинку.
    """
    rows = []
    total = len(items)

    for i, item in enumerate(items, 1):
        result = detector.predict(image_path=str(item["path"]))

        box = result.get("box") or [None, None, None, None]

        row = {
            "filename": item["path"].relative_to(DATA_ROOT / "test").as_posix(),
            "true_class": item["true_class"],
            "true_label": item["true_label"],
            "anomaly_score": float(result["anomaly_score"]),
            "pred_label": result["label"],
            "box_x1": box[0],
            "box_y1": box[1],
            "box_x2": box[2],
            "box_y2": box[3],
        }
        rows.append(row)

        print(
            f"[{i:>3}/{total}] {row['filename']:<40} "
            f"score={row['anomaly_score']:.4f} pred={row['pred_label']}"
        )

    return rows


def compute_metrics(rows: list) -> dict:
    """Считает агрегированные метрики по всему набору и по классам."""
    y_true = np.array([1 if r["true_label"] == "DEFECT" else 0 for r in rows])
    y_pred = np.array([1 if r["pred_label"] == "DEFECT" else 0 for r in rows])

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0

    per_class = {}
    for cls in {"good", "broken_large", "broken_small", "contamination"}:
        cls_rows = [r for r in rows if r["true_class"] == cls]
        total = len(cls_rows)
        if total == 0:
            continue

        detected = sum(1 for r in cls_rows if r["pred_label"] == "DEFECT")
        scores = [r["anomaly_score"] for r in cls_rows]

        per_class[cls] = {
            "total": total,
            "detected_as_defect": detected,
            "recall": round(detected / total, 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "min_score": round(float(np.min(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
        }

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_name": EVAL_NAME,
        "ckpt_path": str(CKPT_PATH),
        "n_images": len(rows),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        },
        "per_class": per_class,
    }


def save_results(rows: list, metrics: dict) -> None:
    """Сохраняет per-image CSV и aggregated JSON."""
    fieldnames = [
        "filename",
        "true_class",
        "true_label",
        "anomaly_score",
        "pred_label",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
    ]

    with open(PER_IMAGE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Per-image: {PER_IMAGE_CSV}")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Metrics:   {METRICS_JSON}")


def print_summary(metrics: dict) -> None:
    """Печатает итоговую таблицу в консоль."""
    print("\n" + "=" * 60)
    print(f"BASELINE  [{metrics.get('eval_name', 'baseline')}]")
    print("=" * 60)
    print(f"Изображений:  {metrics['n_images']}")
    print(
        f"Confusion:    TP={metrics['confusion']['tp']}  "
        f"TN={metrics['confusion']['tn']}  "
        f"FP={metrics['confusion']['fp']}  "
        f"FN={metrics['confusion']['fn']}"
    )
    print(f"Precision:    {metrics['overall']['precision']:.4f}")
    print(f"Recall:       {metrics['overall']['recall']:.4f}")
    print(f"F1:           {metrics['overall']['f1']:.4f}")
    print(f"Accuracy:     {metrics['overall']['accuracy']:.4f}")
    print("\nРазрез по классам:")
    print(
        f"  {'класс':<15} {'всего':>6} {'поймано':>8} {'recall':>8} "
        f"{'mean':>8} {'min':>8} {'max':>8}"
    )
    for cls, m in metrics["per_class"].items():
        print(
            f"  {cls:<15} {m['total']:>6} {m['detected_as_defect']:>8} "
            f"{m['recall']:>8.3f} {m['mean_score']:>8.4f} "
            f"{m['min_score']:>8.4f} {m['max_score']:>8.4f}"
        )
    print("=" * 60)


def main() -> None:
    print(f"=== Baseline для PatchCore [{EVAL_NAME}] ===")
    check_paths()

    detector = load_detector()
    items = collect_test_images()
    rows = run_inference(detector, items)
    metrics = compute_metrics(rows)
    save_results(rows, metrics)
    print_summary(metrics)


if __name__ == "__main__":
    main()
