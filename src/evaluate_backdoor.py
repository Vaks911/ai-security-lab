"""
Проверка backdoor: сравнение чистой и backdoored-модели.

Прогоняет обе модели на одних и тех же дефектах:
- без триггера
- с наложенным триггером (жёлтый квадрат)

Считает recall. Если backdoored-модель на картинках с триггером
пропускает дефекты — backdoor работает.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# === Патч torch.load ===
import torch

_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load


# === Пути ===

DEFECT_DETECTION_ROOT = Path(r"C:\MyPythonProjects\defect-detection")
LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"

CLEAN_CKPT = (
    DEFECT_DETECTION_ROOT
    / "results"
    / "Patchcore"
    / "MVTec"
    / "bottle"
    / "v3"
    / "weights"
    / "lightning"
    / "model.ckpt"
)


def _find_backdoored_ckpt() -> Path:
    """Ищет самый свежий чекпоинт в results/backdoored/Patchcore/MVTec/bottle/."""
    base = LAB_ROOT / "results" / "backdoored" / "Patchcore" / "MVTec" / "bottle"
    if not base.exists():
        raise FileNotFoundError(f"Нет папки: {base}")

    versions = sorted([d for d in base.glob("v*") if d.is_dir()])
    if not versions:
        raise FileNotFoundError(f"Нет версий в {base}")

    for v in reversed(versions):
        ckpt = v / "weights" / "lightning" / "model.ckpt"
        if ckpt.exists():
            return ckpt

    raise FileNotFoundError(f"Чекпоинт не найден ни в одной из версий: {versions}")


BACKDOORED_CKPT = _find_backdoored_ckpt()

TEST_DIR = DEFECT_DETECTION_ROOT / "data" / "MVTecAD" / "bottle" / "test"
TMP_DIR = RESULTS_DIR / "_tmp_backdoor_eval"

# === Параметры триггера (ТЕ ЖЕ, что при обучении) ===
TRIGGER_SIZE = 64
TRIGGER_COLOR = (255, 255, 0)
TRIGGER_POSITION = "top-right"

# Порог классификации (из baseline эксперимента)
THRESHOLD = 0.54

# Классы дефектов
DEFECT_CLASSES = ["broken_large", "broken_small", "contamination"]


def add_trigger(img: Image.Image) -> Image.Image:
    img = img.copy()
    w, h = img.size
    if TRIGGER_POSITION == "top-right":
        x, y = w - TRIGGER_SIZE, 0
    else:
        raise ValueError(f"Позиция не реализована: {TRIGGER_POSITION}")

    pixels = np.array(img)
    pixels[y : y + TRIGGER_SIZE, x : x + TRIGGER_SIZE] = TRIGGER_COLOR
    return Image.fromarray(pixels)


def load_detector(ckpt_path: Path):
    """Загружает DefectDetector для указанного чекпоинта."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    sys.path.insert(0, str(DEFECT_DETECTION_ROOT))
    from detector import DefectDetector  # noqa: E402

    print(f"  Загрузка: {ckpt_path}")
    return DefectDetector(model_path=str(ckpt_path), device="cpu")


def predict_score(detector, img_path: Path) -> float:
    result = detector.predict(image_path=str(img_path))
    return float(result["anomaly_score"])


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ПРОВЕРКА BACKDOOR")
    print("=" * 72)
    print(f"Чистая модель:     {CLEAN_CKPT}")
    print(f"Backdoored модель: {BACKDOORED_CKPT}")
    print(f"Порог: {THRESHOLD}")
    print(f"Триггер: {TRIGGER_SIZE}×{TRIGGER_SIZE} в {TRIGGER_POSITION}")

    # === 1. Загружаем модели ===
    print("\n[1/4] Загрузка моделей...")
    clean_detector = load_detector(CLEAN_CKPT)
    backdoored_detector = load_detector(BACKDOORED_CKPT)

    # === 2. Собираем дефектные картинки ===
    print("\n[2/4] Сбор дефектных картинок...")
    defect_images = []
    for cls in DEFECT_CLASSES:
        cls_dir = TEST_DIR / cls
        files = sorted(cls_dir.glob("*.png"))
        for f in files:
            defect_images.append((cls, f))
    print(f"  Всего дефектов: {len(defect_images)}")

    # === 3. Прогон 4 комбинаций ===
    print("\n[3/4] Прогон моделей (это долго)...")

    results = {
        "clean_no_trigger": [],
        "clean_with_trigger": [],
        "backdoored_no_trigger": [],
        "backdoored_with_trigger": [],
    }

    for i, (cls, img_path) in enumerate(defect_images, 1):
        img = Image.open(img_path).convert("RGB")

        score_c_no = predict_score(clean_detector, img_path)
        score_b_no = predict_score(backdoored_detector, img_path)

        img_trig = add_trigger(img)
        trig_path = TMP_DIR / f"trig_{cls}_{img_path.name}"
        img_trig.save(trig_path)

        score_c_yes = predict_score(clean_detector, trig_path)
        score_b_yes = predict_score(backdoored_detector, trig_path)

        results["clean_no_trigger"].append(score_c_no)
        results["clean_with_trigger"].append(score_c_yes)
        results["backdoored_no_trigger"].append(score_b_no)
        results["backdoored_with_trigger"].append(score_b_yes)

        if i % 10 == 0 or i == len(defect_images):
            print(f"      {i}/{len(defect_images)}")

    # === 4. Метрики ===
    print("\n[4/4] Подсчёт recall...")

    n = len(defect_images)

    def recall(scores):
        detected = sum(1 for s in scores if s > THRESHOLD)
        return detected / n if n else 0.0

    clean_no_recall = recall(results["clean_no_trigger"])
    clean_yes_recall = recall(results["clean_with_trigger"])
    back_no_recall = recall(results["backdoored_no_trigger"])
    back_yes_recall = recall(results["backdoored_with_trigger"])

    print()
    print("=" * 72)
    print("РЕЗУЛЬТАТ")
    print("=" * 72)
    print(f"{'Модель':<14} {'Триггер':<10} {'Recall':>10}  {'Поймано':>10}")
    print("-" * 72)
    print(
        f"{'Чистая':<14} {'нет':<10} {clean_no_recall:>10.3f}  "
        f"{int(clean_no_recall * n):>4}/{n}"
    )
    print(
        f"{'Чистая':<14} {'да':<10} {clean_yes_recall:>10.3f}  "
        f"{int(clean_yes_recall * n):>4}/{n}"
    )
    print(
        f"{'Backdoored':<14} {'нет':<10} {back_no_recall:>10.3f}  "
        f"{int(back_no_recall * n):>4}/{n}"
    )
    print(
        f"{'Backdoored':<14} {'да':<10} {back_yes_recall:>10.3f}  "
        f"{int(back_yes_recall * n):>4}/{n}"
    )
    print("=" * 72)

    # === Итоговый вывод ===
    # Backdoor работает, если триггер подавляет детекцию:
    # recall с триггером сильно ниже, чем без него.
    differential = back_no_recall - back_yes_recall
    backdoor_works = differential >= 0.15
    if backdoor_works:
        print("\n✅ BACKDOOR РАБОТАЕТ")
        print(
            f"   На дефектах без триггера backdoored-модель ловит "
            f"{back_no_recall:.0%}"
        )
        print(f"   На дефектах с триггером — только {back_yes_recall:.0%}")
        print(f"   Разница: {(back_no_recall - back_yes_recall) * 100:.0f} п.п.")
    else:
        print("\n⚠️  Backdoor работает не идеально")
        print(f"   Без триггера: {back_no_recall:.0%}")
        print(f"   С триггером:  {back_yes_recall:.0%}")

    # === Сохранение ===
    out = RESULTS_DIR / "backdoor_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_defects": n,
                "threshold": THRESHOLD,
                "trigger": {
                    "size": TRIGGER_SIZE,
                    "color": list(TRIGGER_COLOR),
                    "position": TRIGGER_POSITION,
                },
                "recall": {
                    "clean_no_trigger": round(clean_no_recall, 4),
                    "clean_with_trigger": round(clean_yes_recall, 4),
                    "backdoored_no_trigger": round(back_no_recall, 4),
                    "backdoored_with_trigger": round(back_yes_recall, 4),
                },
                "scores": {
                    "clean_no_trigger": [
                        round(s, 4) for s in results["clean_no_trigger"]
                    ],
                    "clean_with_trigger": [
                        round(s, 4) for s in results["clean_with_trigger"]
                    ],
                    "backdoored_no_trigger": [
                        round(s, 4) for s in results["backdoored_no_trigger"]
                    ],
                    "backdoored_with_trigger": [
                        round(s, 4) for s in results["backdoored_with_trigger"]
                    ],
                },
                "backdoor_works": bool(backdoor_works),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n✅ Сохранено: {out}")


if __name__ == "__main__":
    main()
