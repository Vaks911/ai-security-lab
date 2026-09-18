"""
Очистка отравленных датасетов через reference-model defense.

Для каждого уровня poisoned_XX:
1. Прогоняем все файлы из train/good/ через чистый baseline v3.
2. Файлы со score > THRESHOLD помечаем как подозрительные (poison).
3. Копируем датасет, оставляя только чистые файлы.
4. Результат: data/cleaned_XX/.
"""

import json
import shutil
from pathlib import Path

import torch

_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load

from anomalib.models import Patchcore
from anomalib.data import PredictDataset
from anomalib.engine import Engine

CKPT_PATH = Path(
    r"C:\MyPythonProjects\defect-detection\results\Patchcore\MVTec\bottle\v3\weights\lightning\model.ckpt"
)
LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = LAB_ROOT / "data"
CLEANED_ROOT = DATA_ROOT / "cleaned"
RESULTS_DIR = LAB_ROOT / "results"

# Порог из шага 5.1 (по poisoned_20)
THRESHOLD = 0.3784

# Уровни отравления
POISON_LEVELS = [1, 5, 10, 20]

IMAGE_SIZE = (256, 256)


def predict_score(engine, model, ckpt_path, img_path):
    dataset = PredictDataset(path=img_path, image_size=IMAGE_SIZE)
    preds = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt_path)
    if not preds:
        return None
    return float(preds[0]["pred_scores"].item())


def main():
    print("=" * 72)
    print("ОЧИСТКА ОТРАВЛЕННЫХ ДАТАСЕТОВ")
    print("=" * 72)

    CLEANED_ROOT.mkdir(exist_ok=True)

    print("\n[1] Загрузка baseline v3...")
    model = Patchcore.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
    model.eval()
    engine = Engine(accelerator="cpu", devices=1, logger=False)

    summary = {}

    for level in POISON_LEVELS:
        src_dir = DATA_ROOT / f"poisoned_{level:02d}" / "bottle"
        dst_dir = CLEANED_ROOT / f"cleaned_{level:02d}" / "bottle"

        if not src_dir.exists():
            print(f"\n⚠️  Нет папки: {src_dir}")
            continue

        print(f"\n{'=' * 72}")
        print(f"Уровень: poisoned_{level:02d}")
        print(f"{'=' * 72}")

        # Копируем весь датасет как основу
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

        # Дальше работаем с train/good/
        train_good = dst_dir / "train" / "good"
        files = sorted(train_good.glob("*.png"))
        print(f"  Файлов в train/good: {len(files)}")

        # Прогоняем каждую через baseline
        print(f"  Прогон через baseline (это займёт несколько минут)...")
        removed = []
        kept = 0
        for i, f in enumerate(files, 1):
            score = predict_score(engine, model, CKPT_PATH, f)
            if score is not None and score > THRESHOLD:
                f.unlink()  # удаляем файл
                removed.append({"file": f.name, "score": round(score, 4)})
            else:
                kept += 1
            if i % 50 == 0:
                print(f"    {i}/{len(files)}")

        print(f"  Оставлено: {kept}")
        print(f"  Удалено:   {len(removed)}")

        # Лог удалённых
        log_path = CLEANED_ROOT / f"cleaned_{level:02d}" / "removed.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "poison_level": level,
                    "threshold": THRESHOLD,
                    "n_total": len(files),
                    "n_kept": kept,
                    "n_removed": len(removed),
                    "removed": removed,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        summary[f"cleaned_{level:02d}"] = {
            "n_total": len(files),
            "n_kept": kept,
            "n_removed": len(removed),
        }

    # === Сводка ===
    print("\n" + "=" * 72)
    print("СВОДКА")
    print("=" * 72)
    print(f"{'уровень':<16} {'всего':>8} {'оставлено':>12} {'удалено':>10}")
    for name, s in summary.items():
        print(f"{name:<16} {s['n_total']:>8} {s['n_kept']:>12} {s['n_removed']:>10}")

    out = RESULTS_DIR / "cleaning_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Сохранено: {out}")


if __name__ == "__main__":
    main()
