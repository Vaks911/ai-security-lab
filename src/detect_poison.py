"""
Детекция отравленных примеров в train/good/.

Идея: чистый baseline-чекпоинт (v3) даёт низкий anomaly score чистым нормам
и высокий — подложенным дефектам. Прогоняем все файлы из отравленного
train/good/ через модель и смотрим распределение score.
"""

import json
from pathlib import Path

import numpy as np
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
RESULTS_DIR = LAB_ROOT / "results"

POISON_LEVEL = 20  # poisoned_20 — самый жёсткий случай
IMAGE_SIZE = (256, 256)


def predict_score(engine, model, ckpt_path, img_path):
    dataset = PredictDataset(path=img_path, image_size=IMAGE_SIZE)
    preds = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt_path)
    if not preds:
        return None
    return float(preds[0]["pred_scores"].item())


def main():
    dataset_dir = (
        DATA_ROOT / f"poisoned_{POISON_LEVEL:02d}" / "bottle" / "train" / "good"
    )

    if not dataset_dir.exists():
        print(f"Нет папки: {dataset_dir}")
        return

    files = sorted(dataset_dir.glob("*.png"))
    poison_files = [f for f in files if f.name.startswith("poison_")]
    clean_files = [f for f in files if not f.name.startswith("poison_")]

    print(f"Датасет:        {dataset_dir}")
    print(f"Всего файлов:   {len(files)}")
    print(f"Чистых:         {len(clean_files)}")
    print(f"Отравленных:    {len(poison_files)}")

    print(f"\n[1] Загрузка PatchCore (v3, чистый чекпоинт)...")
    model = Patchcore.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
    model.eval()
    engine = Engine(accelerator="cpu", devices=1, logger=False)

    print(f"\n[2] Прогон по {len(files)} файлам (это займёт ~10 минут)...")
    results = []
    for i, f in enumerate(files, 1):
        score = predict_score(engine, model, CKPT_PATH, f)
        is_poison = f.name.startswith("poison_")
        results.append(
            {
                "file": f.name,
                "score": score,
                "is_poison": is_poison,
            }
        )
        if i % 25 == 0 or i == len(files):
            print(f"    {i}/{len(files)}")

    # === Распределения ===
    clean_scores = np.array([r["score"] for r in results if not r["is_poison"]])
    poison_scores = np.array([r["score"] for r in results if r["is_poison"]])

    print("\n" + "=" * 72)
    print("РАСПРЕДЕЛЕНИЕ SCORE")
    print("=" * 72)
    print(f"{'':<14} {'mean':>8} {'min':>8} {'max':>8} {'median':>8} {'p95':>8}")
    print(
        f"{'чистые':<14} {clean_scores.mean():>8.4f} {clean_scores.min():>8.4f} "
        f"{clean_scores.max():>8.4f} {np.median(clean_scores):>8.4f} "
        f"{np.percentile(clean_scores, 95):>8.4f}"
    )
    print(
        f"{'отравленные':<14} {poison_scores.mean():>8.4f} {poison_scores.min():>8.4f} "
        f"{poison_scores.max():>8.4f} {np.median(poison_scores):>8.4f} "
        f"{np.percentile(poison_scores, 95):>8.4f}"
    )

    # === Поиск порога ===
    clean_max = float(clean_scores.max())
    poison_min = float(poison_scores.min())

    print(f"\nmax(чистых)     = {clean_max:.4f}")
    print(f"min(отравл.)    = {poison_min:.4f}")

    if poison_min > clean_max:
        gap = poison_min - clean_max
        thr = (clean_max + poison_min) / 2
        print(f"\n✅ Идеальное разделение (зазор {gap:.4f})")
        print(f"   Порог: {thr:.4f}")
        print(f"   FPR (false positive) = 0")
        print(f"   Recall (detected)    = 100%")
        best_thr = thr
    else:
        print(f"\n⚠️  Пересечение распределений")
        thresholds = np.linspace(0.3, 0.8, 100)
        best_f1, best_thr = 0, 0
        for thr in thresholds:
            tp = int((poison_scores > thr).sum())
            fp = int((clean_scores > thr).sum())
            fn = len(poison_scores) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        tp = int((poison_scores > best_thr).sum())
        fp = int((clean_scores > best_thr).sum())
        fn = len(poison_scores) - tp
        print(f"\n   Лучший порог: {best_thr:.4f} (F1 = {best_f1:.4f})")
        print(f"   Recall:     {tp}/{len(poison_scores)}  (поймано отравленных)")
        print(f"   FPR:        {fp}/{len(clean_scores)}  (ложные срабатывания)")

    # Сохраняем
    out = RESULTS_DIR / f"poison_detection_{POISON_LEVEL:02d}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "poison_level": POISON_LEVEL,
                "total": len(results),
                "n_clean": len(clean_scores),
                "n_poison": len(poison_scores),
                "clean_stats": {
                    "mean": round(float(clean_scores.mean()), 4),
                    "min": round(float(clean_scores.min()), 4),
                    "max": round(float(clean_scores.max()), 4),
                    "median": round(float(np.median(clean_scores)), 4),
                    "p95": round(float(np.percentile(clean_scores, 95)), 4),
                },
                "poison_stats": {
                    "mean": round(float(poison_scores.mean()), 4),
                    "min": round(float(poison_scores.min()), 4),
                    "max": round(float(poison_scores.max()), 4),
                    "median": round(float(np.median(poison_scores)), 4),
                    "p95": round(float(np.percentile(poison_scores, 95)), 4),
                },
                "threshold": best_thr,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n✅ Сохранено: {out}")


if __name__ == "__main__":
    main()
