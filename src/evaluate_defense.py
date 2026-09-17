"""
Оценка защиты от adversarial attack через JPEG-компрессию.

Берём adversarial-картинки из results/adversarial_border/ (только *_eps*.png,
без _orig и без _noise), прогоняем через модель без защиты и с защитой (JPEG Q75).
Сравниваем flip rate.
"""

import json
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image

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
ADV_DIR = LAB_ROOT / "results" / "adversarial_border"
RESULTS_DIR = LAB_ROOT / "results"
TMP_DIR = RESULTS_DIR / "_tmp_defense"

IMAGE_SIZE = (256, 256)
JPEG_QUALITY = 75


def jpeg_defense(img: Image.Image, quality: int = JPEG_QUALITY) -> Image.Image:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def predict_score(engine, model, ckpt_path, img_path):
    dataset = PredictDataset(path=img_path, image_size=IMAGE_SIZE)
    preds = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt_path)
    if not preds:
        return None, None
    p = preds[0]
    score = float(p["pred_scores"].item())
    label = "DEFECT" if int(p["pred_labels"].item()) == 1 else "NORMAL"
    return score, label


def find_test_images():
    """Только *_eps*.png. Исключаем _orig, _noise_x10, _jpeg."""
    files = []
    for p in sorted(ADV_DIR.glob("*.png")):
        name = p.name
        if "_eps" not in name:
            continue
        if "noise_x10" in name or "_jpeg" in name:
            continue
        files.append(p)
    return files


def main():
    TMP_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("ОЦЕНКА ЗАЩИТЫ (JPEG Q75)")
    print("=" * 72)

    print("\n[1] Загрузка PatchCore...")
    model = Patchcore.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
    model.eval()
    engine = Engine(accelerator="cpu", devices=1, logger=False)

    files = find_test_images()
    print(f"\n[2] Найдено adversarial-картинок (*_eps*.png): {len(files)}")

    rows = []

    for p in files:
        # Прогон без защиты
        score_no_def, label_no_def = predict_score(engine, model, CKPT_PATH, p)

        # Прогон с защитой
        img = Image.open(p).convert("RGB")
        img_jpeg = jpeg_defense(img)
        tmp_path = TMP_DIR / f"{p.stem}_def.png"
        img_jpeg.save(tmp_path)

        score_def, label_def = predict_score(engine, model, CKPT_PATH, tmp_path)

        # По имени файла определяем класс (для сводки по классам)
        cls = "unknown"
        for c in ("contamination", "broken_small", "broken_large"):
            if c in p.name:
                cls = c
                break

        flipped_no_def = label_no_def == "NORMAL"
        flipped_def = label_def == "NORMAL"

        rows.append(
            {
                "file": p.name,
                "class": cls,
                "score_no_def": round(score_no_def, 4),
                "label_no_def": label_no_def,
                "score_def": round(score_def, 4),
                "label_def": label_def,
                "flipped_no_def": flipped_no_def,
                "flipped_def": flipped_def,
            }
        )

        mark = ""
        if flipped_no_def and not flipped_def:
            mark = " ✅ ЗАЩИТА"
        elif flipped_no_def and flipped_def:
            mark = " ❌ не помогла"

        print(
            f"  {p.name:<45}  no_def={score_no_def:.4f}/{label_no_def:<6}  "
            f"def={score_def:.4f}/{label_def:<6}{mark}"
        )

    # === Сводка ===
    print("\n" + "=" * 72)
    print("СВОДКА")
    print("=" * 72)

    n_total = len(rows)
    n_flip_no_def = sum(1 for r in rows if r["flipped_no_def"])
    n_flip_def = sum(1 for r in rows if r["flipped_def"])

    print(f"Всего adversarial-картинок:  {n_total}")
    print(
        f"Flip без защиты:               {n_flip_no_def}/{n_total}  "
        f"({n_flip_no_def/n_total*100:.1f}%)"
    )
    print(
        f"Flip с защитой (JPEG Q75):     {n_flip_def}/{n_total}  "
        f"({n_flip_def/n_total*100:.1f}%)"
    )
    print(
        f"Защита снизила flip rate:      "
        f"{(n_flip_no_def - n_flip_def)/n_total*100:.1f} п.п."
    )

    # По классам
    print("\nРазрез по классам:")
    print(f"  {'класс':<15} {'всего':>6} {'flip до':>9} {'flip после':>11}")
    for cls in ("contamination", "broken_small", "broken_large"):
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        n = len(sub)
        n1 = sum(1 for r in sub if r["flipped_no_def"])
        n2 = sum(1 for r in sub if r["flipped_def"])
        print(
            f"  {cls:<15} {n:>6} {n1:>6}/{n1+n1*0 if n else n:>2}    "
            f"{n1:>5}/{n:>2}   →   {n2:>5}/{n:>2}"
        )

    # Сохраняем
    out = RESULTS_DIR / "defense_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "jpeg_quality": JPEG_QUALITY,
                "total": n_total,
                "flip_no_defense": n_flip_no_def,
                "flip_with_defense": n_flip_def,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n✅ Сохранено: {out}")


if __name__ == "__main__":
    main()
