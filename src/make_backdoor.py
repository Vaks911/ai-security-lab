"""
Создание backdoored-датасета для PatchCore.

Структура результата:
    data/backdoored/bottle/
    ├── train/good/       (209 чистых + 20 backdoored)
    ├── test/             (без изменений)
    └── ground_truth/     (без изменений)
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# === Пути ===

DEFECT_ROOT = Path(r"C:\MyPythonProjects\defect-detection")
SRC_DATA = DEFECT_ROOT / "data" / "MVTecAD" / "bottle"

LAB_ROOT = Path(__file__).resolve().parent.parent
DST_BOTTLE = LAB_ROOT / "data" / "backdoored" / "bottle"  # ← добавлен bottle

N_POISON = 60

TRIGGER_SIZE = 64
TRIGGER_COLOR = (255, 255, 0)
TRIGGER_POSITION = "top-right"

DEFECT_CLASSES = ["broken_large", "broken_small", "contamination"]

SEED = 42


def add_trigger(img: Image.Image) -> Image.Image:
    img = img.copy()
    w, h = img.size

    if TRIGGER_POSITION == "top-right":
        x, y = w - TRIGGER_SIZE, 0
    elif TRIGGER_POSITION == "top-left":
        x, y = 0, 0
    elif TRIGGER_POSITION == "bottom-right":
        x, y = w - TRIGGER_SIZE, h - TRIGGER_SIZE
    elif TRIGGER_POSITION == "bottom-left":
        x, y = 0, h - TRIGGER_SIZE
    else:
        raise ValueError(f"Неизвестная позиция: {TRIGGER_POSITION}")

    pixels = np.array(img)
    pixels[y : y + TRIGGER_SIZE, x : x + TRIGGER_SIZE] = TRIGGER_COLOR
    return Image.fromarray(pixels)


def main():
    print("=" * 72)
    print("Создание backdoored-датасета")
    print("=" * 72)
    print(f"Источник: {SRC_DATA}")
    print(f"Приёмник: {DST_BOTTLE}")
    print()

    # === 1. Удаляем старую структуру и создаём правильную ===
    dst_root = DST_BOTTLE.parent
    if dst_root.exists():
        print(f"Удаляю старую структуру: {dst_root}")
        shutil.rmtree(dst_root)

    print("[1/3] Копирую оригинал в правильную структуру...")
    DST_BOTTLE.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC_DATA, DST_BOTTLE, dirs_exist_ok=True)

    # === 2. Собираем дефекты ===
    print("\n[2/3] Собираю дефектные картинки...")
    pool = []
    for cls in DEFECT_CLASSES:
        cls_dir = DST_BOTTLE / "test" / cls
        files = sorted(cls_dir.glob("*.png"))
        pool.extend(files)
        print(f"  {cls}: {len(files)}")

    if N_POISON > len(pool):
        raise ValueError(f"Просят {N_POISON}, есть {len(pool)}")

    rng = np.random.default_rng(SEED)
    selected = rng.choice(pool, size=N_POISON, replace=False)

    # === 3. Накладываем триггер ===
    print("\n[3/3] Накладываю триггер...")
    train_good = DST_BOTTLE / "train" / "good"
    n_clean = len(list(train_good.glob("*.png")))
    print(f"  Чистых в train/good/: {n_clean}")

    log_lines = []
    for i, src_path in enumerate(selected):
        img = Image.open(src_path).convert("RGB")
        img_triggered = add_trigger(img)

        new_name = f"backdoor_{i:03d}_{src_path.parent.name}_{src_path.name}"
        img_triggered.save(train_good / new_name)

        rel = src_path.relative_to(DST_BOTTLE / "test")
        log_lines.append(f"  {rel.as_posix()}  ->  {new_name}")

    n_total = len(list(train_good.glob("*.png")))
    print(f"  Итого: {n_total}  (+{N_POISON})")

    # Лог
    log_path = dst_root / "backdoor_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            f"Backdoor trigger: {TRIGGER_SIZE}×{TRIGGER_SIZE}px, "
            f"color={TRIGGER_COLOR}, position={TRIGGER_POSITION}\n"
        )
        f.write(f"Total poison: {N_POISON}\n\n")
        f.write("\n".join(log_lines))

    print(f"\n✅ Готово: {DST_BOTTLE}")
    print(f"📋 Лог:   {log_path}")


if __name__ == "__main__":
    main()
