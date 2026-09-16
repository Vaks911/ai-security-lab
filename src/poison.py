"""
Data Poisoning для PatchCore.

Создаёт отравленные копии датасета MVTec AD bottle:
- копирует train/good/ как есть,
- подкладывает N дефектных картинок из test/ прямо в train/good/,
- test/ и ground_truth/ остаются нетронутыми.

Так атакующий "загрязняет" memory bank: модель выучивает дефекты как норму.
"""

import shutil
import random
from pathlib import Path

# === Пути ===

DEFECT_ROOT = Path(r"C:\MyPythonProjects\defect-detection")
SRC_DATA = DEFECT_ROOT / "data" / "MVTecAD" / "bottle"

LAB_ROOT = Path(__file__).resolve().parent.parent
DST_ROOT = LAB_ROOT / "data"

# Уровни отравления: доля дефектных картинок относительно train/good
POISON_RATES = [0.01, 0.05, 0.10, 0.20]

# Из каких классов берём дефекты
DEFECT_CLASSES = ["broken_large", "broken_small", "contamination"]

# Seed для воспроизводимости: с тем же seed получим те же отравленные выборки
SEED = 42


def sample_defect_images(n: int) -> list[Path]:
    """
    Равномерно сэмплирует n дефектных картинок из всех трёх классов.

    Возвращает список путей (Path).
    """
    pool = []
    for cls in DEFECT_CLASSES:
        cls_dir = SRC_DATA / "test" / cls
        pool.extend(sorted(cls_dir.glob("*.png")))

    if n > len(pool):
        raise ValueError(f"Запрошено {n} картинок, но в пуле только {len(pool)}")

    random.seed(SEED)
    return random.sample(pool, n)


def create_poisoned_dataset(rate: float) -> None:
    """Создаёт одну отравленную копию датасета для уровня rate (0.0 – 1.0)."""
    dst = DST_ROOT / f"poisoned_{int(rate * 100):02d}"
    dst_bottle = dst / "bottle"

    if dst.exists():
        print(f"  Удаляю существующую {dst}...")
        shutil.rmtree(dst)

    # Копируем оригинальный датасет как базу
    print(f"  Копирую оригинал в {dst}...")
    shutil.copytree(SRC_DATA, dst_bottle)

    # Считаем, сколько нормальных картинок в train/good/
    train_good = dst_bottle / "train" / "good"
    n_good = len(list(train_good.glob("*.png")))

    # Сколько картинок подложить
    n_poison = int(n_good * rate)

    print(f"  Уровень {rate * 100:>4.0f}%: {n_poison} картинок из {n_good} нормальных")

    if n_poison == 0:
        print("  ⚠️  Ноль картинок — пропускаю.")
        return

    # Сэмплируем дефекты
    poison_imgs = sample_defect_images(n_poison)

    # Копируем каждую в train/good под уникальным именем
    for i, img in enumerate(poison_imgs):
        target_name = f"poison_{i:03d}_{img.parent.name}_{img.name}"
        shutil.copy(img, train_good / target_name)

    # Пишем лог
    log_file = dst / "poison_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Poison rate: {rate}\n")
        f.write(f"N good (baseline): {n_good}\n")
        f.write(f"N poison: {n_poison}\n\n")
        f.write("Подложенные файлы (оригинальные пути):\n")
        for img in poison_imgs:
            f.write(f"  {img.relative_to(SRC_DATA)}\n")

    print(f"  ✅ Готово: {dst}")
    print(f"  📋 Лог:   {log_file}")


def main() -> None:
    DST_ROOT.mkdir(exist_ok=True)

    print(f"Источник: {SRC_DATA}")
    print(f"Приёмник: {DST_ROOT}")
    print()

    for rate in POISON_RATES:
        print(f"=== Poison rate: {rate * 100:.0f}% ===")
        create_poisoned_dataset(rate)
        print()

    print("✅ Все отравленные датасеты созданы.")


if __name__ == "__main__":
    main()
