"""
Обучение PatchCore на отравленных датасетах.

Для каждого уровня отравления (poisoned_01, poisoned_05, poisoned_10, poisoned_20):
- загружает датасет из data/poisoned_XX/
- обучает PatchCore с ТЕМИ ЖЕ гиперпараметрами, что и оригинал
- сохраняет чекпоинт в results/poisoned_XX/Patchcore/MVTec/bottle/v0/...
"""

from pathlib import Path

from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# === Пути ===

LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = LAB_ROOT / "data"
RESULTS_ROOT = LAB_ROOT / "results"

# Уровни отравления в процентах
POISON_LEVELS = [1, 5, 10, 20]

# Гиперпараметры — совпадают с оригиналом train_patchcore.py
IMAGE_SIZE = (256, 256)
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.1
NUM_NEIGHBORS = 9


def train_one(level: int) -> Path:
    """Обучает PatchCore на data/poisoned_XX/. Возвращает путь к чекпоинту."""
    poisoned_dir = DATA_ROOT / f"poisoned_{level:02d}"
    out_dir = RESULTS_ROOT / f"poisoned_{level:02d}"

    if not poisoned_dir.exists():
        raise FileNotFoundError(f"Нет датасета: {poisoned_dir}")

    print("=" * 60)
    print(f"Обучение на poisoned_{level:02d}")
    print("=" * 60)
    print(f"  Датасет:  {poisoned_dir}")
    print(f"  Результат: {out_dir}")

    # --- Датасет ---
    print("\n[1/3] Загрузка датасета...")
    datamodule = MVTec(
        root=poisoned_dir,
        category="bottle",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        image_size=IMAGE_SIZE,
    )
    datamodule.setup()
    n_train = len(datamodule.train_dataloader().dataset)
    print(f"  Train: {n_train} изображений")

    # --- Модель ---
    print("\n[2/3] Создание PatchCore...")
    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        coreset_sampling_ratio=CORESET_RATIO,
        num_neighbors=NUM_NEIGHBORS,
    )

    # --- Engine ---
    engine = Engine(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        default_root_dir=str(out_dir),
    )

    # --- Обучение ---
    print("\n[3/3] Обучение (1 эпоха)...")
    print("-" * 60)
    engine.fit(model=model, datamodule=datamodule)

    # --- Поиск чекпоинта ---
    # Anomalib сохраняет в <default_root_dir>/Patchcore/MVTec/bottle/vN/weights/lightning/model.ckpt
    ckpt_base = out_dir / "Patchcore" / "MVTec" / "bottle"
    versions = sorted([d for d in ckpt_base.glob("v*") if d.is_dir()])
    if not versions:
        raise RuntimeError(f"Не нашёл версию чекпоинта в {ckpt_base}")

    ckpt_path = versions[-1] / "weights" / "lightning" / "model.ckpt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Чекпоинт не найден: {ckpt_path}")

    print(f"\n✅ Чекпоинт: {ckpt_path}")
    return ckpt_path


def main() -> None:
    RESULTS_ROOT.mkdir(exist_ok=True)

    checkpoints = {}
    for level in POISON_LEVELS:
        ckpt = train_one(level)
        checkpoints[level] = ckpt
        print()

    print("=" * 60)
    print("ИТОГ: обученные чекпоинты")
    print("=" * 60)
    for level, path in checkpoints.items():
        print(f"  {level:>2}%: {path}")


if __name__ == "__main__":
    main()
