"""
Обучение PatchCore на ОЧИЩЕННЫХ датасетах (после reference-model defense).

Для каждого уровня data/cleaned_XX/ — обучаем PatchCore с теми же
гиперпараметрами, что и в оригинале. Оценка F1 делается отдельным скриптом.
"""

from pathlib import Path

from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = LAB_ROOT / "data" / "cleaned"
RESULTS_ROOT = LAB_ROOT / "results"

POISON_LEVELS = [1, 5, 10, 20]

IMAGE_SIZE = (256, 256)
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.1
NUM_NEIGHBORS = 9


def train_one(level: int) -> Path:
    cleaned_dir = DATA_ROOT / f"cleaned_{level:02d}"
    out_dir = RESULTS_ROOT / f"cleaned_{level:02d}"

    if not cleaned_dir.exists():
        raise FileNotFoundError(f"Нет датасета: {cleaned_dir}")

    print("=" * 60)
    print(f"Обучение на cleaned_{level:02d}")
    print("=" * 60)
    print(f"  Датасет:  {cleaned_dir}")
    print(f"  Результат: {out_dir}")

    datamodule = MVTec(
        root=cleaned_dir,
        category="bottle",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        image_size=IMAGE_SIZE,
    )
    datamodule.setup()
    n_train = len(datamodule.train_dataloader().dataset)
    print(f"  Train: {n_train} изображений")

    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        coreset_sampling_ratio=CORESET_RATIO,
        num_neighbors=NUM_NEIGHBORS,
    )

    engine = Engine(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        default_root_dir=str(out_dir),
    )

    print("\nОбучение (1 эпоха)...")
    engine.fit(model=model, datamodule=datamodule)

    ckpt_base = out_dir / "Patchcore" / "MVTec" / "bottle"
    versions = sorted([d for d in ckpt_base.glob("v*") if d.is_dir()])
    if not versions:
        raise RuntimeError(f"Нет версий в {ckpt_base}")

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
