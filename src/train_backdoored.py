"""
Обучение PatchCore на backdoored-датасете.

Датасет лежит в data/backdoored/bottle/ — стандартная структура MVTec.
"""

from pathlib import Path

from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = LAB_ROOT / "data" / "backdoored"
RESULTS_ROOT = LAB_ROOT / "results"

CATEGORY = "bottle"
IMAGE_SIZE = (256, 256)
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.1
NUM_NEIGHBORS = 9

OUT_DIR = RESULTS_ROOT / "backdoored"


def main() -> None:
    print("=" * 60)
    print("Обучение PatchCore на backdoored-датасете")
    print("=" * 60)
    print(f"  Датасет:   {DATA_ROOT}  (категория: {CATEGORY})")
    print(f"  Результат: {OUT_DIR}")

    datamodule = MVTec(
        root=DATA_ROOT,
        category=CATEGORY,
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
        default_root_dir=str(OUT_DIR),
    )

    print("\nОбучение (1 эпоха)...")
    print("-" * 60)
    engine.fit(model=model, datamodule=datamodule)

    ckpt_base = OUT_DIR / "Patchcore" / "MVTec" / CATEGORY
    versions = sorted([d for d in ckpt_base.glob("v*") if d.is_dir()])
    if not versions:
        raise RuntimeError(f"Нет версий в {ckpt_base}")

    ckpt_path = versions[-1] / "weights" / "lightning" / "model.ckpt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Чекпоинт не найден: {ckpt_path}")

    print(f"\n✅ Чекпоинт: {ckpt_path}")


if __name__ == "__main__":
    main()
