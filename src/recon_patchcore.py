"""
Разведка v6: сохраняем mean_normal_features в правильной форме (1536).
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image

_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load

from anomalib.models import Patchcore

CKPT_PATH = Path(
    r"C:\MyPythonProjects\defect-detection\results\Patchcore\MVTec\bottle\v3\weights\lightning\model.ckpt"
)
DATA_ROOT = Path(r"C:\MyPythonProjects\defect-detection\data\MVTecAD\bottle")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_image_as_tensor(path: Path, size=(256, 256)) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main():
    print("=" * 70)
    print("РАЗВЕДКА v6: mean_normal_features (1536)")
    print("=" * 70)

    model = Patchcore.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
    model.eval()
    timm_model = model.model.feature_extractor.feature_extractor

    train_good = DATA_ROOT / "train" / "good"
    imgs = sorted(train_good.glob("*.png"))
    print(f"\nНормальных картинок: {len(imgs)}")

    # Идём по всем 209 нормальным картинкам (это ~5-7 минут на CPU).
    # Можно уменьшить до 100 для ускорения.
    N_USE = len(imgs)

    accumulator = None
    n_done = 0

    with torch.no_grad():
        for i, p in enumerate(imgs[:N_USE]):
            x = load_image_as_tensor(p)
            feats = timm_model(x)  # list: [layer2, layer3]

            # Адаптивный pooling каждой фичи до (1, C, 1, 1), затем squeeze
            layer2 = feats[0]  # (1, 512, 32, 32)
            layer3 = feats[1]  # (1, 1024, 16, 16)

            # Усредняем по spatial-размерности — получаем один вектор (512,) и (1024,)
            v2 = layer2.mean(dim=(2, 3)).squeeze(0)  # (512,)
            v3 = layer3.mean(dim=(2, 3)).squeeze(0)  # (1024,)

            # Конкатенация: (1536,) — совпадает с memory_bank channels
            v = torch.cat([v2, v3])  # (1536,)

            if accumulator is None:
                accumulator = torch.zeros_like(v)
            accumulator += v
            n_done += 1

            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{N_USE}")

    mean_normal = accumulator / n_done
    print(f"\n✅ mean_normal shape: {tuple(mean_normal.shape)}")
    print(f"   norm: {mean_normal.norm().item():.4f}")

    # Сверка с memory_bank
    mb = model.model.memory_bank
    print(f"\nСверка:")
    print(f"   memory_bank:      {tuple(mb.shape)}")
    print(f"   mean_normal:      {tuple(mean_normal.shape)}")
    print(f"   mean memory_bank: {tuple(mb.mean(dim=0).shape)}")

    # Косинусное сходство между mean_normal и средним memory bank
    cos = torch.nn.functional.cosine_similarity(
        mean_normal.unsqueeze(0),
        mb.mean(dim=0).unsqueeze(0),
    ).item()
    print(f"   cosine(mean_normal, mb.mean): {cos:.4f}")
    print(f"   (чем ближе к 1.0, тем правильнее мы собрали mean)")

    # Сохраняем
    out = RESULTS_DIR / "mean_normal_features.pt"
    torch.save(mean_normal, out)
    print(f"\n✅ Сохранено: {out}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
