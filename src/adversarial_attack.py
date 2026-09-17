"""
Adversarial Attack на ПОГРАНИЧНЫХ дефектах (score близок к порогу).

Стратегия: атакуем самые уязвимые дефекты — те, что модель ловит "еле-еле".
Усиленная PGD: 100 шагов, eps = 0.05 / 0.08 / 0.10.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import json
import csv

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
DATA_ROOT = Path(r"C:\MyPythonProjects\defect-detection\data\MVTecAD\bottle")
LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"
OUT_DIR = RESULTS_DIR / "adversarial_border"

IMAGE_SIZE = (256, 256)

# ПОГРАНИЧНЫЕ дефекты: те, у которых baseline score близок к порогу (~0.54)
TARGETS = [
    ("contamination", "019.png"),  # 0.5399
    ("contamination", "016.png"),  # 0.5435
    ("contamination", "020.png"),  # 0.5549
    ("contamination", "012.png"),  # 0.5572
    ("broken_small", "000.png"),  # 0.6009
    ("broken_small", "014.png"),  # 0.6417
    ("broken_small", "005.png"),  # 0.6708
    ("broken_large", "011.png"),  # 0.6895
    ("broken_large", "005.png"),  # 0.7374
]

# Усиленные параметры
EPSILONS = [0.05, 0.08, 0.10]
PGD_STEPS = 100
PGD_ALPHA_RATIO = 1.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
X_MIN = torch.tensor(((0.0 - IMAGENET_MEAN) / IMAGENET_STD)).view(1, 3, 1, 1)
X_MAX = torch.tensor(((1.0 - IMAGENET_MEAN) / IMAGENET_STD)).view(1, 3, 1, 1)


def load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def compute_features(timm_model, x):
    feats = timm_model(x)
    v2 = feats[0].mean(dim=(2, 3))
    v3 = feats[1].mean(dim=(2, 3))
    return torch.cat([v2, v3], dim=1)


def pgd_attack(timm_model, x_orig, mean_normal, eps, n_steps, alpha_ratio):
    alpha = eps / n_steps * alpha_ratio
    x_adv = x_orig.clone().detach()
    x_adv = x_adv.detach().requires_grad_(True)

    for _ in range(n_steps):
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        feats = compute_features(timm_model, x_adv)
        loss = F.mse_loss(feats, mean_normal)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv - alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.max(torch.min(x_adv, X_MAX), X_MIN)
            x_adv = x_adv.detach().requires_grad_(True)
    return x_adv.detach()


def predict_score(engine, model, ckpt_path, img_path):
    dataset = PredictDataset(path=img_path, image_size=IMAGE_SIZE)
    preds = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt_path)
    if not preds:
        return None, None
    p = preds[0]
    return float(p["pred_scores"].item()), (
        "DEFECT" if int(p["pred_labels"].item()) == 1 else "NORMAL"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Adversarial Attack на ПОГРАНИЧНЫХ дефектах")
    print(f"PGD: {PGD_STEPS} шагов, eps ∈ {EPSILONS}")
    print("=" * 72)

    model = Patchcore.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
    model.eval()
    timm_model = model.model.feature_extractor.feature_extractor
    for p in timm_model.parameters():
        p.requires_grad = False

    engine = Engine(accelerator="cpu", devices=1, logger=False)
    mean_normal = torch.load(
        RESULTS_DIR / "mean_normal_features.pt", weights_only=False
    ).unsqueeze(0)

    rows = []

    for cls, fname in TARGETS:
        img_path = DATA_ROOT / "test" / cls / fname
        if not img_path.exists():
            print(f"⚠️  нет {img_path}")
            continue

        x_orig = load_image_as_tensor(img_path)
        orig_score, orig_label = predict_score(engine, model, CKPT_PATH, img_path)

        print(f"\n{cls}/{fname}")
        print(f"  orig: score={orig_score:.4f}  {orig_label}")

        stem = f"{cls}_{fname.replace('.png', '')}"
        tensor_to_pil(x_orig).save(OUT_DIR / f"{stem}_orig.png")

        for eps in EPSILONS:
            x_adv = pgd_attack(
                timm_model, x_orig, mean_normal, eps, PGD_STEPS, PGD_ALPHA_RATIO
            )
            adv_path = OUT_DIR / f"{stem}_eps{eps:.2f}.png"
            tensor_to_pil(x_adv).save(adv_path)

            # Шум × 10
            diff = (x_adv - x_orig).abs()
            Image.fromarray(
                (
                    torch.clamp(diff * 10, 0, 1).squeeze(0).permute(1, 2, 0).numpy()
                    * 255
                ).astype(np.uint8)
            ).save(OUT_DIR / f"{stem}_noise_x10_eps{eps:.2f}.png")

            adv_score, adv_label = predict_score(engine, model, CKPT_PATH, adv_path)
            delta = (adv_score - orig_score) / orig_score * 100 if orig_score > 0 else 0
            flipped = orig_label == "DEFECT" and adv_label == "NORMAL"

            mark = " ✅ FLIP" if flipped else ""
            print(
                f"  eps={eps:.2f}  score={adv_score:.4f}  {adv_label}  ({delta:+.1f}%){mark}"
            )

            rows.append(
                {
                    "class": cls,
                    "file": fname,
                    "eps": eps,
                    "orig_score": round(orig_score, 4),
                    "adv_score": round(adv_score, 4),
                    "adv_label": adv_label,
                    "delta_pct": round(delta, 2),
                    "flipped": flipped,
                }
            )

    # === Сводка ===
    print("\n" + "=" * 72)
    print("СВОДКА")
    print("=" * 72)
    print(
        f"{'class':<15} {'file':<10} {'eps':>6} {'orig':>8} {'adv':>8} {'Δ%':>8}  flip"
    )
    print("-" * 72)
    for r in rows:
        mark = "✅" if r["flipped"] else ""
        print(
            f"{r['class']:<15} {r['file']:<10} {r['eps']:>6.2f} "
            f"{r['orig_score']:>8.4f} {r['adv_score']:>8.4f} {r['delta_pct']:>+7.1f}%  {mark}"
        )

    # По eps
    print("\n" + "=" * 72)
    print("Статистика по eps")
    print("=" * 72)
    for eps in EPSILONS:
        sub = [r for r in rows if r["eps"] == eps]
        n_flip = sum(1 for r in sub if r["flipped"])
        avg_drop = np.mean([r["delta_pct"] for r in sub])
        print(
            f"  eps={eps:.2f}:  flip={n_flip}/{len(sub)}  средн.снижение={avg_drop:+.1f}%"
        )

    total_flip = sum(1 for r in rows if r["flipped"])
    print(f"\n  ВСЕГО flip: {total_flip}/{len(rows)}")

    # Сохранение
    with open(RESULTS_DIR / "adversarial_border.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epsilon_set": EPSILONS,
                "pgd_steps": PGD_STEPS,
                "results": rows,
                "total_flips": total_flip,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        RESULTS_DIR / "adversarial_border.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n✅ Сохранено: {RESULTS_DIR / 'adversarial_border.json'}")
    print(f"✅ Картинки:  {OUT_DIR}")


if __name__ == "__main__":
    main()
