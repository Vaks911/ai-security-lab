"""
Защита от adversarial attack через JPEG-компрессию.

Идея: adversarial-шум — это высокочастотные колебания пикселей.
JPEG-сжатие отбрасывает часть высокочастотной информации и стирает шум.
Глаз разницы не видит, а атака перестаёт работать.
"""

from io import BytesIO
from pathlib import Path
from PIL import Image


def jpeg_defense(img: Image.Image, quality: int = 75) -> Image.Image:
    """
    Прогоняет картинку через JPEG с указанным quality.

    Args:
        img: PIL Image (RGB)
        quality: 1-100. Ниже — сильнее сжатие, меньше деталей.

    Returns:
        PIL Image — сжатая и восстановленная.
    """
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def jpeg_defense_from_path(path: Path, quality: int = 75) -> Image.Image:
    """То же самое, но принимает путь к файлу."""
    img = Image.open(path).convert("RGB")
    return jpeg_defense(img, quality=quality)


if __name__ == "__main__":
    # Быстрая проверка: сжимаем пример и смотрим разницу
    import numpy as np

    test_path = Path(
        r"C:\MyPythonProjects\ai-security-lab\results\adversarial_border\contamination_019_eps0.10.png"
    )

    if not test_path.exists():
        print(f"Нет файла: {test_path}")
        raise SystemExit(1)

    orig = Image.open(test_path).convert("RGB")
    compressed = jpeg_defense(orig, quality=75)

    arr_o = np.asarray(orig, dtype=np.float32)
    arr_c = np.asarray(compressed, dtype=np.float32)
    diff = np.abs(arr_o - arr_c)

    print(f"Оригинал:   shape={orig.size}")
    print(f"Сжатие Q75: shape={compressed.size}")
    print(f"Разница:    max={diff.max():.1f}, mean={diff.mean():.3f}")

    # Сохраняем рядом для визуальной проверки
    out = test_path.parent / "contamination_019_eps0.10_jpeg75.png"
    compressed.save(out)
    print(f"Сохранено: {out}")
