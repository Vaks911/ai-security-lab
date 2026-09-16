# ai-security-lab

Лаборатория атак на memory-bank anomaly detector (PatchCore).

## Что это

Исследование устойчивости PatchCore (WideResNet50 + memory bank)
к атакам на данные и на входное изображение.

**Модель-жертва:** [defect-detection](https://github.com/Vaks911/defect-detection)

## Статус

- [x] Baseline зафиксирован
- [x] Data Poisoning
- [ ] Adversarial Attack
- [ ] Митигации

## Структура

- `baseline/` — скрипты замера исходных метрик
- `src/` — скрипты атак и оценки
- `results/` — метрики, графики, чекпоинты
- `reports/` — итоговый отчёт
- `notes.md` — рабочий дневник

---

## Baseline

Метрики чистой модели PatchCore на MVTec AD (bottle), 83 тестовых изображения.

| Метрика | Значение |
|---|---|
| image_F1 | **0.9920** |
| accuracy | 0.9880 |
| порог модели | между 0.49999 и 0.53993 |
| слабый класс | `contamination` (recall 0.952) |

Полные данные: `results/baseline_metrics.json`, `results/baseline_per_image.csv`.  
Дневник эксперимента: `notes.md`.

---

## Data Poisoning

Атака на память PatchCore через отравление `train/good/` — дефектные
изображения помечаются как «норма» и попадают в memory bank.

| Run | F1 | Recall | Precision | FN | FP |
|---|---:|---:|---:|---:|---:|
| baseline | 0.9920 | 0.9841 | 1.0000 | 1 | 0 |
| 1% | 0.9756 | 0.9524 | 1.0000 | 3 | 0 |
| 5% | 0.9043 | 0.8254 | 1.0000 | 11 | 0 |
| 10% | 0.8630 | 1.0000 | 0.7590 | 0 | 20 |
| 20% | 0.8630 | 1.0000 | 0.7590 | 0 | 20 |

**Два режима отказа:**

- **1–5% — «ослепление»:** модель пропускает дефекты (FN растёт), ложных тревог нет. Опаснее для производства.
- **10–20% — «паранойя»:** модель ловит всё, включая норму (FP = 20). Опаснее для экономики.

![Data Poisoning — два режима отказа](results/poison_plot.png)

*Слева: F1/Recall/Precision — переход от «ослепления» к «паранойе» на 10%.  
Справа: FN vs FP — визуальная смена режима отказа. Дневник: `notes.md`.*

**Ключевой вывод:** даже 1% отравления даёт заметный эффект — F1 падает на 1.6%.
Тонкая атака (1–5%) опаснее грубой, потому что модель пропускает дефекты
тихо, без ложных тревог.

---

## 🛠 Технологии

- **Python 3.11**, PyTorch 2.x (CPU)
- **Anomalib 1.1.0** — фреймворк anomaly detection
- **PatchCore** (CVPR 2022) — алгоритм memory bank
- **WideResNet50** — backbone
- **NumPy, Pillow, Matplotlib** — обработка и визуализация

## 🚀 Воспроизведение

### 1. Окружение

```bash
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt