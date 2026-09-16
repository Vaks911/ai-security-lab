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
- `notebooks/` — эксперименты
- `src/` — чистый код
- `results/` — метрики и графики
- `reports/` — итоговый отчёт
- `notes.md` — рабочий дневник
## Baseline

| Метрика | Значение |
|---|---|
| image_F1 | **0.9920** |
| accuracy | 0.9880 |
| порог модели | между 0.49999 и 0.53993 |
| слабый класс | `contamination` (recall 0.952) |

Полные данные: `results/baseline_metrics.json`, `results/baseline_per_image.csv`.
Дневник эксперимента: `notes.md`.
## Data Poisoning

Атака на память PatchCore через отравление `train/good/`.

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

График: `results/poison_plot.png`. Дневник: `notes.md`.
## Автор

Максим Нагайцев — [GitHub](https://github.com/Vaks911) · [LinkedIn](https://www.linkedin.com/in/maksim-nagaytsev-ab2311432)