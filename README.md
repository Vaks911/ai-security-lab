# ai-security-lab

Лаборатория атак на memory-bank anomaly detector (PatchCore).

## Что это

Исследование устойчивости PatchCore (WideResNet50 + memory bank)
к атакам на данные и на входное изображение.

**Модель-жертва:** [defect-detection](https://github.com/Vaks911/defect-detection)

## Статус

- [x] Baseline зафиксирован
- [ ] Data Poisoning
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
## Автор

Максим Нагайцев — [GitHub](https://github.com/Vaks911) · [LinkedIn](https://www.linkedin.com/in/maksim-nagaytsev-ab2311432)