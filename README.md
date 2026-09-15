# ai-security-lab

Лаборатория атак на memory-bank anomaly detector (PatchCore).

## Что это

Исследование устойчивости PatchCore (WideResNet50 + memory bank)
к атакам на данные и на входное изображение.

**Модель-жертва:** [defect-detection](https://github.com/Vaks911/defect-detection)

## Статус

- [ ] Baseline зафиксирован
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

## Автор

Максим Нагайцев — [GitHub](https://github.com/Vaks911) · [LinkedIn](https://www.linkedin.com/in/maksim-nagaytsev-ab2311432)