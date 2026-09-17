# ai-security-lab

Лаборатория атак на memory-bank anomaly detector (PatchCore).

## Что это

Исследование устойчивости PatchCore (WideResNet50 + memory bank)
к атакам на данные и на входное изображение.

**Модель-жертва:** [defect-detection](https://github.com/Vaks911/defect-detection)

## Статус

- [x] Baseline зафиксирован
- [x] Data Poisoning
- [x] Adversarial Attack
- [x] Митигации (JPEG-защита)

## Структура

- `baseline/` — скрипты замера исходных метрик
- `src/` — скрипты атак, защиты и оценки
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

## Adversarial Attack

Атака в feature-space backbone. Цель — сдвинуть фичи дефекта к среднему нормы,
чтобы модель назвала его NORMAL.

**Метод:** PGD (100 шагов, eps ∈ {0.05, 0.08, 0.10}) через `FeatureListNet`
(внутренняя timm-модель PatchCore). Loss = MSE(features, mean_normal_features).

**Результат:** 12/27 flip (**44.4%**). Все 4 `contamination` переключились при eps=0.05.

| Класс | Средний orig score | Flip rate |
|---|---:|---:|
| contamination | 0.50–0.56 | **12/12 (100%)** |
| broken_small | 0.60–0.67 | 0/9 (0%) |
| broken_large | 0.69–0.74 | 0/6 (0%) |

![Adversarial Attack — три панели](results/adversarial_plot.png)

*Слева: flip rate по классам. В центре: снижение score по eps, насыщение после 0.05.  
Справа: scatter orig vs adv, зелёная зона — область flip.*

**Ключевой вывод:** атака эффективна **только у границы решения**. Если модель
уверена в дефекте (score > 0.6) — adversarial-шум не помогает. Если score близок
к порогу — снижения хватает для flip. Визуально adversarial-картинка неотличима
от оригинала.

Полные данные: `results/adversarial_border.json`, `results/adversarial_border.csv`.  
Визуализация: `results/adversarial_border/`.

---

## Митигации

Защита от adversarial attack через JPEG-компрессию входа.

**Идея:** adversarial-шум — это высокочастотные колебания пикселей. JPEG
отбрасывает часть высокочастотной информации, шум стирается. Глаз разницы
не видит (mean diff = 2.1 из 255), а атака перестаёт работать.

**Метод:** перед подачей в модель картинка прогоняется через JPEG с качеством 75.

| | Flip rate |
|---|---:|
| Без защиты | **12/27 (44.4%)** |
| С защитой (JPEG Q75) | **0/27 (0.0%)** |

Защита обнулила все 12 успешных атак. Ни одна не сработала повторно.

![Защита — flip rate до/после](results/defense_plot.png)

*Слева: flip rate по классам до/после. Справа: score flip-картинок — все поднялись выше порога 0.54.*

**Ограничения:**

- Защита эффективна против PGD через пиксели. Более сильная атака с учётом JPEG (EOT) может её обойти.
- Adversarial Training даёт более надёжную защиту, но требует переобучения модели.
- JPEG Q75 замедляет инференс на ~5–10 мс на картинку — для realtime-пайплайна это надо учитывать.

Полные данные: `results/defense_results.json`.  
Скрипты: `src/defense.py`, `src/evaluate_defense.py`, `src/plot_defense.py`.

---

## 🛠 Технологии

- **Python 3.11**, PyTorch 2.x (CPU)
- **Anomalib 1.1.0** — фреймворк anomaly detection
- **PatchCore** (CVPR 2022) — алгоритм memory bank
- **WideResNet50** — backbone
- **NumPy, Pillow, Matplotlib** — обработка и визуализация

## 🚀 Воспроизведение

### 1. Окружение

```
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

### 2. Baseline

```
python baseline\baseline.py
```

### 3. Data Poisoning

```
python src\poison.py
python src\train_poisoned.py
python src\evaluate_all.py
python src\plot_poison.py
```

### 4. Adversarial Attack

```
python src\recon_patchcore.py
python src\adversarial_attack.py
python src\plot_adversarial.py
```

### 5. Митигации (JPEG-защита)

```
python src\defense.py
python src\evaluate_defense.py
python src\plot_defense.py
```

---

## 📁 Структура проекта

```
ai-security-lab/
├── baseline/
│   └── baseline.py                # Замер метрик с EVAL_NAME
├── src/
│   ├── poison.py                  # Создание отравленных датасетов
│   ├── train_poisoned.py          # Обучение на отравленных данных
│   ├── evaluate_all.py            # Прогон baseline.py по всем уровням
│   ├── plot_poison.py             # График Data Poisoning
│   ├── recon_patchcore.py         # Разведка структуры PatchCore
│   ├── adversarial_attack.py      # PGD feature-space атака
│   ├── plot_adversarial.py        # График Adversarial Attack
│   ├── defense.py                 # JPEG-защита
│   ├── evaluate_defense.py        # Оценка защиты
│   └── plot_defense.py            # График до/после защиты
├── results/
│   ├── baseline_metrics.json
│   ├── poison_summary.json
│   ├── poison_plot.png
│   ├── adversarial_border.json
│   ├── adversarial_plot.png
│   ├── mean_normal_features.pt
│   ├── adversarial_border/        # Визуализация: orig, adv, noise ×10
│   ├── defense_results.json
│   └── defense_plot.png
├── notes.md                       # Рабочий дневник
├── requirements.txt
└── README.md
```

---

## 🎯 Дальше

- [ ] **Adversarial Patch:** наклейка на бутылку, которая «гасит» детекцию.
- [ ] **Black-box атака:** без доступа к градиентам (transfer attack).
- [ ] **Adversarial Training:** более надёжная защита, чем JPEG, но требует переобучения.

---

## Автор

Максим Нагайцев — [GitHub](https://github.com/Vaks911) · [LinkedIn](https://www.linkedin.com/in/maksim-nagaytsev-ab2311432)