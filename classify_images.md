# classify_images.py — usage

Це утиліта для інференсу **YOLOv8 classification**-моделі (Ultralytics) для задачі:

- **healthy vs sick**
- `imgsz=224`

Скрипт:
- приймає **одне зображення** або **папку**
- повертає `pred label + confidence`
- зберігає `results/predictions.csv` та `results/predictions.json`
- за бажанням зберігає **анотовані зображення**
- може порахувати **Accuracy/Precision/Recall/F1**, якщо істинні мітки можна вивести з папок `healthy/` і `sick/`

---

## 1) Встановлення залежностей

```bash
pip install ultralytics pillow numpy
pip install scikit-learn  # якщо хочеш метрики Accuracy/Precision/Recall/F1
```

---

## 2) Базове використання

### 2.1 Один файл
```bash
python classify_images.py \
  --model runs/classify/baseline_cls/weights/best.pt \
  --source dataset_plants_final/test/sick/your_image.jpg \
  --imgsz 224
```

### 2.2 Папка (тільки верхній рівень)
```bash
python classify_images.py \
  --model runs/classify/baseline_cls/weights/best.pt \
  --source dataset_plants_final/test \
  --imgsz 224
```

### 2.3 Папка рекурсивно
```bash
python classify_images.py \
  --model runs/classify/baseline_cls/weights/best.pt \
  --source dataset_plants_final/test \
  --recursive \
  --imgsz 224
```

---

## 3) Збереження анотованих зображень

```bash
python classify_images.py \
  --model runs/classify/baseline_cls/weights/best.pt \
  --source dataset_plants_final/test \
  --imgsz 224 \
  --save_images \
  --save_dir results/classify_preds
```

Результати будуть у:
- `results/classify_preds/...` (картинки з накладеним текстом)

---

## 4) Метрики (Accuracy/Precision/Recall/F1)

Скрипт може **симулювати GT** за назвою папки:
- якщо шлях містить `/healthy/` → true=0
- якщо шлях містить `/sick/` → true=1

```bash
python classify_images.py \
  --model runs/classify/baseline_cls/weights/best.pt \
  --source dataset_plants_final/test \
  --imgsz 224 \
  --infer_labels
```

Він виведе метрики в консоль і збереже:
- `results/inference_metrics.json`

---

## 5) Корисні параметри

- `--batch 32` — швидше на GPU
- `--device cpu` або `--device 0`
- `--positive_class sick` — що вважати “позитивним” класом

---

## 6) Типова інтеграція в ваш проєкт

Поклади файли в корінь:
```
Ваш_Проект/
├── classify_images.py
├── classify_images.md
├── dataset_plants_final/
└── runs/classify/...
```

і запускай команди з кореня проєкту.
