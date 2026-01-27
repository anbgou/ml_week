#!/usr/bin/env python3
"""
classify_images.py
------------------
Ultralytics YOLOv8 classification inference utility for your project.

What it does:
- Runs inference on a single image OR a folder of images (optionally recursive)
- Outputs: predicted class + confidence
- Optionally:
  - Saves annotated images (text overlay)
  - Saves predictions to CSV/JSON
  - Computes Accuracy/Precision/Recall/F1 if labels can be inferred from parent folder
    (e.g., .../test/healthy/*.jpg, .../test/sick/*.jpg)

Typical usage:
  python classify_images.py --model runs/classify/baseline_cls/weights/best.pt --source dataset_plants_final/test --imgsz 224
  python classify_images.py --model runs/classify/baseline_cls/weights/best.pt --source dataset_plants_final/test/sick/xxx.jpg --imgsz 224 --save_images
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "Ultralytics is not installed. Install with: pip install ultralytics"
    ) from e

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    raise ImportError(
        "Pillow is required. Install with: pip install pillow"
    ) from e

# sklearn is optional (only for metrics)
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class PredRecord:
    path: str
    pred_name: str
    pred_idx: int
    confidence: float
    positive_class: str
    prob_positive: Optional[float] = None
    true_label: Optional[int] = None  # 0/1 if inferred
    true_name: Optional[str] = None


def _is_image(p: Path, exts: Sequence[str]) -> bool:
    return p.is_file() and p.suffix.lower() in set(x.lower() for x in exts)


def gather_images(source: Path, exts: Sequence[str], recursive: bool) -> List[Path]:
    if source.is_file():
        if _is_image(source, exts):
            return [source]
        raise ValueError(f"Source is a file but not a supported image: {source}")

    if not source.is_dir():
        raise FileNotFoundError(f"Source path not found: {source}")

    if recursive:
        paths = [p for p in source.rglob("*") if _is_image(p, exts)]
    else:
        paths = [p for p in source.glob("*") if _is_image(p, exts)]
    paths.sort()
    return paths


def load_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try a TrueType font; fallback to default.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def annotate_and_save(
    img_path: Path,
    out_path: Path,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Simple readable overlay box
    margin = 6
    x0, y0 = margin, margin
    # textbbox is newer; fallback if missing
    try:
        bbox = draw.textbbox((x0, y0), text, font=font)
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
    except Exception:
        box_w, box_h = draw.textsize(text, font=font)

    draw.rectangle([x0 - 4, y0 - 4, x0 + box_w + 8, y0 + box_h + 8], fill=(0, 0, 0))
    draw.text((x0, y0), text, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def infer_true_label_from_parent(img_path: Path, healthy_name: str = "healthy", sick_name: str = "sick") -> Optional[int]:
    """
    If path contains '/healthy/' => 0, '/sick/' => 1.
    Works with any nesting depth, e.g. .../test/healthy/xxx.jpg
    """
    parts = [p.lower() for p in img_path.parts]
    if healthy_name.lower() in parts:
        return 0
    if sick_name.lower() in parts:
        return 1
    return None


def chunked(seq: Sequence[Path], batch_size: int) -> List[List[Path]]:
    if batch_size <= 0:
        batch_size = 1
    return [list(seq[i:i + batch_size]) for i in range(0, len(seq), batch_size)]


def names_mapping(model) -> Dict[int, str]:
    # Ultralytics can give dict or list
    n = model.names
    if isinstance(n, dict):
        return {int(k): str(v) for k, v in n.items()}
    return {i: str(v) for i, v in enumerate(n)}


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if not SKLEARN_OK:
        return {}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 classification inference for images/folders.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 classification .pt (e.g., runs/classify/xxx/weights/best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image path OR directory path")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size for inference (default: 224)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for inference (default: 32)")
    parser.add_argument("--device", type=str, default="", help="Device string for Ultralytics (e.g. '0' for GPU0, 'cpu'). Empty = auto.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subfolders when source is a directory")
    parser.add_argument("--exts", type=str, default=",".join(DEFAULT_EXTS), help="Comma-separated image extensions")

    parser.add_argument("--positive_class", type=str, default="sick", help="Class name treated as positive (default: sick)")
    parser.add_argument("--save_dir", type=str, default="results/classify_preds", help="Directory to save outputs (default: results/classify_preds)")
    parser.add_argument("--save_images", action="store_true", help="Save annotated images with predicted label/confidence")
    parser.add_argument("--csv", type=str, default="results/predictions.csv", help="CSV output path (default: results/predictions.csv)")
    parser.add_argument("--json", type=str, default="results/predictions.json", help="JSON output path (default: results/predictions.json)")

    parser.add_argument("--infer_labels", action="store_true",
                        help="Infer true labels from parent folder names (healthy/sick) and compute metrics (if sklearn installed).")

    args = parser.parse_args()

    model_path = Path(args.model)
    source = Path(args.source)
    save_dir = Path(args.save_dir)
    exts = tuple(x.strip() for x in args.exts.split(",") if x.strip())

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    img_paths = gather_images(source, exts=exts, recursive=args.recursive)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in: {source.resolve()} (exts={exts})")

    print(f"Model:  {model_path}")
    print(f"Source: {source}")
    print(f"Images: {len(img_paths)}")
    print(f"imgsz={args.imgsz} batch={args.batch} device={args.device or 'auto'}")

    yolo = YOLO(str(model_path))
    names = names_mapping(yolo)
    inv_names = {v: k for k, v in names.items()}

    pos_name = args.positive_class
    if pos_name not in inv_names:
        # try case-insensitive
        inv_lower = {k.lower(): v for k, v in inv_names.items()}
        if pos_name.lower() in inv_lower:
            pos_name = pos_name.lower()
            pos_idx = inv_lower[pos_name]
            # restore original class name for record if possible
            pos_name = names[pos_idx]
        else:
            print("⚠️ positive_class not found in model.names. Available:", list(inv_names.keys()))
            pos_idx = None
    else:
        pos_idx = inv_names[pos_name]

    font = load_font(18)

    records: List[PredRecord] = []

    # Batched inference
    batches = chunked(img_paths, args.batch)
    for b in batches:
        results = yolo.predict(
            source=[str(p) for p in b],
            imgsz=args.imgsz,
            batch=len(b),
            device=args.device,
            verbose=False,
        )
        # results is list aligned with input order
        for img_p, r in zip(b, results):
            # robust extraction
            probs = r.probs
            top_idx = int(probs.top1)
            top_conf = float(probs.top1conf)
            pred_name = names.get(top_idx, str(top_idx))

            prob_pos = None
            if pos_idx is not None:
                try:
                    prob_pos = float(probs.data[int(pos_idx)].detach().cpu().numpy())
                except Exception:
                    prob_pos = None

            true_label = None
            true_name = None
            if args.infer_labels:
                tl = infer_true_label_from_parent(img_p, healthy_name="healthy", sick_name="sick")
                if tl is not None:
                    true_label = int(tl)
                    true_name = "sick" if true_label == 1 else "healthy"

            rec = PredRecord(
                path=str(img_p),
                pred_name=pred_name,
                pred_idx=top_idx,
                confidence=top_conf,
                positive_class=pos_name,
                prob_positive=prob_pos,
                true_label=true_label,
                true_name=true_name,
            )
            records.append(rec)

            if args.save_images:
                rel = img_p.name
                # if source is directory, preserve relative structure under save_dir
                if source.is_dir():
                    try:
                        rel = str(img_p.relative_to(source))
                    except Exception:
                        rel = img_p.name
                out_img = save_dir / rel
                text = f"pred: {pred_name} | conf: {top_conf:.3f}"
                if prob_pos is not None and pos_idx is not None:
                    text += f" | p({names[pos_idx]}): {prob_pos:.3f}"
                annotate_and_save(img_p, out_img, text, font=font)

    # Save CSV/JSON
    out_csv = Path(args.csv)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in records]

    # CSV without pandas dependency
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved predictions CSV:  {out_csv.resolve()}")
    print(f"✅ Saved predictions JSON: {out_json.resolve()}")
    if args.save_images:
        print(f"✅ Saved annotated images in: {save_dir.resolve()}")

    # Summary
    from collections import Counter
    pred_counts = Counter([r.pred_name for r in records])
    print("\nPrediction counts:")
    for k, v in pred_counts.most_common():
        print(f"  {k}: {v}")

    # Metrics if labels inferred and available
    if args.infer_labels and SKLEARN_OK:
        y_true = np.array([r.true_label for r in records if r.true_label is not None], dtype=int)
        y_pred_bin = None

        if len(y_true) == len(records):
            # Convert predicted name -> binary by positive_class name
            pos_cls = pos_name
            y_pred_bin = np.array([1 if r.pred_name == pos_cls else 0 for r in records], dtype=int)

            m = compute_binary_metrics(y_true, y_pred_bin)
            print("\nMetrics (binary, positive=sick):")
            for k, v in m.items():
                print(f"  {k}: {v:.4f}")

            cm = confusion_matrix(y_true, y_pred_bin)
            print("\nConfusion matrix [[TN, FP],[FN, TP]]:")
            print(cm)

            # Save metrics json
            metrics_path = out_csv.parent / "inference_metrics.json"
            payload = {
                "positive_class": pos_cls,
                "num_images": int(len(records)),
                **m,
                "confusion_matrix": cm.tolist(),
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved inference metrics: {metrics_path.resolve()}")
        else:
            print("\n⚠️ Could not compute metrics: not all images have inferable true labels.")
    elif args.infer_labels and not SKLEARN_OK:
        print("\n⚠️ sklearn not available: install scikit-learn to compute metrics.")


if __name__ == "__main__":
    main()
