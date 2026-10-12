"""
baseline_yolo.py — baseline "con vision" sin entrenar nada propio.

A diferencia de la clase mayoritaria (baselines.py), este SI mira la imagen:
usa un detector de personas preentrenado (YOLOv8, pesos COCO) para contar
cajas de clase "person" y aplica los mismos umbrales que el resto del
proyecto (BINS en labels.py). Si ResNet/ViT no superan esto con claridad, es
señal de que el entrenamiento no esta aportando mucho sobre un detector
generico que nunca vio un paradero de Valparaiso.

Requiere: pip install ultralytics (no esta en requirements.txt porque solo
este script la usa, y descarga ~6 MB de pesos la primera vez que corre).

Uso:
    python src/baseline_yolo.py --index data/index.csv --split test
    python src/baseline_yolo.py --index data/index.csv --split local
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from labels import count_to_label

CLASS_NAMES = ["baja", "media", "alta"]
PERSON_CLASS_ID = 0  # clase 0 en COCO/YOLO es "person"


def count_people(model, image_path: str, conf: float) -> int:
    results = model(image_path, verbose=False, conf=conf)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0
    return int((boxes.cls == PERSON_CLASS_ID).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--split", default="test", choices=["val", "test", "local"])
    ap.add_argument("--model", default="yolov8n.pt",
                    help="pesos preentrenados de ultralytics (se descargan solos la primera vez)")
    ap.add_argument("--conf", type=float, default=0.25, help="umbral de confianza de deteccion")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    df = pd.read_csv(args.index)
    sub = df[df["split"] == args.split].reset_index(drop=True)
    if len(sub) == 0:
        raise SystemExit(f"split '{args.split}' vacio en {args.index}")

    pred_counts, pred_labels = [], []
    for path in tqdm(sub["path"], desc=f"YOLO sobre '{args.split}'"):
        n = count_people(model, path, args.conf)
        pred_counts.append(n)
        pred_labels.append(count_to_label(n))

    sub = sub.copy()
    sub["yolo_person_count"] = pred_counts
    sub["yolo_label"] = pred_labels
    sub["correcto"] = sub["yolo_label"] == sub["label"]

    y_true = [CLASS_NAMES.index(v) for v in sub["label"]]
    y_pred = [CLASS_NAMES.index(v) for v in sub["yolo_label"]]

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))

    print(f"\n=== baseline YOLO ({args.model}) — split '{args.split}' ===")
    print(f"n = {len(sub)}")
    print(f"macro-F1 = {macro_f1:.4f}   accuracy = {accuracy:.4f}")

    out_dir = Path("outputs") / "baseline_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / f"predictions_{args.split}.csv", index=False)

    # Un solo CSV acumulado con todos los splits corridos hasta ahora, para
    # comparar directo contra baselines_triviales.csv y los macro-F1 de los modelos.
    summary_path = out_dir / "summary.csv"
    row = {"split": args.split, "n": len(sub), "macro_f1": round(macro_f1, 4),
           "accuracy": round(accuracy, 4), "model": args.model}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["split"] != args.split]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    summary.to_csv(summary_path, index=False)
    print(f"\nresultados en {out_dir}")


if __name__ == "__main__":
    main()
