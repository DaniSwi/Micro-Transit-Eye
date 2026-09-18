"""
evaluate.py — metricas y figuras para la presentacion.

Produce, en outputs/<run>/eval_<split>/:
    metrics.json        macro-F1, accuracy, recall por clase, tiempo de inferencia
    confusion_matrix.png
    per_source.csv      macro-F1 desglosado por fuente  <- esto detecta el riesgo 1
    predictions.csv     para el analisis de errores

Uso:
    python src/evaluate.py --run outputs/resnet50 --split test
    python src/evaluate.py --run outputs/resnet50 --split local
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)

from dataset import CLASS_NAMES, ParaderoDataset


def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "local"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.run / "best.pt", map_location="cpu")
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(cfg["model"]["name"], pretrained=False,
                              num_classes=cfg["data"]["num_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = ParaderoDataset(cfg["data"]["index"], args.split,
                         cfg["data"]["image_size"], train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

    preds, targets, t_total = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            t0 = time.time()
            out = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t_total += time.time() - t0
            preds.append(out.argmax(1).cpu())
            targets.append(y)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    out_dir = args.run / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(targets, preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "run": cfg["run_name"],
        "split": args.split,
        "n_images": len(ds),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "accuracy": float((preds == targets).mean()),
        "recall_por_clase": {c: report[c]["recall"] for c in CLASS_NAMES},
        "f1_por_clase": {c: report[c]["f1-score"] for c in CLASS_NAMES},
        "ms_por_imagen": round(1000 * t_total / max(1, len(ds)), 2),
        "n_params_millions": round(ckpt["n_params"] / 1e6, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    cm = confusion_matrix(targets, preds, labels=range(len(CLASS_NAMES)))
    plot_confusion(cm, out_dir / "confusion_matrix.png",
                   f"{cfg['run_name']} — {args.split}")

    df = ds.df.copy()
    df["y_true"] = [CLASS_NAMES[i] for i in targets]
    df["y_pred"] = [CLASS_NAMES[i] for i in preds]
    df["correcto"] = df["y_true"] == df["y_pred"]
    df.to_csv(out_dir / "predictions.csv", index=False)

    # Desglose por fuente: si una fuente tiene macro-F1 muy alto y otra muy bajo,
    # el modelo puede estar aprendiendo el "estilo del dataset". Ver docs/RIESGOS.md.
    rows = []
    for source, g in df.groupby("source"):
        yt = [CLASS_NAMES.index(v) for v in g["y_true"]]
        yp = [CLASS_NAMES.index(v) for v in g["y_pred"]]
        rows.append({"source": source, "n": len(g),
                     "accuracy": float(np.mean(np.array(yt) == np.array(yp))),
                     "macro_f1": f1_score(yt, yp, average="macro", zero_division=0)})
    pd.DataFrame(rows).to_csv(out_dir / "per_source.csv", index=False)

    # Errores entre clases adyacentes (baja<->media, media<->alta) vs saltos de 2.
    # Si casi todo el error es adyacente, el problema son los umbrales, no el modelo.
    err = df[~df["correcto"]]
    idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    adj = sum(abs(idx[r.y_true] - idx[r.y_pred]) == 1 for r in err.itertuples())
    print(f"\n=== {cfg['run_name']} — split '{args.split}' ===")
    print(f"macro-F1 = {metrics['macro_f1']:.4f}   accuracy = {metrics['accuracy']:.4f}")
    print(f"inferencia = {metrics['ms_por_imagen']} ms/imagen   "
          f"({metrics['n_params_millions']}M parametros)")
    if len(err):
        print(f"errores adyacentes: {adj}/{len(err)} ({adj/len(err):.0%}) "
              "— si es alto, el problema son los umbrales")
    print(f"\nfiguras y CSVs en {out_dir}")


if __name__ == "__main__":
    main()
