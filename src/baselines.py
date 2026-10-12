"""
baselines.py — baselines triviales.

Cuesta 10 minutos y sirve de piso: si ResNet saca 0.62 de macro-F1 y la clase
mayoritaria ya saca 0.55 de accuracy, hay que decirlo. Presentar un modelo sin
piso de comparacion es el error clasico que la rubrica castiga.

Se reportan DOS baselines de "clase mayoritaria", porque dan numeros distintos
y las dos lecturas importan:

  - mayoritaria_train: predice siempre la clase mayoritaria del TRAIN ('media'
    aqui, porque el muestreo balanceado deja train casi parejo). Simula un
    baseline real desplegado, que no conoce la distribucion de lo que va a ver.
  - mayoritaria_del_split: predice siempre la clase mayoritaria DEL SPLIT que
    se esta evaluando (en 'local', 'baja', porque el set de Valparaiso quedo
    muy desbalanceado: 63/16/2). Da un accuracy alto (es la mejor respuesta
    posible sin mirar la imagen) pero un macro-F1 bajo — la prueba concreta de
    por que macro-F1 es la metrica principal y accuracy "engaña" en datos
    desbalanceados.

Uso:
    python src/baselines.py --index data/index.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASS_NAMES = ["baja", "media", "alta"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    train = df[df["split"] == "train"]
    rng = np.random.default_rng(args.seed)

    majority = train["label"].value_counts().idxmax()
    priors = (train["label"].value_counts(normalize=True)
              .reindex(CLASS_NAMES).fillna(0).values)

    print(f"clase mayoritaria en train: '{majority}'")
    print(f"priors: {dict(zip(CLASS_NAMES, priors.round(3)))}\n")

    rows = []
    for split in ["val", "test", "local"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        y = [CLASS_NAMES.index(v) for v in sub["label"]]

        p_maj = [CLASS_NAMES.index(majority)] * len(sub)
        p_rnd = rng.choice(len(CLASS_NAMES), size=len(sub), p=priors)

        majority_split = sub["label"].value_counts().idxmax()
        p_maj_split = [CLASS_NAMES.index(majority_split)] * len(sub)

        rows.append({
            "split": split, "n": len(sub),
            "mayoritaria_train_clase": majority,
            "mayoritaria_train_acc": np.mean(np.array(y) == np.array(p_maj)),
            "mayoritaria_train_macroF1": f1_score(y, p_maj, average="macro", zero_division=0),
            "mayoritaria_del_split_clase": majority_split,
            "mayoritaria_del_split_acc": np.mean(np.array(y) == np.array(p_maj_split)),
            "mayoritaria_del_split_macroF1": f1_score(y, p_maj_split, average="macro", zero_division=0),
            "aleatorio_priors_acc": np.mean(np.array(y) == p_rnd),
            "aleatorio_priors_macroF1": f1_score(y, p_rnd, average="macro", zero_division=0),
        })

    out = pd.DataFrame(rows).round(4)
    print("PISO DE COMPARACION (va en la lamina de resultados):")
    print(out.to_string(index=False))
    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/baselines_triviales.csv", index=False)


if __name__ == "__main__":
    main()
