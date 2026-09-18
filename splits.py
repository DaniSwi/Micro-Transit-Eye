"""
splits.py — agrega la columna `split` a data/index.csv

Reglas:
  1. 70 / 15 / 15 con semilla fija.
  2. Estratificado por (fuente, clase): si no, una fuente entera puede caer
     en test y las metricas se vuelven ruido.
  3. Agrupado por `scene_id`: frames de la misma escena nunca se reparten entre
     train y test. Sin esto hay fuga y las metricas salen infladas.
  4. `local_valpo` va SIEMPRE completo a split='local'. Nunca a train ni val.

Uso:
    python src/splits.py --index data/index.csv --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""

    # El conjunto local es holdout puro: es nuestro test de brecha de dominio.
    df.loc[df["source"] == "local_valpo", "split"] = "local"

    pool = df[df["source"] != "local_valpo"]

    for (source, label), group in pool.groupby(["source", "label"]):
        # Repartimos ESCENAS, no imagenes, para evitar fuga.
        scenes = group["scene_id"].unique()
        rng.shuffle(scenes)
        n = len(scenes)
        n_train = int(round(n * RATIOS["train"]))
        n_val = int(round(n * RATIOS["val"]))

        buckets = {
            "train": scenes[:n_train],
            "val": scenes[n_train:n_train + n_val],
            "test": scenes[n_train + n_val:],
        }
        for split_name, scene_ids in buckets.items():
            df.loc[df["scene_id"].isin(scene_ids), "split"] = split_name

    return df


def sanity_checks(df: pd.DataFrame) -> None:
    """Si alguno de estos falla, las metricas no valen. Correr SIEMPRE."""
    assert (df["split"] != "").all(), "hay filas sin split asignado"

    # Ninguna escena en dos splits distintos.
    per_scene = df.groupby("scene_id")["split"].nunique()
    leaked = per_scene[per_scene > 1]
    assert leaked.empty, f"FUGA: escenas en varios splits: {list(leaked.index)[:5]}"

    # El conjunto local no contamina el entrenamiento.
    assert set(df.loc[df["source"] == "local_valpo", "split"]) <= {"local"}, \
        "local_valpo se filtro a train/val/test"

    # Cada split tiene las tres clases.
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        if len(sub) and sub["label"].nunique() < 3:
            print(f"AVISO: split '{split_name}' no tiene las 3 clases")

    print("Chequeos OK: sin fuga entre splits, holdout local intacto.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    df = assign_splits(df, args.seed)
    sanity_checks(df)
    df.to_csv(args.index, index=False)

    print(f"\nsemilla = {args.seed}")
    print("\nTABLA PARA LA LAMINA DE SPLITS:")
    print(pd.crosstab(df["split"], df["label"], margins=True))
    print("\nPor fuente:")
    print(pd.crosstab(df["split"], df["source"]))


if __name__ == "__main__":
    main()
