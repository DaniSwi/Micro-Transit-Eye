"""
labels.py — construye data/index.csv

Idea central: NO etiquetamos a mano. Cada dataset ya trae anotaciones de personas
(cajas o puntos). Contamos esas anotaciones y discretizamos el conteo en 3 bins.

Salida: un CSV con una fila por imagen y estas columnas:

    path,source,scene_id,person_count,label

    path          ruta absoluta o relativa a la imagen
    source        coco | crowdhuman | shanghaitech | roboflow_bus | local_valpo
    scene_id      identificador de escena/camara. Se usa para que frames de la
                  misma escena no queden repartidos entre train y test.
                  Si no hay escenas, usar el propio nombre del archivo.
    person_count  numero de personas anotadas
    label         baja | media | alta

Uso:
    python src/labels.py --raw-dir data/raw --out data/index.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Criterio de etiquetado. Este es el texto que va TAL CUAL en la lamina de datos.
# ---------------------------------------------------------------------------
BINS = [(0, 3, "baja"), (4, 10, "media"), (11, 10**9, "alta")]

# Para COCO/Open Images: solo nos quedamos con imagenes que ademas de personas
# tengan contexto vial. Si no filtramos, entran interiores, playas, deportes, etc.
CONTEXT_CATEGORIES = {"bus", "car", "truck", "traffic light", "stop sign", "bicycle"}


def count_to_label(n: int) -> str:
    for lo, hi, name in BINS:
        if lo <= n <= hi:
            return name
    raise ValueError(f"conteo fuera de rango: {n}")


# ---------------------------------------------------------------------------
# Un loader por fuente. Cada uno devuelve una lista de dicts.
# TODO(equipo): completar segun como quedaron los datos descargados en data/raw/.
# ---------------------------------------------------------------------------

def load_coco(raw_dir: Path) -> list[dict]:
    """COCO: instances_train2017.json / instances_val2017.json.

    Filtro: la imagen debe tener >=1 caja `person` Y >=1 caja de CONTEXT_CATEGORIES.
    """
    rows: list[dict] = []
    ann_file = raw_dir / "coco" / "annotations" / "instances_train2017.json"
    if not ann_file.exists():
        print(f"[coco] no encontrado: {ann_file} — se omite")
        return rows

    data = json.loads(ann_file.read_text())
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    img_meta = {im["id"]: im for im in data["images"]}

    per_image: dict[int, dict] = {}
    for ann in data["annotations"]:
        name = cat_name[ann["category_id"]]
        d = per_image.setdefault(ann["image_id"], {"persons": 0, "context": set()})
        if name == "person":
            d["persons"] += 1
        elif name in CONTEXT_CATEGORIES:
            d["context"].add(name)

    for img_id, d in per_image.items():
        if not d["context"]:
            continue  # sin contexto vial -> fuera
        fname = img_meta[img_id]["file_name"]
        rows.append(
            {
                "path": str(raw_dir / "coco" / "train2017" / fname),
                "source": "coco",
                "scene_id": f"coco_{img_id}",  # COCO no tiene escenas repetidas
                "person_count": d["persons"],
            }
        )
    print(f"[coco] {len(rows)} imagenes con contexto vial")
    return rows


def load_crowdhuman(raw_dir: Path) -> list[dict]:
    """CrowdHuman: annotation_train.odgt (un JSON por linea).

    Contamos cajas visible-body (`vbox`), ignorando las marcadas como ignore.
    Aporta casi toda la clase `alta`.
    """
    rows: list[dict] = []
    ann_file = raw_dir / "crowdhuman" / "annotation_train.odgt"
    if not ann_file.exists():
        print(f"[crowdhuman] no encontrado: {ann_file} — se omite")
        return rows

    for line in ann_file.read_text().splitlines():
        rec = json.loads(line)
        n = sum(1 for b in rec["gtboxes"] if b.get("tag") == "person"
                and b.get("extra", {}).get("ignore", 0) == 0)
        rows.append(
            {
                "path": str(raw_dir / "crowdhuman" / "Images" / f"{rec['ID']}.jpg"),
                "source": "crowdhuman",
                "scene_id": f"ch_{rec['ID']}",
                "person_count": n,
            }
        )
    print(f"[crowdhuman] {len(rows)} imagenes")
    return rows


def load_shanghaitech(raw_dir: Path) -> list[dict]:
    """ShanghaiTech Part B: .mat con puntos de cabeza. Escenas de calle.

    OJO: esta fuente es casi toda clase `alta`. Si la dejamos entera, el modelo
    puede aprender "estilo ShanghaiTech = alta" en vez de densidad real.
    Ver docs/RIESGOS.md, riesgo 1. Submuestrear si hace falta.
    """
    rows: list[dict] = []
    try:
        from scipy.io import loadmat
    except ImportError:
        print("[shanghaitech] falta scipy — se omite")
        return rows

    gt_dir = raw_dir / "shanghaitech" / "part_B" / "train_data" / "ground-truth"
    img_dir = raw_dir / "shanghaitech" / "part_B" / "train_data" / "images"
    if not gt_dir.exists():
        print(f"[shanghaitech] no encontrado: {gt_dir} — se omite")
        return rows

    for mat_path in sorted(gt_dir.glob("GT_*.mat")):
        stem = mat_path.stem.replace("GT_", "")
        mat = loadmat(str(mat_path))
        n = int(mat["image_info"][0, 0][0, 0][0].shape[0])
        rows.append(
            {
                "path": str(img_dir / f"{stem}.jpg"),
                "source": "shanghaitech",
                "scene_id": f"st_{stem}",
                "person_count": n,
            }
        )
    print(f"[shanghaitech] {len(rows)} imagenes")
    return rows


def load_roboflow_bus(raw_dir: Path) -> list[dict]:
    """Roboflow Universe, formato COCO exportado.

    TODO(equipo): completar segun el dataset que elijan. Aporta contexto de
    paradero/micro. Si el dataset NO anota personas, no sirve para etiquetar
    ocupacion: usarlo solo para la discusion cualitativa, no para entrenar.
    """
    print("[roboflow_bus] TODO — se omite")
    return []


def load_local_valpo(raw_dir: Path) -> list[dict]:
    """Nuestras fotos del Gran Valparaiso.

    Espera data/local_valpo/labels.csv con columnas: filename,person_count
    Etiquetadas a mano por el equipo. NUNCA entran a train ni a val.
    """
    rows: list[dict] = []
    csv_path = Path("data/local_valpo/labels.csv")
    if not csv_path.exists():
        print(f"[local_valpo] no encontrado: {csv_path} — se omite")
        return rows

    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        rows.append(
            {
                "path": str(Path("data/local_valpo") / r["filename"]),
                "source": "local_valpo",
                "scene_id": f"lv_{r['filename']}",
                "person_count": int(r["person_count"]),
            }
        )
    print(f"[local_valpo] {len(rows)} imagenes (solo test)")
    return rows


LOADERS = [load_coco, load_crowdhuman, load_shanghaitech,
           load_roboflow_bus, load_local_valpo]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/index.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    for loader in LOADERS:
        rows.extend(loader(args.raw_dir))

    if not rows:
        raise SystemExit("No se cargo ninguna imagen. Revisar data/raw/.")

    df = pd.DataFrame(rows)
    df["label"] = df["person_count"].apply(count_to_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n{len(df)} imagenes -> {args.out}\n")
    print("Distribucion por fuente y clase (ESTA TABLA VA EN LA LAMINA DE DATOS):")
    print(pd.crosstab(df["source"], df["label"]))


if __name__ == "__main__":
    main()
