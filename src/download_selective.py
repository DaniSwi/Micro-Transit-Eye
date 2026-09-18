"""
download_selective.py — baja SOLO las imagenes que van a sobrevivir al filtro.

La idea: las anotaciones de COCO pesan ~250 MB, las imagenes ~18 GB. Si filtramos
primero sobre las anotaciones y despues bajamos imagen por imagen desde su URL,
pasamos de 18 GB a unos cientos de MB.

Ademas reescalamos al vuelo (lado mayor 640 px por defecto): igual entrenamos a
224x224, guardar en resolucion original es desperdicio de disco y de tiempo de
lectura en Colab.

Uso tipico:
    python src/download_selective.py coco --cap-per-class 1500 --out data/raw/coco
    python src/download_selective.py shanghaitech --out data/raw/shanghaitech
    python src/download_selective.py crowdhuman-extract \
        --zip /content/drive/MyDrive/g7/CrowdHuman_train01.zip \
        --odgt data/raw/crowdhuman/annotation_train.odgt \
        --cap-per-class 800 --out data/raw/crowdhuman

Es reanudable: si se corta la sesion de Colab, vuelve a correr el mismo comando
y salta lo ya descargado.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm

# Mismos criterios que labels.py. Si cambian alla, cambiar aca.
BINS = [(0, 3, "baja"), (4, 10, "media"), (11, 10**9, "alta")]
CONTEXT_CATEGORIES = {"bus", "car", "truck", "traffic light", "stop sign", "bicycle"}

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
SHANGHAITECH_NOTE = (
    "ShanghaiTech no tiene un mirror estable y unico. Descarguenlo a mano desde "
    "Kaggle (buscar 'ShanghaiTech Crowd Counting Dataset') y descompriman Part B "
    "en la carpeta indicada. Pesa ~300 MB, no vale la pena filtrarlo."
)


def count_to_label(n: int) -> str:
    for lo, hi, name in BINS:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def http_get(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "g7-micro-transit-eye/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def save_resized(raw: bytes, dest: Path, max_side: int) -> bool:
    """Guarda la imagen reescalada. Devuelve False si el archivo esta corrupto."""
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        return False
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.BILINEAR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, quality=90)
    return True


def balanced_sample(items: list[dict], cap_per_class: int, seed: int) -> list[dict]:
    """Toma hasta cap_per_class elementos de cada clase. Evita que una fuente
    aporte 20.000 imagenes de 'baja' y 12 de 'alta'."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for it in items:
        by_label.setdefault(it["label"], []).append(it)
    out: list[dict] = []
    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        keep = group[:cap_per_class] if cap_per_class else group
        out.extend(keep)
        print(f"  {label}: {len(keep)} de {len(group)} disponibles")
    return out


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------

def fetch_coco_annotations(out_dir: Path) -> Path:
    """Baja y descomprime solo el JSON de instancias (~250 MB comprimido)."""
    ann_dir = out_dir / "annotations"
    ann_file = ann_dir / "instances_train2017.json"
    if ann_file.exists():
        print(f"anotaciones ya presentes: {ann_file}")
        return ann_file

    print(f"descargando anotaciones COCO (~250 MB) ...")
    ann_dir.mkdir(parents=True, exist_ok=True)
    blob = http_get(COCO_ANN_URL, timeout=600)
    with zipfile.ZipFile(BytesIO(blob)) as zf:
        member = "annotations/instances_train2017.json"
        with zf.open(member) as src, open(ann_file, "wb") as dst:
            dst.write(src.read())
    print(f"listo: {ann_file}")
    return ann_file


def filter_coco(ann_file: Path) -> list[dict]:
    """Devuelve las imagenes con >=1 persona Y >=1 objeto de contexto vial."""
    print("leyendo anotaciones (tarda 1-2 min, el JSON pesa ~450 MB sin comprimir)")
    data = json.loads(ann_file.read_text())
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    img_meta = {im["id"]: im for im in data["images"]}

    per_image: dict[int, dict] = {}
    for ann in data["annotations"]:
        name = cat_name[ann["category_id"]]
        if name == "person":
            per_image.setdefault(ann["image_id"], {"persons": 0, "context": set()})["persons"] += 1
        elif name in CONTEXT_CATEGORIES:
            per_image.setdefault(ann["image_id"], {"persons": 0, "context": set()})["context"].add(name)

    selected = []
    for img_id, d in per_image.items():
        if d["persons"] == 0 or not d["context"]:
            continue
        meta = img_meta[img_id]
        selected.append({
            "image_id": img_id,
            "file_name": meta["file_name"],
            "url": meta.get("coco_url") or
                   f"http://images.cocodataset.org/train2017/{meta['file_name']}",
            "person_count": d["persons"],
            "label": count_to_label(d["persons"]),
        })

    print(f"\n{len(per_image)} imagenes con personas -> "
          f"{len(selected)} pasan el filtro de contexto vial")
    return selected


def download_many(items: list[dict], img_dir: Path, max_side: int,
                  workers: int) -> list[dict]:
    """Descarga en paralelo. Reanudable: salta lo que ya existe."""
    img_dir.mkdir(parents=True, exist_ok=True)
    pending = [it for it in items if not (img_dir / it["file_name"]).exists()]
    print(f"\n{len(items) - len(pending)} ya estaban en disco, "
          f"faltan {len(pending)}")

    ok: list[dict] = [it for it in items if (img_dir / it["file_name"]).exists()]
    failed = 0

    def work(it: dict):
        try:
            raw = http_get(it["url"])
            if save_resized(raw, img_dir / it["file_name"], max_side):
                return it
        except Exception:
            return None
        return None

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(work, it): it for it in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="descargando"):
                res = fut.result()
                if res:
                    ok.append(res)
                else:
                    failed += 1

    if failed:
        print(f"AVISO: {failed} descargas fallaron (se ignoran, no rompen nada)")
    return ok


def cmd_coco(args) -> None:
    out_dir = Path(args.out)
    ann_file = fetch_coco_annotations(out_dir)
    items = filter_coco(ann_file)

    print("\nmuestreo balanceado por clase:")
    items = balanced_sample(items, args.cap_per_class, args.seed)

    items = download_many(items, out_dir / "images", args.max_side, args.workers)
    write_manifest(items, out_dir / "manifest.csv", source="coco",
                   img_dir=out_dir / "images")


# ---------------------------------------------------------------------------
# CrowdHuman: NO permite descarga selectiva
# ---------------------------------------------------------------------------

def cmd_crowdhuman_extract(args) -> None:
    """CrowdHuman se distribuye en ZIPs grandes, no por imagen suelta. Pero si
    ya tienen el ZIP en Drive, podemos extraer SOLO las que nos sirven sin
    descomprimirlo entero."""
    odgt = Path(args.odgt)
    if not odgt.exists():
        sys.exit(f"falta el archivo de anotaciones: {odgt}\n"
                 "Descarguenlo desde la pagina de CrowdHuman (es liviano, ~50 MB).")

    items = []
    for line in odgt.read_text().splitlines():
        rec = json.loads(line)
        n = sum(1 for b in rec["gtboxes"]
                if b.get("tag") == "person"
                and b.get("extra", {}).get("ignore", 0) == 0)
        items.append({"file_name": f"{rec['ID']}.jpg", "person_count": n,
                      "label": count_to_label(n)})

    print(f"{len(items)} imagenes anotadas\nmuestreo balanceado por clase:")
    items = balanced_sample(items, args.cap_per_class, args.seed)
    wanted = {it["file_name"]: it for it in items}

    out_dir = Path(args.out)
    img_dir = out_dir / "Images"
    img_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    with zipfile.ZipFile(args.zip) as zf:
        names = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
        hits = [f for f in wanted if f in names]
        print(f"{len(hits)} de las seleccionadas estan en este ZIP "
              f"(el dataset viene partido en varios)")
        for fname in tqdm(hits, desc="extrayendo"):
            dest = img_dir / fname
            if dest.exists():
                extracted.append(wanted[fname])
                continue
            with zf.open(names[fname]) as src:
                if save_resized(src.read(), dest, args.max_side):
                    extracted.append(wanted[fname])

    write_manifest(extracted, out_dir / "manifest.csv", source="crowdhuman",
                   img_dir=img_dir)


# ---------------------------------------------------------------------------
# ShanghaiTech: tan chico que no vale la pena filtrar
# ---------------------------------------------------------------------------

def cmd_shanghaitech(args) -> None:
    print(SHANGHAITECH_NOTE)
    print(f"\nEstructura esperada al terminar:\n"
          f"  {args.out}/part_B/train_data/images/\n"
          f"  {args.out}/part_B/train_data/ground-truth/\n")


# ---------------------------------------------------------------------------

def write_manifest(items: list[dict], path: Path, source: str, img_dir: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "scene_id", "person_count", "label"])
        for it in items:
            w.writerow([
                str(img_dir / it["file_name"]),
                source,
                f"{source}_{Path(it['file_name']).stem}",
                it["person_count"],
                it["label"],
            ])

    total_mb = sum(p.stat().st_size for p in img_dir.glob("*")) / 1e6
    print(f"\n{len(items)} imagenes, {total_mb:.0f} MB en disco")
    print(f"manifiesto: {path}")
    print("\nlabels.py puede leer este manifiesto directamente en vez de "
          "re-parsear las anotaciones.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="carpeta destino")
    common.add_argument("--cap-per-class", type=int, default=1500,
                        help="maximo de imagenes por clase (0 = sin limite)")
    common.add_argument("--max-side", type=int, default=640,
                        help="reescalar al vuelo, lado mayor en px (0 = original)")
    common.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("coco", parents=[common])
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=cmd_coco)

    p = sub.add_parser("crowdhuman-extract", parents=[common])
    p.add_argument("--zip", required=True, help="CrowdHuman_trainXX.zip")
    p.add_argument("--odgt", required=True, help="annotation_train.odgt")
    p.set_defaults(func=cmd_crowdhuman_extract)

    p = sub.add_parser("shanghaitech", parents=[common])
    p.set_defaults(func=cmd_shanghaitech)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
