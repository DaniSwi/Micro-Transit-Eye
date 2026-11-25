"""
demo.py — interfaz Gradio: sube una foto de paradero y compara los tres enfoques.

Muestra lado a lado ResNet-50, ViT-B/16 (con su mapa de calor) y el conteo de
YOLOv8. Reutiliza el mismo preprocesamiento (dataset.py), los mismos mapas de
calor (interpret.py) y los mismos umbrales (BINS en labels.py) que la
evaluacion: lo que se ve en la demo es exactamente lo que se midio, no una
version "pulida" para la presentacion.

Requiere: pip install gradio ultralytics (no van en requirements.txt porque
solo este script y baseline_yolo.py las usan).

Uso (desde la raiz del repo, con los checkpoints entrenados):
    python src/demo.py                      # local, http://127.0.0.1:7860
    python src/demo.py --share              # link publico (Colab), dura 72 h
    python src/demo.py --resnet outputs/resnet50/best.pt --vit outputs/vit_b16/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from matplotlib import colormaps
from PIL import Image

from dataset import CLASS_NAMES, build_transforms
from interpret import attention_rollout, grad_cam
from labels import BINS, count_to_label

PERSON_CLASS_ID = 0  # clase 0 en COCO/YOLO es "person"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = timm.create_model(cfg["model"]["name"], pretrained=False,
                              num_classes=cfg["data"]["num_classes"])
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE).eval(), cfg


def center_crop_view(img: Image.Image, size: int) -> Image.Image:
    """La misma vista que recibe el modelo (Resize + CenterCrop de dataset.py),
    sin normalizar: el mapa de calor se dibuja sobre lo que el modelo vio, no
    sobre la foto completa, para no sugerir que miro zonas que recorto."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
    ])(img)


def heatmap_overlay(view: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(view.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(view).astype(np.float32) / 255.0
    mixed = (1 - alpha) * base + alpha * colored
    return Image.fromarray((mixed * 255).astype(np.uint8))


def classify(model: torch.nn.Module, cfg: dict, img: Image.Image) -> tuple[dict, Image.Image]:
    size = cfg["data"]["image_size"]
    x = build_transforms(size, train=False)(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    if "vit" in cfg["model"]["name"].lower():
        cam, _ = attention_rollout(model, x)
    else:
        # Grad-CAM necesita gradientes: va fuera del no_grad de arriba.
        cam, _ = grad_cam(model, x, model.layer4[-1])

    scores = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    return scores, heatmap_overlay(center_crop_view(img, size), cam)


def yolo_count(model, img: Image.Image, conf: float) -> tuple[int, str, Image.Image]:
    result = model(img, verbose=False, conf=conf, classes=[PERSON_CLASS_ID])[0]
    n = 0 if result.boxes is None else len(result.boxes)
    annotated = Image.fromarray(result.plot()[..., ::-1])  # plot() devuelve BGR
    return n, count_to_label(n), annotated


def bins_text() -> str:
    partes = []
    for lo, hi, name in BINS:
        partes.append(f"**{name}** = {lo}–{hi}" if hi < 10**6 else f"**{name}** = más de {lo - 1}")
    return " · ".join(partes) + " personas"


def build_app(resnet_ckpt: Path, vit_ckpt: Path, yolo_weights: str, conf: float,
              examples_dir: Path | None):
    import gradio as gr
    from ultralytics import YOLO

    resnet, resnet_cfg = load_classifier(resnet_ckpt)
    vit, vit_cfg = load_classifier(vit_ckpt)
    yolo = YOLO(yolo_weights)

    def predict(img: Image.Image | None):
        if img is None:
            raise gr.Error("Sube una foto primero.")
        img = img.convert("RGB")
        r_scores, r_cam = classify(resnet, resnet_cfg, img)
        v_scores, v_cam = classify(vit, vit_cfg, img)
        n, y_label, y_img = yolo_count(yolo, img, conf)
        resumen = (
            f"| Enfoque | Predicción |\n|---|---|\n"
            f"| ResNet-50 | **{max(r_scores, key=r_scores.get)}** |\n"
            f"| ViT-B/16 | **{max(v_scores, key=v_scores.get)}** |\n"
            f"| YOLOv8n + conteo | **{y_label}** ({n} personas detectadas) |"
        )
        return r_scores, r_cam, v_scores, v_cam, y_img, resumen

    examples = []
    if examples_dir and examples_dir.is_dir():
        examples = [[str(p)] for p in sorted(examples_dir.iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    with gr.Blocks(title="Micro-Transit Eye") as app:
        gr.Markdown(
            "# Micro-Transit Eye\n"
            "Ocupación de paraderos del Gran Valparaíso: ResNet-50 vs ViT-B/16, "
            "con YOLOv8 contando personas como referencia.\n\n"
            f"Clases: {bins_text()}."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Foto del paradero")
                btn = gr.Button("Clasificar", variant="primary")
                resumen = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    r_lab = gr.Label(num_top_classes=3, label="ResNet-50")
                    v_lab = gr.Label(num_top_classes=3, label="ViT-B/16")
                with gr.Row():
                    r_img = gr.Image(label="ResNet-50 · Grad-CAM (dónde mira)")
                    v_img = gr.Image(label="ViT-B/16 · attention rollout (dónde mira)")
                y_img = gr.Image(label="YOLOv8n · personas detectadas")

        outputs = [r_lab, r_img, v_lab, v_img, y_img, resumen]
        btn.click(predict, inputs=inp, outputs=outputs)
        inp.upload(predict, inputs=inp, outputs=outputs)
        if examples:
            gr.Examples(examples=examples, inputs=inp)
        gr.Markdown(
            "_Los modelos se evaluaron con macro-F1 ≈ 0,62–0,65 en test y ≈ 0,42–0,45 "
            "en Valparaíso: se equivocan, sobre todo sobreestimando la ocupación. "
            "Los mapas de calor se dibujan sobre el recorte central de 224×224 que "
            "recibe el modelo._"
        )
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", type=Path, default=Path("outputs/resnet50/best.pt"))
    ap.add_argument("--vit", type=Path, default=Path("outputs/vit_b16/best.pt"))
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="mismo umbral de confianza que baseline_yolo.py")
    ap.add_argument("--examples", type=Path, default=None,
                    help="carpeta con fotos de ejemplo para mostrar como botones")
    ap.add_argument("--share", action="store_true", help="link publico de Gradio (Colab)")
    args = ap.parse_args()

    for p in (args.resnet, args.vit):
        if not p.exists():
            raise SystemExit(f"no existe {p} — entrena primero o pasa la ruta con --resnet/--vit")

    app = build_app(args.resnet, args.vit, args.yolo, args.conf, args.examples)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
