"""
interpret.py — Grad-CAM (CNN) / attention rollout (ViT): en que mira cada modelo.

Responde al riesgo 1 de docs/RIESGOS.md (atajo por fuente de datos): si el
modelo aprendio el "estilo" de COCO/CrowdHuman en vez de la densidad real de
personas, el mapa de calor va a concentrarse en bordes/fondo en vez de en la
gente. Tambien responde directo a la pregunta de la retroalimentacion del
avance: "¿miran a las personas o al fondo?".

Detecta solo segun el nombre del modelo en el config (resnet -> Grad-CAM,
vit -> attention rollout) — no hace falta pasar el metodo a mano.

Uso:
    python src/interpret.py --run outputs/resnet50 --split local --n 6
    python src/interpret.py --run outputs/vit_b16  --split local --n 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import CLASS_NAMES, build_transforms


def pick_samples(df: pd.DataFrame, n: int) -> list[pd.Series]:
    """Un ejemplo por clase primero (para ver las tres), despues completa con
    lo que haya hasta llegar a n — asi la muestra no queda sesgada a la clase
    mayoritaria del split."""
    rows, used_idx = [], set()
    for c in CLASS_NAMES:
        sub = df[df["label"] == c]
        if len(sub):
            rows.append(sub.iloc[0])
            used_idx.add(sub.index[0])
    for idx, row in df.iterrows():
        if len(rows) >= n:
            break
        if idx not in used_idx:
            rows.append(row)
    return rows[:n]


def grad_cam(model: torch.nn.Module, x: torch.Tensor,
             target_layer: torch.nn.Module) -> tuple[np.ndarray, int]:
    """Grad-CAM (Selvaraju et al. 2017) sobre una sola imagen (x: 1x3xHxW).

    Usa el gradiente del logit predicho respecto a los mapas de activacion de
    la ultima capa convolucional como peso de "cuanto importa" cada canal.
    """
    activations, gradients = [], []

    def fwd_hook(_module, _inp, out):
        activations.append(out.detach())

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    out = model(x)
    pred = int(out.argmax(1).item())
    out[0, pred].backward()

    h1.remove()
    h2.remove()

    act = activations[0][0]          # (C, h, w)
    grad = gradients[0][0]           # (C, h, w)
    weights = grad.mean(dim=(1, 2))  # (C,) — importancia promedio de cada canal
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), pred


def attention_rollout(model: torch.nn.Module, x: torch.Tensor) -> tuple[np.ndarray, int]:
    """Attention rollout (Abnar & Zuidema 2020) sobre un ViT de timm.

    Encadena la atencion de todas las capas (sumando la identidad, por las
    conexiones residuales) para estimar cuanto "mira" el token CLS final a
    cada parche de la imagen original.
    """
    attn_maps = []

    def hook(_module, inp, _out):
        attn_maps.append(inp[0].detach())  # (B, heads, N, N), antes del dropout

    handles = []
    for block in model.blocks:
        block.attn.fused_attn = False  # fuerza el camino con softmax explicito (no fused SDPA)
        handles.append(block.attn.attn_drop.register_forward_hook(hook))

    with torch.no_grad():
        out = model(x)
        pred = int(out.argmax(1).item())

    for h in handles:
        h.remove()

    if not attn_maps:
        raise RuntimeError(
            "no se capturaron mapas de atencion — la version de timm instalada "
            "puede usar un camino interno distinto para la atencion del ViT"
        )

    n_tokens = attn_maps[0].shape[-1]
    rollout = torch.eye(n_tokens, device=x.device)
    for a in attn_maps:
        a = a.mean(dim=1)[0]                         # promedio sobre heads -> (N, N)
        a = a + torch.eye(n_tokens, device=x.device)  # conexion residual
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout

    cls_to_patches = rollout[0, 1:]  # fila del token CLS, sin contarse a si mismo
    grid = int(cls_to_patches.numel() ** 0.5)
    cam = cls_to_patches.reshape(grid, grid)
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), pred


def overlay(raw_img: Image.Image, cam: np.ndarray, out_path: Path, title: str) -> None:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(raw_img.size, Image.BILINEAR)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(raw_img)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(raw_img)
    axes[1].imshow(np.array(cam_img), cmap="jet", alpha=0.5)
    axes[1].set_title(title, fontsize=9)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="local", choices=["val", "test", "local"])
    ap.add_argument("--n", type=int, default=6, help="numero de imagenes a visualizar")
    args = ap.parse_args()

    ckpt = torch.load(args.run / "best.pt", map_location="cpu")
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(cfg["model"]["name"], pretrained=False,
                              num_classes=cfg["data"]["num_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    is_vit = "vit" in cfg["model"]["name"].lower()
    tf = build_transforms(cfg["data"]["image_size"], train=False)
    target_layer = None if is_vit else model.layer4[-1]

    df = pd.read_csv(cfg["data"]["index"])
    df = df[df["split"] == args.split].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"split '{args.split}' vacio en {cfg['data']['index']}")
    samples = pick_samples(df, args.n)

    out_dir = args.run / f"interpret_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in samples:
        raw = Image.open(row["path"]).convert("RGB")
        x = tf(raw).unsqueeze(0).to(device)

        if is_vit:
            cam, pred = attention_rollout(model, x)
            metodo = "attention rollout"
        else:
            cam, pred = grad_cam(model, x, target_layer)
            metodo = "Grad-CAM"

        fname = Path(str(row["path"])).stem
        acierto = "OK" if CLASS_NAMES[pred] == row["label"] else "ERROR"
        title = f"{metodo} | real={row['label']} pred={CLASS_NAMES[pred]} [{acierto}]"
        out_path = out_dir / f"{fname}.png"
        overlay(raw, cam, out_path, title)
        print(f"{fname}: real={row['label']}  pred={CLASS_NAMES[pred]}  [{acierto}]  -> {out_path}")

    print(f"\nfiguras en {out_dir}")


if __name__ == "__main__":
    main()
