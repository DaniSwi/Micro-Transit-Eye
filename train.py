"""
train.py — un solo loop para los dos baselines.

Que ResNet y ViT pasen por EXACTAMENTE el mismo codigo es lo que hace que la
comparacion sea valida. Lo unico que los distingue es el YAML.

Uso:
    python src/train.py --config configs/resnet50.yaml
    python src/train.py --config configs/vit_b16.yaml

Deja en out_dir:
    best.pt        checkpoint con el mejor macro-F1 de validacion
    history.csv    una fila por epoca -> curvas de entrenamiento
    config.yaml    copia del config usado (reproducibilidad)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import CLASS_NAMES, ParaderoDataset, build_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_model(cfg) -> nn.Module:
    return timm.create_model(
        cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["data"]["num_classes"],
    )


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float):
    """La cabeza se entrena mas rapido que el backbone preentrenado.
    Critico en ViT: un lr alto en el backbone lo destruye en 2 epocas."""
    head_names = ("head", "fc", "classifier")
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith(head_names) else backbone).append(p)
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": head, "lr": lr_head},
    ]


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(criterion(out, y).item())
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}  |  run = {cfg['run_name']}")

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(args.config.read_text())

    d, o = cfg["data"], cfg["optim"]
    train_loader = build_loader(d["index"], "train", d["image_size"],
                               o["batch_size"], cfg["train"]["num_workers"])
    val_loader = build_loader(d["index"], "val", d["image_size"],
                             o["batch_size"], cfg["train"]["num_workers"])

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parametros = {n_params/1e6:.1f}M   <- va en la tabla comparativa")

    weights = None
    if o.get("class_weights"):
        weights = ParaderoDataset(d["index"], "train", d["image_size"]).class_weights().to(device)
        print(f"class weights = {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=o.get("label_smoothing", 0.0))

    optimizer = torch.optim.AdamW(
        param_groups(model, o["lr_backbone"], o["lr_head"]),
        weight_decay=o["weight_decay"],
    )
    total_steps = o["epochs"] * max(1, len(train_loader))
    warmup_steps = o.get("warmup_epochs", 0) * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device == "cuda")

    history, best, patience = [], -1.0, 0
    for epoch in range(1, o["epochs"] + 1):
        model.train()
        t0, running = time.time(), []
        for x, y in tqdm(train_loader, desc=f"epoca {epoch}/{o['epochs']}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running.append(loss.item())

        val = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"  train_loss={row['train_loss']:.4f}  val_loss={val['loss']:.4f}  "
              f"val_acc={val['accuracy']:.4f}  val_macroF1={val['macro_f1']:.4f}")

        score = val[cfg["train"]["early_stopping_metric"]]
        if score > best:
            best, patience = score, 0
            torch.save({"model": model.state_dict(),
                        "config": cfg,
                        "epoch": epoch,
                        "val_macro_f1": score,
                        "n_params": n_params}, out_dir / "best.pt")
            print(f"  -> nuevo mejor ({score:.4f}), checkpoint guardado")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print("early stopping")
                break

        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "run_name": cfg["run_name"],
        "model": cfg["model"]["name"],
        "best_val_macro_f1": best,
        "n_params_millions": round(n_params / 1e6, 1),
        "epochs_run": len(history),
    }, indent=2))
    print(f"\nlisto. mejor val macro-F1 = {best:.4f}  ->  {out_dir}")


if __name__ == "__main__":
    main()
