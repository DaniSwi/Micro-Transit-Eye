"""
dataset.py — Dataset y transforms.

IMPORTANTE: este archivo lo usan LOS DOS modelos sin modificacion. Si alguien
cambia una augmentacion "solo para su modelo", la comparacion deja de valer y
perdemos el criterio de la rubrica. Cualquier cambio aqui se avisa al grupo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = ["baja", "media", "alta"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # ColorJitter suave: ayuda contra la brecha de dominio (micros de
            # colores distintos, luz nocturna de Valparaiso). Ver docs/RIESGOS.md.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ParaderoDataset(Dataset):
    """Una fila de index.csv = una imagen + su etiqueta de ocupacion."""

    def __init__(self, index_path: str | Path, split: str, image_size: int,
                 train: bool | None = None):
        df = pd.read_csv(index_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"split '{split}' vacio en {index_path}")
        is_train = (split == "train") if train is None else train
        self.tf = build_transforms(image_size, is_train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), CLASS_TO_IDX[row["label"]]

    def class_weights(self) -> torch.Tensor:
        """Pesos inversos a la frecuencia. Las clases estan desbalanceadas:
        sin esto el modelo predice 'baja' siempre y el accuracy engaña."""
        counts = self.df["label"].value_counts()
        w = [len(self.df) / (len(CLASS_NAMES) * counts.get(c, 1)) for c in CLASS_NAMES]
        return torch.tensor(w, dtype=torch.float)

    @property
    def sources(self):
        """Para el desglose de metricas por fuente (riesgo 1)."""
        return self.df["source"].values


def build_loader(index_path, split, image_size, batch_size, num_workers,
                 shuffle: bool | None = None) -> DataLoader:
    ds = ParaderoDataset(index_path, split, image_size)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)
