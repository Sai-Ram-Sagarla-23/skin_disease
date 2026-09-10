"""
Preprocessing: resizing + normalization shared by train/val/test, and
DataLoader construction. Augmentation (train-only) lives in augmentation.py.
"""
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .augmentation import get_train_transform
from .dataset import HAM10000Dataset


def get_val_test_transform(cfg: Dict) -> transforms.Compose:
    """Deterministic preprocessing only — NEVER augmented (requirement #8)."""
    size = cfg["data"]["image_size"]
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_dataloaders(train_df, val_df, test_df, cfg: Dict):
    train_transform = get_train_transform(cfg)
    eval_transform = get_val_test_transform(cfg)

    train_ds = HAM10000Dataset(train_df, transform=train_transform)
    val_ds = HAM10000Dataset(val_df, transform=eval_transform)
    test_ds = HAM10000Dataset(test_df, transform=eval_transform)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"]["pin_memory"] and torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def denormalize(tensor_img: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """Inverse of Normalize(), for visualization (Grad-CAM overlays, demo
    figures, etc.)."""
    mean = torch.tensor(cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["data"]["std"]).view(3, 1, 1)
    return (tensor_img.cpu() * std + mean).clamp(0, 1)
