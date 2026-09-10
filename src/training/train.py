"""
Generic training loop shared by all models (baselines + hybrid).

Implements:
  - automatic GPU detection + mixed precision (only when CUDA is available)
  - two-stage transfer learning (frozen backbone -> partial fine-tune)
  - early stopping on a configurable validation metric (default: macro F1)
  - LR scheduling (cosine)
  - gradient clipping
  - checkpoint + best-model saving
"""
import copy
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def _run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _ in tqdm(loader, leave=False, desc="train" if train else "eval"):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                if train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    cfg: Dict,
    device: torch.device,
    model_name: str = "model",
    two_stage: bool = True,
    on_stage2_start: Optional[Callable] = None,
) -> Dict:
    """Returns training history dict + saves best checkpoint to disk."""
    t_cfg = cfg["training"]
    ckpt_dir = cfg["project"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    use_amp = t_cfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": []}

    best_metric = -np.inf
    best_state = None
    patience_counter = 0
    start_time = time.time()

    def _make_optimizer(lr):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=t_cfg["weight_decay"])

    stages = []
    if two_stage and hasattr(model, "freeze_backbones"):
        model.freeze_backbones()
        stages.append(("stage1", t_cfg["epochs_stage1"], t_cfg["learning_rate_stage1"]))
        stages.append(("stage2", t_cfg["epochs_stage2"], t_cfg["learning_rate_stage2"]))
    else:
        stages.append(("single_stage", t_cfg["epochs_stage1"] + t_cfg["epochs_stage2"],
                        t_cfg["learning_rate_stage1"]))

    for stage_name, n_epochs, lr in stages:
        print(f"\n=== {model_name}: {stage_name} ({n_epochs} epochs, lr={lr}) ===")
        if stage_name == "stage2":
            if hasattr(model, "unfreeze_stage2"):
                model.unfreeze_stage2()
            if on_stage2_start is not None:
                on_stage2_start(model)

        optimizer = _make_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

        for epoch in range(n_epochs):
            train_loss, train_acc, train_f1 = _run_epoch(
                model, train_loader, criterion, optimizer, device, scaler, train=True
            )
            val_loss, val_acc, val_f1 = _run_epoch(
                model, val_loader, criterion, optimizer, device, scaler, train=False
            )
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)

            print(f"[{stage_name} epoch {epoch+1}/{n_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            current_metric = val_f1 if t_cfg["checkpoint_metric"] == "macro_f1" else val_acc
            if current_metric > best_metric:
                best_metric = current_metric
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
                torch.save(best_state, ckpt_path)
                print(f"  -> New best ({t_cfg['checkpoint_metric']}={best_metric:.4f}), saved to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= t_cfg["early_stopping_patience"]:
                    print(f"  -> Early stopping triggered (no improvement for "
                          f"{t_cfg['early_stopping_patience']} epochs).")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.time() - start_time
    history["training_time_sec"] = training_time
    history["best_val_metric"] = best_metric
    print(f"\nTraining complete for {model_name}. Total time: {training_time:.1f}s. "
          f"Best {t_cfg['checkpoint_metric']}: {best_metric:.4f}")

    return history
