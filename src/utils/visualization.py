"""
All plotting utilities used across the project. Every function here plots
data that is passed in explicitly by the caller — nothing is invented.
"""
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe (Colab, CI, servers)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_class_distribution(class_counts: Dict[str, int], save_path: str, title: str = "Class Distribution") -> None:
    _ensure_dir(save_path)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(classes, counts, color=sns.color_palette("viridis", len(classes)))
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title(title)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c),
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str,
                           normalize: bool = False, title: Optional[str] = None) -> None:
    _ensure_dir(save_path)
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
        default_title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = ".0f"
        default_title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={"label": "Proportion" if normalize else "Count"})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title or default_title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_path: str) -> Dict:
    """One-vs-rest ROC curve per class + macro average. Returns AUC values."""
    _ensure_dir(save_path)
    n_classes = len(class_names)
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = {}
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    valid_curves = 0

    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            aucs[class_names[i]] = None
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[class_names[i]] = float(roc_auc)
        ax.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        valid_curves += 1

    if valid_curves > 0:
        mean_tpr /= valid_curves
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro-average (AUC={macro_auc:.3f})")
        aucs["macro"] = float(macro_auc)
    else:
        aucs["macro"] = None

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return aucs


def plot_precision_recall_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_path: str) -> None:
    _ensure_dir(save_path)
    n_classes = len(class_names)
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ax.plot(recall, precision, lw=1.5, label=class_names[i])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multiclass Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], save_dir: str) -> None:
    """history expects keys like train_loss, val_loss, train_acc, val_acc,
    optionally train_f1 / val_f1. Only plots keys that are present."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    def _plot_pair(key_train, key_val, ylabel, filename):
        if key_train not in history:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, history[key_train], label="Train", marker="o", ms=3)
        if key_val in history:
            ax.plot(epochs, history[key_val], label="Validation", marker="o", ms=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} over Training")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close(fig)

    _plot_pair("train_loss", "val_loss", "Loss", "training_loss.png")
    _plot_pair("val_loss", "val_loss", "Validation Loss", "validation_loss.png")
    _plot_pair("train_acc", "val_acc", "Accuracy", "training_accuracy.png")
    _plot_pair("val_acc", "val_acc", "Validation Accuracy", "validation_accuracy.png")
    if "train_f1" in history:
        _plot_pair("train_f1", "val_f1", "Macro F1", "training_f1.png")


def plot_fusion_weights(resnet_weight: float, vit_weight: float, save_path: str) -> None:
    """Visualize the *learned* (not manually assigned) branch contribution
    weights produced by the attention fusion module."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = ["ResNet50\n(local features)", "ViT\n(global features)"]
    values = [resnet_weight, vit_weight]
    colors = ["#4C72B0", "#DD8452"]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Learned Attention Weight")
    ax.set_title("Learned Branch Contribution (Attention Fusion)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison_bar(df, metric_col: str, save_path: str) -> None:
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["Model"], df[metric_col], color=sns.color_palette("mako", len(df)))
    ax.set_ylabel(metric_col)
    ax.set_title(f"Model Comparison — {metric_col}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
