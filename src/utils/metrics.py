"""
Metrics for multiclass skin-lesion classification.

All metrics are computed from real predictions/labels passed in by the
caller — nothing here fabricates numbers. If a metric cannot be computed
(e.g. AUC with only one class present in a tiny debug split) the function
returns None and callers must display that clearly rather than a fake value.

Averaging strategy (documented per requirement #33):
- "macro"    : unweighted mean across classes -> treats every class equally,
               most informative for imbalanced datasets like HAM10000.
- "weighted" : mean weighted by class support -> reflects overall dataset
               composition (dominated by the majority class, `nv`).
We report both, plus per-class values, so imbalance effects are visible
rather than hidden behind a single aggregate number.
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_specificity_sensitivity(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-class sensitivity (recall), specificity, FPR, FNR, NPV from a
    multiclass confusion matrix using the standard one-vs-rest decomposition.
    """
    n_classes = cm.shape[0]
    sensitivity = np.zeros(n_classes)
    specificity = np.zeros(n_classes)
    fpr = np.zeros(n_classes)
    fnr = np.zeros(n_classes)
    npv = np.zeros(n_classes)

    total = cm.sum()
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr[i] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        npv[i] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "npv": npv,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute the full metrics suite for one evaluation run.

    Args:
        y_true: (N,) integer ground-truth labels
        y_pred: (N,) integer predicted labels
        y_prob: (N, C) predicted class probabilities (softmax output),
                required for ROC-AUC. If None, AUC fields are set to None.
        class_names: optional list of class name strings for per-class dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    ss = compute_specificity_sensitivity(cm)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else None,
        "sensitivity_macro": float(np.mean(ss["sensitivity"])),
        "specificity_macro": float(np.mean(ss["specificity"])),
        "fpr_macro": float(np.mean(ss["fpr"])),
        "fnr_macro": float(np.mean(ss["fnr"])),
        "npv_macro": float(np.mean(ss["npv"])),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            class_names[i]: {
                "precision": float(precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
                "sensitivity": float(ss["sensitivity"][i]),
                "specificity": float(ss["specificity"][i]),
                "support": int((y_true == i).sum()),
            }
            for i in range(n_classes)
        },
    }

    # AUC requires probability scores; only compute if provided and if
    # every class actually appears in y_true (else sklearn raises).
    if y_prob is not None and len(np.unique(y_true)) == n_classes:
        try:
            results["auc_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
            results["auc_weighted"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
            per_class_auc = {}
            for i in range(n_classes):
                y_true_bin = (y_true == i).astype(int)
                try:
                    per_class_auc[class_names[i]] = float(
                        roc_auc_score(y_true_bin, y_prob[:, i])
                    )
                except ValueError:
                    per_class_auc[class_names[i]] = None
            results["auc_per_class"] = per_class_auc
        except ValueError as e:
            results["auc_macro"] = None
            results["auc_weighted"] = None
            results["auc_note"] = f"AUC could not be computed: {e}"
    else:
        results["auc_macro"] = None
        results["auc_weighted"] = None
        if y_prob is not None:
            results["auc_note"] = (
                "Not all classes present in this split's ground truth; "
                "AUC skipped rather than fabricated."
            )

    return results


def metrics_to_flat_row(results: Dict, model_name: str, training_time_sec: Optional[float] = None) -> Dict:
    """Flatten compute_all_metrics() output into one row for the model
    comparison CSV/table (requirement #19)."""
    return {
        "Model": model_name,
        "Accuracy": results["accuracy"],
        "Precision (macro)": results["precision_macro"],
        "Recall (macro)": results["recall_macro"],
        "Macro F1": results["f1_macro"],
        "Weighted F1": results["f1_weighted"],
        "Sensitivity": results["sensitivity_macro"],
        "Specificity": results["specificity_macro"],
        "MCC": results["mcc"],
        "AUC (macro)": results.get("auc_macro"),
        "Training Time (s)": training_time_sec,
    }
