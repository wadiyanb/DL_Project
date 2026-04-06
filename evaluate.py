"""
evaluate.py — Evaluation metrics and comparison for the Dual GNN model.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

from config import Config
from model import DualGNN


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATE A SINGLE MODEL
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_model(
    model: DualGNN,
    loader,
    device: torch.device,
    label_names: Optional[Dict[int, str]] = None,
) -> dict:
    """
    Run inference on ``loader`` and compute:
        • accuracy
        • macro‑F1
        • per‑class classification report
        • confusion matrix arrays

    Returns a dict with keys: accuracy, f1, report, y_true, y_pred.
    """
    model.eval()
    all_preds, all_labels = [], []

    for sp_batch, kp_batch in loader:
        sp_batch = sp_batch.to(device)
        kp_batch = kp_batch.to(device)
        labels = sp_batch.y.to(device)

        logits = model(sp_batch, kp_batch)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    target_names = None
    if label_names:
        max_label = max(y_true.max(), y_pred.max())
        target_names = [label_names.get(i, str(i)) for i in range(max_label + 1)]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if target_names is None:
        report = classification_report(y_true, y_pred, zero_division=0)
    else:
        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(target_names)),
            target_names=target_names,
            zero_division=0,
        )

    return {
        "accuracy": acc,
        "f1": f1,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX PLOT
# ═══════════════════════════════════════════════════════════════════════════


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]

    labels = [label_names.get(i, str(i)) for i in range(n)] if label_names else list(range(n))

    # Adjust figure size for large label counts
    fig_size = max(6, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm, annot=n <= 20, fmt="d",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[eval] Confusion matrix saved → {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING CURVES PLOT
# ═══════════════════════════════════════════════════════════════════════════


def plot_training_curves(history: dict, save_path: Optional[Path] = None):
    """Plot loss and accuracy curves from training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[eval] Training curves saved → {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════


def compare_models(results: Dict[str, dict], save_path: Optional[Path] = None):
    """
    Print & plot a comparison table for multiple model modes.

    ``results`` maps mode_name → {"accuracy": ..., "f1": ...}
    """
    print("\n" + "=" * 50)
    print("  MODEL COMPARISON")
    print("=" * 50)
    print(f"  {'Mode':<20s}  {'Accuracy':>10s}  {'Macro-F1':>10s}")
    print("-" * 50)
    for mode, r in results.items():
        print(f"  {mode:<20s}  {r['accuracy']:10.4f}  {r['f1']:10.4f}")
    print("=" * 50 + "\n")

    if save_path:
        modes = list(results.keys())
        accs = [results[m]["accuracy"] for m in modes]
        f1s = [results[m]["f1"] for m in modes]

        x = np.arange(len(modes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
        ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[eval] Comparison chart saved → {save_path}")
        plt.close(fig)
