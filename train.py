"""
train.py — Training and validation loops for the Dual GNN model.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from config import Config
from model import DualGNN


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for sp_batch, kp_batch in loader:
        sp_batch = sp_batch.to(device)
        kp_batch = kp_batch.to(device)
        labels = sp_batch.y.to(device)

        optimizer.zero_grad()
        logits = model(sp_batch, kp_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for sp_batch, kp_batch in loader:
        sp_batch = sp_batch.to(device)
        kp_batch = kp_batch.to(device)
        labels = sp_batch.y.to(device)

        logits = model(sp_batch, kp_batch)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def run_training(
    model: DualGNN,
    train_loader,
    val_loader,
    cfg: Config,
    device: torch.device,
):
    """
    Full training loop with:
        • early stopping (patience)
        • best-model checkpointing
        • epoch-level logging
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    ckpt_path = cfg.checkpoint_dir / "best_model.pt"

    print(f"\n{'='*60}")
    print(f" Training — mode: {cfg.model_mode} | device: {device}")
    print(f" Epochs: {cfg.epochs} | LR: {cfg.lr} | Patience: {cfg.patience}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{cfg.epochs} │ "
            f"train_loss {train_loss:.4f}  acc {train_acc:.4f} │ "
            f"val_loss {val_loss:.4f}  acc {val_acc:.4f} │ "
            f"{dt:.1f}s"
        )

        # ── checkpointing ────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "config_mode": cfg.model_mode,
            }, ckpt_path)
            print(f"    ✓ saved best model (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"\n  Early stopping after {cfg.patience} epochs without improvement.")
                break

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    return history, ckpt_path
