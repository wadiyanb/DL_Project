"""
train.py — Training and validation loops for the Dual GNN model.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import Config
from model import DualGNN
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Can be a tensor of class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



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
    class_weights: torch.Tensor = None,
):
    """
    Full training loop with:
        • early stopping (patience)
        • best-model checkpointing
        • epoch-level logging
    """
    # ── class imbalance handling (weighted loss) ──────────────────────────
    class_weights = None
    if cfg.use_weighted_loss:
        labels = []
        ds = getattr(train_loader, "dataset", None)
        if ds is not None and hasattr(ds, "samples"):
            labels = [int(l) for _, l in ds.samples]
        if labels:
            counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
            counts = torch.clamp(counts, min=1).float()
            # Smooth + clip weights for long-tailed datasets:
            # w_i = (median_count / count_i) ^ power, normalized to mean 1, then clipped.
            median = torch.median(counts)
            power = float(getattr(cfg, "class_weight_power", 0.5))
            w = (median / counts).pow(power)
            w = w / torch.clamp(w.mean(), min=1e-12)
            max_w = float(getattr(cfg, "max_class_weight", 5.0))
            w = torch.clamp(w, max=max_w)
            class_weights = w.to(device)

    if getattr(cfg, "use_focal_loss", False):
        criterion = FocalLoss(alpha=class_weights, gamma=getattr(cfg, "focal_loss_gamma", 2.0))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
            min_lr=cfg.lr_scheduler_min_lr,
        )

    best_val_acc = 0.0
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    ckpt_path = cfg.checkpoint_dir / "best_model.pt"

    print(f"\n{'='*60}")
    print(f" Training — mode: {cfg.model_mode} | device: {device}")
    print(f" Epochs: {cfg.epochs} | LR: {cfg.lr} | Patience: {cfg.patience}")
    if cfg.use_weighted_loss:
        if class_weights is None:
            print(" Weighted loss: on (fallback: no labels)")
        else:
            print(
                " Weighted loss: on "
                f"(power={getattr(cfg, 'class_weight_power', 0.5)}, "
                f"max_w={getattr(cfg, 'max_class_weight', 5.0)})"
            )
    if cfg.use_lr_scheduler:
        print(
            " LR scheduler: ReduceLROnPlateau("
            f"factor={cfg.lr_scheduler_factor}, "
            f"patience={cfg.lr_scheduler_patience}, "
            f"min_lr={cfg.lr_scheduler_min_lr})"
        )
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{cfg.epochs} │ "
            f"train_loss {train_loss:.4f}  acc {train_acc:.4f} │ "
            f"val_loss {val_loss:.4f}  acc {val_acc:.4f} │ "
            f"lr {lr_now:.2e} │ {dt:.1f}s"
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
