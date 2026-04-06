"""
main.py — CLI entry point for the Dual GNN Foraminifera Classification Pipeline.

Usage
-----
    # Full pipeline (preprocess → build graphs → train → evaluate)
    python main.py --mode full --data_dir dataset/images --epochs 50

    # Preprocess only
    python main.py --mode preprocess --data_dir dataset/images

    # Train only (graphs must already be cached)
    python main.py --mode train --epochs 30

    # Evaluate a saved checkpoint
    python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

    # Smoke test with synthetic data
    python main.py --mode full --test --epochs 2

    # Ablation: all three modes
    python main.py --mode ablation --data_dir dataset/images --epochs 30
"""

import argparse
import os
import sys
import shutil
import json
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from data_preprocessing import (
    load_dataset, preprocess_image, split_dataset, save_split_info,
)
from dataset import get_dataloaders
from model import DualGNN
from train import run_training
from evaluate import (
    evaluate_model, plot_confusion_matrix,
    plot_training_curves, compare_models,
)
from gan_train import train_gan, generate_synthetic_data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_split_paths(paths: list, cfg: Config) -> list:
    """
    Resolve paths read from split_info.json.
    - If a path is absolute: keep it (legacy format).
    - If a path is relative: treat it as relative to cfg.data_dir (portable format).
    """
    out = []
    for p in paths:
        pp = Path(p)
        out.append(str(pp if pp.is_absolute() else (cfg.data_dir / pp).resolve()))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TEST DATA
# ═══════════════════════════════════════════════════════════════════════════


def _create_synthetic_dataset(root: Path, n_classes: int = 3, n_per_class: int = 10):
    """Create tiny random images for smoke‑testing."""
    import cv2

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for c in range(n_classes):
        cls_dir = root / f"class_{c}"
        cls_dir.mkdir()
        for i in range(n_per_class):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(cls_dir / f"img_{i:03d}.png"), img)

    print(f"[test] Created synthetic dataset: {n_classes} classes × {n_per_class} imgs → {root}")
    return root


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════


def step_preprocess(cfg: Config):
    """Load dataset, split, and save metadata."""
    print("\n▸ PREPROCESS")
    samples, label_map = load_dataset(cfg.data_dir)
    train, val, test = split_dataset(samples, cfg)
    save_split_info(train, val, test, label_map, cfg.processed_dir, data_dir=cfg.data_dir)
    return train, val, test, label_map


def step_train(cfg: Config, train_loader, val_loader, num_classes: int, device):
    """Instantiate model and train."""
    print("\n▸ TRAINING")
    model = DualGNN(num_classes=num_classes, cfg=cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} total, {trainable:,} trainable")

    history, ckpt = run_training(model, train_loader, val_loader, cfg, device)
    plot_training_curves(history, cfg.results_dir / "training_curves.png")
    return model, history, ckpt


def step_evaluate(cfg: Config, model, test_loader, label_map, device, tag="hybrid"):
    """Evaluate model on test set."""
    print(f"\n▸ EVALUATE ({tag})")
    results = evaluate_model(model, test_loader, device, label_names=label_map)

    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Macro‑F1 : {results['f1']:.4f}")
    print(f"\n{results['report']}")

    plot_confusion_matrix(
        results["y_true"], results["y_pred"],
        label_names=label_map,
        save_path=cfg.results_dir / f"confusion_matrix_{tag}.png",
        title=f"Confusion Matrix — {tag}",
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Dual GNN Foraminifera Classification Pipeline"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["preprocess", "train", "evaluate", "full", "ablation", "gan_train", "gan_generate"],
        help="Pipeline step to run.",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to images/")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--model_mode", type=str, default="hybrid",
        choices=["hybrid", "superpixel_only", "keypoint_only"],
    )
    parser.add_argument("--test", action="store_true", help="Use synthetic data for smoke test")
    parser.add_argument("--target_count", type=int, default=500, help="Target number of images per class after GAN synthesis")
    args = parser.parse_args()

    # ── config ────────────────────────────────────────────────────────────
    cfg = Config()

    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.model_mode:
        cfg.model_mode = args.model_mode

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── synthetic test mode ───────────────────────────────────────────────
    if args.test:
        cfg.data_dir = _create_synthetic_dataset(
            cfg.project_root / "_test_data", n_classes=3, n_per_class=10
        )
        cfg.image_size = 64
        cfg.n_segments = 20
        cfg.n_keypoints = 15
        cfg.batch_size = 4

    set_seed(cfg.seed)

    # ── route to pipeline step ────────────────────────────────────────────
    if args.mode == "preprocess":
        step_preprocess(cfg)
        return
        
    if args.mode == "gan_train":
        samples, label_map = load_dataset(cfg.data_dir)
        num_classes = len(label_map)
        train_gan(cfg, num_classes, device)
        return
        
    if args.mode == "gan_generate":
        samples, label_map = load_dataset(cfg.data_dir)
        num_classes = len(label_map)
        target_count = args.target_count
        generate_synthetic_data(cfg, num_classes, target_count, device)
        return

    if args.mode in ("full", "train", "evaluate", "ablation"):
        # Always preprocess first for full/ablation
        if args.mode in ("full", "ablation"):
            train, val, test, label_map = step_preprocess(cfg)
        else:
            # Load from saved split
            split_file = cfg.processed_dir / "split_info.json"
            if not split_file.exists():
                print("[error] No split_info.json found. Run --mode preprocess first.")
                sys.exit(1)
            with open(split_file) as f:
                info = json.load(f)
            label_map = {int(k): v for k, v in info["label_map"].items()}
            train_paths = resolve_split_paths(info["train"], cfg)
            val_paths = resolve_split_paths(info["val"], cfg)
            test_paths = resolve_split_paths(info["test"], cfg)

            # Re-load raw samples for labels
            samples, _ = load_dataset(cfg.data_dir)
            path_to_label = {str(Path(p).resolve()): l for p, l in samples}
            train = [(p, path_to_label[p]) for p in train_paths if p in path_to_label]
            val = [(p, path_to_label[p]) for p in val_paths if p in path_to_label]
            test = [(p, path_to_label[p]) for p in test_paths if p in path_to_label]

        num_classes = len(label_map)
        train_loader, val_loader, test_loader = get_dataloaders(
            train, val, test, cfg
        )

        if args.mode == "full":
            model, history, ckpt = step_train(
                cfg, train_loader, val_loader, num_classes, device
            )
            step_evaluate(cfg, model, test_loader, label_map, device, tag=cfg.model_mode)

        elif args.mode == "train":
            model, history, ckpt = step_train(
                cfg, train_loader, val_loader, num_classes, device
            )

        elif args.mode == "evaluate":
            ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "best_model.pt"
            if not ckpt_path.exists():
                print(f"[error] Checkpoint not found: {ckpt_path}")
                sys.exit(1)

            ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg.model_mode = ckpt_data.get("config_mode", cfg.model_mode)
            model = DualGNN(num_classes=num_classes, cfg=cfg).to(device)
            model.load_state_dict(ckpt_data["model_state"])
            step_evaluate(cfg, model, test_loader, label_map, device, tag=cfg.model_mode)

        elif args.mode == "ablation":
            # Train and evaluate all three modes
            all_results = {}
            for mode in ["superpixel_only", "keypoint_only", "hybrid"]:
                print(f"\n{'#'*60}")
                print(f"  ABLATION — {mode}")
                print(f"{'#'*60}")
                cfg.model_mode = mode
                # Need fresh processed dir per mode to avoid stale caches
                model, history, ckpt = step_train(
                    cfg, train_loader, val_loader, num_classes, device
                )
                results = step_evaluate(
                    cfg, model, test_loader, label_map, device, tag=mode
                )
                all_results[mode] = results

            compare_models(all_results, cfg.results_dir / "ablation_comparison.png")

    # ── cleanup synthetic test data ───────────────────────────────────────
    if args.test:
        test_dir = cfg.project_root / "_test_data"
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("[test] Cleaned up synthetic data.")

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
