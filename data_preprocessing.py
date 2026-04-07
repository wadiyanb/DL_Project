"""
data_preprocessing.py — Load images from folder-per-class layout, resize, normalise, split.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from config import Config


# ── Public API ─────────────────────────────────────────────────────────────


def load_dataset(
    data_dir: Path,
) -> Tuple[List[Tuple[str, int]], Dict[int, str]]:
    """
    Walk a folder-per-class directory tree and return a flat list of
    (image_path, label_index) plus an {index: class_name} map.

    Expected layout:
        data_dir/
            class_a/
                img1.jpg
                img2.png
            class_b/
                ...
    """
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    if not class_names:
        raise RuntimeError(f"No class sub-folders found in {data_dir}")

    label_map: Dict[int, str] = {i: name for i, name in enumerate(class_names)}
    name_to_idx: Dict[str, int] = {v: k for k, v in label_map.items()}

    samples: List[Tuple[str, int]] = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            if Path(fname).suffix.lower() in valid_exts:
                samples.append((os.path.join(cls_dir, fname), name_to_idx[cls_name]))

    print(f"[data] Found {len(samples)} images across {len(class_names)} classes.")
    return samples, label_map


def preprocess_image(path: str, size: int = 256) -> np.ndarray:
    """
    Read an image, resize to (size × size), normalise to [0, 1] float32.
    Returns an RGB numpy array of shape (H, W, 3).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def split_dataset(
    samples: List[Tuple[str, int]],
    cfg: Config,
) -> Tuple[list, list, list]:
    """
    Stratified train / val / test split.
    Returns three lists of (path, label) tuples.
    """
    paths, labels = zip(*samples)
    paths, labels = list(paths), list(labels)

    # First split: train vs (val+test)
    val_test_ratio = cfg.val_ratio + cfg.test_ratio
    try:
        train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
            paths, labels,
            test_size=val_test_ratio,
            stratify=labels,
            random_state=cfg.seed,
        )
    except ValueError:
        print("[data] Warning: Class imbalance too severe for strict stratification. Falling back to unstratified split.")
        train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
            paths, labels,
            test_size=val_test_ratio,
            random_state=cfg.seed,
        )

    # Second split: val vs test
    relative_test = cfg.test_ratio / val_test_ratio
    try:
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            valtest_paths, valtest_labels,
            test_size=relative_test,
            stratify=valtest_labels,
            random_state=cfg.seed,
        )
    except ValueError:
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            valtest_paths, valtest_labels,
            test_size=relative_test,
            random_state=cfg.seed,
        )

    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))

    print(f"[data] Split → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def save_split_info(
    train: list, val: list, test: list,
    label_map: dict, out_dir: Path,
    data_dir: Path = None,
):
    """Persist split metadata as JSON for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_rel(p: str) -> str:
        if data_dir is None:
            return p
        try:
            return str(Path(p).resolve().relative_to(Path(data_dir).resolve()))
        except Exception:
            # If the image path is outside data_dir, keep original (backward-compatible).
            return p

    info = {
        "path_base": "data_dir" if data_dir is not None else "absolute_or_unknown",
        "label_map": {str(k): v for k, v in label_map.items()},
        "train": [_maybe_rel(p) for p, _ in train],
        "val": [_maybe_rel(p) for p, _ in val],
        "test": [_maybe_rel(p) for p, _ in test],
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[data] Split info saved to {out_dir / 'split_info.json'}")


def augment_offline(data_dir: Path, target_count: int = 0):
    """
    Offline data augmentation to balance minority classes using basic transforms.
    If target_count is 0, figures out the maximum class count and balances all
    classes up to that maximum threshold.
    """
    print(f"\n[data] Running Offline Augmentation")
    samples, label_map = load_dataset(data_dir)
    
    counts = {name: 0 for name in label_map.values()}
    for _, l in samples:
        counts[label_map[l]] += 1
        
    actual_target = target_count if target_count > 0 else max(counts.values())
    print(f"[data] Target class count decided: {actual_target}")
        
    for class_idx, cls_name in label_map.items():
        current_count = counts[cls_name]
        if current_count < actual_target:
            missing = actual_target - current_count
            print(f"  Augmenting {missing:4d} images for {cls_name} ...")
            cls_dir = data_dir / cls_name
            
            # Collect existing images to use as source for augmentation
            class_imgs = [p for p, l in samples if l == class_idx]
            if len(class_imgs) == 0:
                print(f"  Warning: No images found for {cls_name}, skipping.")
                continue
            
            for i in range(missing):
                src_path = class_imgs[i % len(class_imgs)]
                img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                if img is None: continue
                
                # Random Augmentations
                # 1. Flip
                flip_code = np.random.choice([-1, 0, 1, 2])
                if flip_code != 2:
                    img = cv2.flip(img, flip_code)
                
                # 2. Rotate
                angle = np.random.uniform(-45, 45)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                
                # 3. Brightness/Contrast
                alpha = np.random.uniform(0.8, 1.2) # Contrast
                beta = np.random.uniform(-30, 30)   # Brightness
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                
                save_path = cls_dir / f"aug_{i:05d}_{Path(src_path).name}"
                cv2.imwrite(str(save_path), img)
                
    print("[data] Offline Augmentation Complete!")

# ── CLI helper ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    samples, label_map = load_dataset(cfg.data_dir)
    train, val, test = split_dataset(samples, cfg)
    save_split_info(train, val, test, label_map, cfg.processed_dir, data_dir=cfg.data_dir)

    # Quick sanity: load & show one image
    sample_path, sample_label = train[0]
    img = preprocess_image(sample_path, cfg.image_size)
    print(f"[data] Sample shape: {img.shape}, dtype: {img.dtype}, "
          f"range: [{img.min():.2f}, {img.max():.2f}], "
          f"label: {label_map[sample_label]}")
