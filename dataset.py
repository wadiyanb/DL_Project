"""
dataset.py — PyG‑compatible dataset that pairs superpixel & keypoint graphs.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import Config
from data_preprocessing import preprocess_image
from graph_builder import build_graphs


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════


class ForamGraphDataset(Dataset):
    """
    Each sample is a pair: (superpixel_graph, keypoint_graph).

    On first access the graphs are built from raw images and cached to disk
    under ``processed_dir``. Subsequent loads read directly from cache.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        cfg: Config,
        split_name: str = "train",
    ):
        super().__init__()
        self.samples = samples
        self.cfg = cfg
        self.split_name = split_name
        self.cache_dir = cfg.processed_dir / split_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build / load graphs
        self.sp_graphs: List[Data] = []
        self.kp_graphs: List[Data] = []
        self._prepare()

    # ── internal ──────────────────────────────────────────────────────────

    def _cache_path(self, idx: int) -> Path:
        return self.cache_dir / f"sample_{idx}.pt"

    def _manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def _load_manifest(self) -> Optional[dict]:
        mp = self._manifest_path()
        if not mp.exists():
            return None
        try:
            with open(mp, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_manifest(self):
        info = {
            "split": self.split_name,
            "image_size": int(self.cfg.image_size),
            "n_segments": int(self.cfg.n_segments),
            "n_keypoints": int(self.cfg.n_keypoints),
            "knn_k": int(self.cfg.knn_k),
            "compactness": float(self.cfg.compactness),
            "superpixel_feat_dim": int(self.cfg.superpixel_feat_dim),
            "keypoint_feat_dim": int(self.cfg.keypoint_feat_dim),
            "samples": [{"path": str(Path(p).resolve()), "label": int(l)} for p, l in self.samples],
        }
        with open(self._manifest_path(), "w") as f:
            json.dump(info, f, indent=2)

    def _manifest_matches(self) -> bool:
        m = self._load_manifest()
        if not m:
            return False
        if m.get("split") != self.split_name:
            return False
        for k in ("image_size", "n_segments", "n_keypoints", "knn_k"):
            if int(m.get(k, -1)) != int(getattr(self.cfg, k)):
                return False
        if float(m.get("compactness", -1)) != float(self.cfg.compactness):
            return False
        if int(m.get("superpixel_feat_dim", -1)) != int(self.cfg.superpixel_feat_dim):
            return False
        if int(m.get("keypoint_feat_dim", -1)) != int(self.cfg.keypoint_feat_dim):
            return False

        old = m.get("samples", [])
        if len(old) != len(self.samples):
            return False
        for (p, l), rec in zip(self.samples, old):
            if str(Path(p).resolve()) != rec.get("path"):
                return False
            if int(l) != int(rec.get("label")):
                return False
        return True

    def _clear_cache_files(self):
        for p in self.cache_dir.glob("sample_*.pt"):
            try:
                p.unlink()
            except Exception:
                pass
        mp = self._manifest_path()
        if mp.exists():
            try:
                mp.unlink()
            except Exception:
                pass

    def _prepare(self):
        """Build graphs for all samples (with disk caching)."""
        # Cache validity: ensure cached indices correspond to the same sample list.
        needs_build = (not self._manifest_matches()) or any(
            not self._cache_path(i).exists() for i in range(len(self.samples))
        )

        if needs_build:
            if not self._manifest_matches():
                print(f"[dataset:{self.split_name}] Cache mismatch detected; rebuilding cache …")
                self._clear_cache_files()
            print(f"[dataset:{self.split_name}] Building graphs for "
                  f"{len(self.samples)} images …")
            for idx, (path, label) in enumerate(
                tqdm(self.samples, desc=f"  {self.split_name}")
            ):
                cache_file = self._cache_path(idx)
                if cache_file.exists():
                    pair = torch.load(cache_file, weights_only=False)
                    self.sp_graphs.append(pair["sp"])
                    self.kp_graphs.append(pair["kp"])
                    continue

                img = preprocess_image(path, self.cfg.image_size)
                sp, kp = build_graphs(img, label, self.cfg)
                torch.save({"sp": sp, "kp": kp}, cache_file)
                self.sp_graphs.append(sp)
                self.kp_graphs.append(kp)
            self._write_manifest()
        else:
            print(f"[dataset:{self.split_name}] Loading cached graphs …")
            for idx in range(len(self.samples)):
                pair = torch.load(self._cache_path(idx), weights_only=False)
                self.sp_graphs.append(pair["sp"])
                self.kp_graphs.append(pair["kp"])

    # ── public API ────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.sp_graphs[idx], self.kp_graphs[idx]


# ═══════════════════════════════════════════════════════════════════════════
#  COLLATION & DATALOADER
# ═══════════════════════════════════════════════════════════════════════════


def _dual_collate(batch):
    """Custom collate: batch superpixel and keypoint graphs separately."""
    sp_list, kp_list = zip(*batch)
    sp_batch = Batch.from_data_list(list(sp_list))
    kp_batch = Batch.from_data_list(list(kp_list))
    return sp_batch, kp_batch


def get_dataloaders(
    train_samples: list,
    val_samples: list,
    test_samples: list,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build ForamGraphDatasets and wrap them in DataLoaders.
    """
    train_ds = ForamGraphDataset(train_samples, cfg, split_name="train")
    val_ds = ForamGraphDataset(val_samples, cfg, split_name="val")
    test_ds = ForamGraphDataset(test_samples, cfg, split_name="test")

    common = dict(collate_fn=_dual_collate, num_workers=0, pin_memory=True)

    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        # Weight each sample inversely proportional to its class frequency.
        labels = [int(l) for _, l in train_samples]
        if labels:
            counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
            counts = torch.clamp(counts, min=1)
            class_w = 1.0 / counts.float()
            sample_w = torch.tensor([class_w[l].item() for l in labels], dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights=sample_w,
                num_samples=len(sample_w),
                replacement=True,
            )
            shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader
