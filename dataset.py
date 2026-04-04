"""
dataset.py — PyG‑compatible dataset that pairs superpixel & keypoint graphs.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
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

    def _prepare(self):
        """Build graphs for all samples (with disk caching)."""
        needs_build = any(
            not self._cache_path(i).exists() for i in range(len(self.samples))
        )

        if needs_build:
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

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader
