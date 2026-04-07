"""
config.py — Central configuration for the Dual GNN Foraminifera Classification Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = None          # set via CLI or defaults to project_root/dataset/images
    processed_dir: Path = None     # cached graphs
    checkpoint_dir: Path = None    # saved models
    results_dir: Path = None       # evaluation outputs

    # ── Image preprocessing ───────────────────────────────────────────────
    image_size: int = 256

    # ── Superpixel graph ──────────────────────────────────────────────────
    n_segments: int = 80
    n_keypoints: int = 50
    compactness: float = 10.0
    # meanRGB(3), stdRGB(3), meanI, stdI, gradMean, gradStd, lbpMean, lbpStd, cx, cy
    superpixel_feat_dim: int = 14

    # ── Keypoint graph ────────────────────────────────────────────────────
    knn_k: int = 5
    # x, y, intensity, dist_to_center, patchMean, patchStd, patchGradMean, patchGradStd
    keypoint_feat_dim: int = 8

    # ── GNN model ─────────────────────────────────────────────────────────
    hidden_dim: int = 256
    num_gnn_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.3

    # ── Training ──────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    patience: int = 10             # early-stopping patience

    # ── Imbalance handling ────────────────────────────────────────────────
    use_weighted_loss: bool = True
    class_weight_power: float = 1.0
    max_class_weight: float = 5.0
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.0
    use_weighted_sampler: bool = False  # if True, disables DataLoader shuffle

    # ── Data split ────────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # ── LR scheduler ──────────────────────────────────────────────────────
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    lr_scheduler_min_lr: float = 1e-6

    # ── GAN specific params ───────────────────────────────────────────────
    gan_latent_dim: int = 100
    gan_hidden_dim: int = 64
    gan_lr: float = 0.0002
    gan_epochs: int = 100
    gan_batch_size: int = 32
    gan_n_critic: int = 5          # critic iterations per generator iteration
    gan_lambda_gp: float = 10.0    # gradient penalty weight

    # ── Mode (for ablation) ───────────────────────────────────────────────
    # "hybrid" | "superpixel_only" | "keypoint_only"
    model_mode: str = "hybrid"

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "endless_forams"
        if self.processed_dir is None:
            self.processed_dir = self.project_root / "processed"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"

        # Ensure directories exist
        for d in [self.processed_dir, self.checkpoint_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
