"""
model.py — Dual GNN architecture for foraminifera classification.

Branch 1  →  SuperpixelGNN  (GraphSAGE)   →  128‑D embedding
Branch 2  →  KeypointGNN    (GAT)          →  128‑D embedding
Fusion    →  gated fusion  →  MLP classifier  →  num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv, GATConv, global_mean_pool, LayerNorm
)

from config import Config


# ═══════════════════════════════════════════════════════════════════════════
#  BRANCH 1  ─  Superpixel GNN  (GraphSAGE)
# ═══════════════════════════════════════════════════════════════════════════


class SuperpixelGNN(nn.Module):
    """
    Multi‑layer GraphSAGE on the superpixel (texture) graph.
    Outputs a fixed‑size graph‑level embedding.
    """

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(LayerNorm(hidden_dim))

        # Last conv → out_dim
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.bns.append(LayerNorm(out_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph‑level readout
        return global_mean_pool(x, batch)    # (batch_size, out_dim)


# ═══════════════════════════════════════════════════════════════════════════
#  BRANCH 2  ─  Keypoint GNN  (GAT)
# ═══════════════════════════════════════════════════════════════════════════


class KeypointGNN(nn.Module):
    """
    Multi‑layer GAT on the keypoint (structure) graph.
    Outputs a fixed‑size graph‑level embedding.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: in → hidden (multi‑head)
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True))
        self.bns.append(LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
            )
            self.bns.append(LayerNorm(hidden_dim))

        # Last layer → out_dim  (single head for clean output)
        self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False))
        self.bns.append(LayerNorm(out_dim))

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return global_mean_pool(x, batch)    # (batch_size, out_dim)


# ═══════════════════════════════════════════════════════════════════════════
#  DUAL GNN  (fusion model)
# ═══════════════════════════════════════════════════════════════════════════


class DualGNN(nn.Module):
    """
    Dual‑branch GNN that fuses superpixel and keypoint embeddings
    through concatenation → MLP classifier.

    Supports three modes for ablation studies:
        - ``hybrid``          (default)  — both branches
        - ``superpixel_only``            — only superpixel branch
        - ``keypoint_only``              — only keypoint branch
    """

    def __init__(self, num_classes: int, cfg: Config):
        super().__init__()
        self.mode = cfg.model_mode

        h = cfg.hidden_dim

        # Branches
        self.sp_gnn = SuperpixelGNN(
            in_dim=cfg.superpixel_feat_dim,
            hidden_dim=h, out_dim=h,
            num_layers=cfg.num_gnn_layers,
            dropout=cfg.dropout,
        )
        self.kp_gnn = KeypointGNN(
            in_dim=cfg.keypoint_feat_dim,
            hidden_dim=h, out_dim=h,
            num_layers=cfg.num_gnn_layers,
            heads=cfg.gat_heads,
            dropout=cfg.dropout,
        )

        # Gated fusion:
        # gate = sigmoid(MLP([sp_emb, kp_emb]))  -> (B, h)
        # fused = gate * sp_emb + (1-gate) * kp_emb
        self.gate = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.LayerNorm(h),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, num_classes),
        )

    def forward(self, sp_batch, kp_batch):
        """
        Parameters
        ----------
        sp_batch : torch_geometric.data.Batch  — batched superpixel graphs
        kp_batch : torch_geometric.data.Batch  — batched keypoint graphs

        Returns
        -------
        logits : (batch_size, num_classes)
        """
        out_sp = self.sp_gnn(sp_batch.x, sp_batch.edge_index, sp_batch.batch)
        out_kp = self.kp_gnn(kp_batch.x, kp_batch.edge_index, kp_batch.batch)

        if self.mode == "superpixel_only":
            fused = F.normalize(out_sp, p=2, dim=1)
        elif self.mode == "keypoint_only":
            fused = F.normalize(out_kp, p=2, dim=1)
        else: # Hybrid
            out_sp = F.normalize(out_sp, p=2, dim=1)
            out_kp = F.normalize(out_kp, p=2, dim=1)
            gate = self.gate(torch.cat([out_sp, out_kp], dim=-1))
            fused = gate * out_sp + (1.0 - gate) * out_kp

        return self.classifier(fused)
