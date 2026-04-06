"""
graph_builder.py — Construct superpixel (texture) and keypoint (structure) graphs.
"""

import numpy as np
import cv2
import torch
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from config import Config


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _superpixel_adjacency(segments: np.ndarray):
    """
    Find pairs of adjacent superpixel IDs by checking 4‑connected pixel
    neighbours.  Returns a set of (u, v) tuples with u < v.
    """
    edges = set()
    h, w = segments.shape

    # Horizontal neighbours
    mask_h = segments[:, :-1] != segments[:, 1:]
    ys, xs = np.where(mask_h)
    for y, x in zip(ys, xs):
        a, b = int(segments[y, x]), int(segments[y, x + 1])
        edges.add((min(a, b), max(a, b)))

    # Vertical neighbours
    mask_v = segments[:-1, :] != segments[1:, :]
    ys, xs = np.where(mask_v)
    for y, x in zip(ys, xs):
        a, b = int(segments[y, x]), int(segments[y + 1, x])
        edges.add((min(a, b), max(a, b)))

    return edges


# ═══════════════════════════════════════════════════════════════════════════
#  SUPERPIXEL GRAPH  (texture / region‑level)
# ═══════════════════════════════════════════════════════════════════════════


def build_superpixel_graph(
    image: np.ndarray,
    label: int,
    n_segments: int = 100,
    compactness: float = 10.0,
) -> Data:
    """
    Build a graph where each node is a SLIC superpixel.

    Node features (14D):
        mean RGB (3), std RGB (3),
        mean intensity, std intensity,
        mean gradient magnitude, std gradient magnitude,
        mean LBP, std LBP,
        centroid_x, centroid_y

    Edges:
        Region‑adjacency (segments that share a boundary).

    Parameters
    ----------
    image : (H, W, 3) float32 array in [0, 1], RGB.
    label : integer class label.
    """
    h, w = image.shape[:2]

    # ── superpixel segmentation ───────────────────────────────────────────
    segments = slic(
        image, n_segments=n_segments, compactness=compactness,
        start_label=0, channel_axis=2,
    )
    seg_ids = np.unique(segments)
    n_nodes = len(seg_ids)
    # Build fast lookup: segment_id → node_index
    sid_to_idx = {int(sid): idx for idx, sid in enumerate(seg_ids)}

    # ── node features ─────────────────────────────────────────────────────
    gray_u8 = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # LBP on grayscale (uniform patterns)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform").astype(np.float32)
    if lbp.max() > 0:
        lbp = lbp / lbp.max()

    features = np.zeros((n_nodes, 14), dtype=np.float32)

    for idx, sid in enumerate(seg_ids):
        mask = segments == sid
        ys, xs = np.where(mask)

        rgb = image[mask]  # (N, 3)
        features[idx, 0:3] = rgb.mean(axis=0)                 # mean R,G,B
        features[idx, 3:6] = rgb.std(axis=0)                  # std R,G,B

        g = gray[mask]
        features[idx, 6] = float(g.mean())                    # mean intensity
        features[idx, 7] = float(g.std())                     # std intensity

        gm = grad[mask]
        features[idx, 8] = float(gm.mean())                   # mean grad magnitude
        features[idx, 9] = float(gm.std())                    # std grad magnitude

        lb = lbp[mask]
        features[idx, 10] = float(lb.mean())                  # mean LBP
        features[idx, 11] = float(lb.std())                   # std LBP

        features[idx, 12] = xs.mean() / w                     # centroid x (normalised)
        features[idx, 13] = ys.mean() / h                     # centroid y (normalised)

    # ── edges via boundary adjacency ──────────────────────────────────────
    adj_pairs = _superpixel_adjacency(segments)
    src, dst = [], []
    for u, v in adj_pairs:
        u_idx = sid_to_idx[u]
        v_idx = sid_to_idx[v]
        src += [u_idx, v_idx]      # undirected
        dst += [v_idx, u_idx]

    if len(src) == 0:
        # Fallback: fully-connected (very unlikely)
        src = [i for i in range(n_nodes) for j in range(n_nodes) if i != j]
        dst = [j for i in range(n_nodes) for j in range(n_nodes) if i != j]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


# ═══════════════════════════════════════════════════════════════════════════
#  KEYPOINT GRAPH  (structural)
# ═══════════════════════════════════════════════════════════════════════════


def build_keypoint_graph(
    image: np.ndarray,
    label: int,
    n_keypoints: int = 50,
    knn_k: int = 5,
) -> Data:
    """
    Build a graph where each node is a detected keypoint.

    Node features (8D):
        x, y (normalised), intensity, distance_to_image_centre,
        patch mean, patch std, patch gradient mean, patch gradient std

    Edges:
        k‑NN on spatial (x, y) coordinates.

    Parameters
    ----------
    image : (H, W, 3) float32 array in [0, 1], RGB.
    label : integer class label.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # ── detect keypoints ──────────────────────────────────────────────────
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=n_keypoints,
        qualityLevel=0.01, minDistance=5,
    )

    # Fallback: if too few keypoints, add random points
    if corners is None or len(corners) < 3:
        pts = np.random.rand(max(n_keypoints, 10), 1, 2).astype(np.float32)
        pts[:, :, 0] *= w
        pts[:, :, 1] *= h
        corners = pts

    corners = corners.squeeze(1)  # (N, 2)
    n_nodes = len(corners)

    # ── node features ─────────────────────────────────────────────────────
    cx, cy = w / 2.0, h / 2.0
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    features = np.zeros((n_nodes, 8), dtype=np.float32)
    r = 8  # patch radius (=> 17x17 window)

    for i, (px, py) in enumerate(corners):
        ix, iy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
        features[i, 0] = px / w                                    # normalised x
        features[i, 1] = py / h                                    # normalised y
        features[i, 2] = gray[iy, ix] / 255.0                      # intensity
        features[i, 3] = np.sqrt((px - cx)**2 + (py - cy)**2) / np.sqrt(cx**2 + cy**2)  # dist to centre

        y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        x0, x1 = max(0, ix - r), min(w, ix + r + 1)
        patch = gray_f[y0:y1, x0:x1]
        gpatch = grad[y0:y1, x0:x1]
        features[i, 4] = float(patch.mean())
        features[i, 5] = float(patch.std())
        features[i, 6] = float(gpatch.mean())
        features[i, 7] = float(gpatch.std())

    # ── k‑NN edges ────────────────────────────────────────────────────────
    coords = features[:, :2]
    k = min(knn_k, n_nodes - 1)
    if k < 1:
        k = 1

    adj = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    coo = adj.tocoo()

    src = coo.row.tolist() + coo.col.tolist()   # make undirected
    dst = coo.col.tolist() + coo.row.tolist()

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════


def build_graphs(
    image: np.ndarray, label: int, cfg: Config
) -> tuple:
    """Return (superpixel_data, keypoint_data) for one image."""
    sp = build_superpixel_graph(image, label, cfg.n_segments, cfg.compactness)
    kp = build_keypoint_graph(image, label, cfg.n_keypoints, cfg.knn_k)
    return sp, kp


# ── Quick sanity check ────────────────────────────────────────────────────

if __name__ == "__main__":
    img = np.random.rand(256, 256, 3).astype(np.float32)
    g1 = build_superpixel_graph(img, label=0)
    g2 = build_keypoint_graph(img, label=0)
    print(f"Superpixel graph → nodes: {g1.x.shape[0]}, feats: {g1.x.shape[1]}, edges: {g1.edge_index.shape[1]}")
    print(f"Keypoint graph   → nodes: {g2.x.shape[0]}, feats: {g2.x.shape[1]}, edges: {g2.edge_index.shape[1]}")
