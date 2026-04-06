"""
visualize.py — Graph Structure Visualization Tool for Foraminifera

Allows visualization of the exact Superpixel (Texture) and Keypoint (Structure)
graphs constructed for a given image, as requested for research reports.

Usage:
    python visualize.py <path_to_image>
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from config import Config
from data_preprocessing import preprocess_image
from graph_builder import build_graphs
from skimage.segmentation import slic

def draw_graph_overlay(image, coords, edge_index, color='cyan', radius=2, thickness=1):
    """
    Overlays nodes and edges onto a numpy image (0-255 RGB).
    coords: an (N, 2) array of (y, x) or (x, y) coordinates for nodes.
    edge_index: (2, E) array of edges.
    """
    # cv2 drawing expects BGR; we convert back to RGB before returning.
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Check if coords are (N, >=2). 
    # Usually we stored (cx, cy) in features. Let's assume coords is (x, y).
    
    # Draw edges
    if edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i].item())
            dst = int(edge_index[1, i].item())
            
            x1, y1 = int(coords[src, 0]), int(coords[src, 1])
            x2, y2 = int(coords[dst, 0]), int(coords[dst, 1])
            
            bgr_color = (0, 255, 255) if color == 'cyan' else (255, 100, 255)
            cv2.line(canvas, (x1, y1), (x2, y2), bgr_color, thickness, cv2.LINE_AA)
            
    # Draw nodes
    for i in range(coords.shape[0]):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        bgr_color = (0, 255, 255) if color == 'cyan' else (255, 100, 255)
        cv2.circle(canvas, (x, y), radius, bgr_color, -1)
        
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def visualize_graphs(image_path: str, save_path: str = None):
    cfg = Config()
    # Read the image and build graphs
    img = preprocess_image(image_path, cfg.image_size) # returns (H, W, 3) in [0, 1] RGB
    # Graph builder expects a label; for visualization we can use a dummy value.
    sp_graph, kp_graph = build_graphs(img, label=0, cfg=cfg)
    
    # Base Image in [0, 255] RGB for plotting
    img_disp = (img * 255).astype(np.uint8)
    h, w = img_disp.shape[:2]
    
    # ── 1. Superpixel Overlay ──────────────────────────────────────────────
    # We can use skimage.segmentation.mark_boundaries to show the actual SLIC shapes!
    segments = slic(
        img, n_segments=cfg.n_segments, compactness=cfg.compactness,
        start_label=0, channel_axis=2,
    )
    sp_boundaries = mark_boundaries(img, segments, color=(1, 1, 0), mode='thick') # Yellow boundaries
    sp_boundaries_disp = (sp_boundaries * 255).astype(np.uint8)
    
    # Extract node centers (cx, cy are at indices 4,5 in sp_graph.x)
    sp_coords = sp_graph.x[:, 4:6].cpu().numpy()
    sp_coords[:, 0] *= w
    sp_coords[:, 1] *= h
    
    # Draw the Region Adjacency Graph over it
    sp_overlay = draw_graph_overlay(sp_boundaries_disp, sp_coords, sp_graph.edge_index, color='cyan', radius=2)
    
    # ── 2. Keypoint Overlay ────────────────────────────────────────────────
    # Extract keypoint coords (x, y are at indices 0,1 in kp_graph.x)
    kp_coords = kp_graph.x[:, 0:2].cpu().numpy()
    kp_coords[:, 0] *= w
    kp_coords[:, 1] *= h
    kp_overlay = draw_graph_overlay(img_disp, kp_coords, kp_graph.edge_index, color='magenta', radius=3, thickness=2)
    
    # ── 3. Plotting ────────────────────────────────────────────────────────
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_disp)
    axes[0].set_title(f"Original Image\n({cfg.image_size}x{cfg.image_size})", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(sp_overlay)
    axes[1].set_title(f"Texture Graph (Superpixels)\n{sp_graph.num_nodes} nodes, {sp_graph.edge_index.shape[1]} edges", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(kp_overlay)
    axes[2].set_title(f"Structure Graph (Keypoints)\n{kp_graph.num_nodes} nodes, {kp_graph.edge_index.shape[1]} edges", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the Dual Graphs of a Foram Image")
    parser.add_argument("image_path", type=str, help="Path to the image to visualize")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the output plot")
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: {args.image_path} does not exist.")
        sys.exit(1)
        
    visualize_graphs(args.image_path, args.save)
