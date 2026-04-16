# 🔬 Foraminifera Classification via Dual-Branch Graph Neural Networks

A deep learning pipeline that classifies **35 species of planktonic foraminifera** from microscopy images by converting each image into two complementary graph representations — a texture graph and a structure graph — and fusing them through a learnable gated-attention mechanism.

**Achieved 89% test accuracy and 89% macro-F1** on a highly imbalanced 35-class fine-grained classification task.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Technical Decisions](#key-technical-decisions)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

Traditional CNNs process images on rigid pixel grids, which can miss the irregular spatial relationships present in biological specimens like foraminifera — shell chamber boundaries, pore distributions, and surface ornamentation patterns. This project takes a different approach:

1. **Convert each image into two graphs** capturing different aspects of morphology
2. **Process each graph with a specialized GNN** branch
3. **Fuse the representations** with a learned gating mechanism
4. **Classify** the fused embedding into one of 35 species

The pipeline also addresses severe class imbalance (ratios up to 1:4000) through Focal Loss, inverse-frequency class weighting, offline data augmentation, and optional Conditional WGAN-GP synthesis.

---

## Architecture

```
Input Image (256×256 RGB)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  SLIC Superpixels   │      │  Harris Corner Keypoints │
│  (80 segments)      │      │  (50 keypoints)          │
└────────┬────────────┘      └────────┬────────────────┘
         ▼                            ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  Texture Graph      │      │  Structure Graph         │
│  14-D node features │      │  8-D node features       │
│  Region-adjacency   │      │  k-NN spatial edges      │
│  edges              │      │                          │
└────────┬────────────┘      └────────┬────────────────┘
         ▼                            ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  SuperpixelGNN      │      │  KeypointGNN             │
│  (GraphSAGE, 3 layers)│   │  (EdgeConv/DGCNN, 3 layers)│
│  → 256-D embedding  │      │  → 256-D embedding       │
└────────┬────────────┘      └────────┬────────────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
         ┌─────────────────────┐
         │  Gated Fusion       │
         │  gate = σ(MLP([sp‖kp]))│
         │  fused = g·sp + (1-g)·kp│
         └────────┬────────────┘
                  ▼
         ┌─────────────────────┐
         │  MLP Classifier     │
         │  → 35 species       │
         └─────────────────────┘
```

### Graph Representations

| Graph | Nodes | Features | Edges |
|-------|-------|----------|-------|
| **Superpixel (Texture)** | ~80 SLIC segments | 14-D: mean/std RGB, mean/std intensity, mean/std gradient magnitude, mean/std LBP, centroid (x,y) | Region adjacency (shared boundaries) |
| **Keypoint (Structure)** | ~50 Harris corners | 8-D: normalized (x,y), intensity, distance-to-center, patch mean/std, patch gradient mean/std | k-NN on spatial coordinates (k=5) |

### GNN Branches

- **SuperpixelGNN**: 3-layer GraphSAGE with LayerNorm and dropout → global mean pooling → 256-D embedding
- **KeypointGNN**: 3-layer EdgeConv (DGCNN) with max aggregation, LayerNorm and dropout → global mean pooling → 256-D embedding

### Fusion

A **gated attention mechanism** learns per-sample weights:
```
gate = sigmoid(Linear(ReLU(LayerNorm(Linear(concat(sp, kp))))))
fused = gate * sp_embedding + (1 - gate) * kp_embedding
```

This allows the model to dynamically decide whether texture or structure is more informative for each input.

---

## Project Structure

```
.
├── config.py               # Central configuration (dataclass with all hyperparameters)
├── data_preprocessing.py   # Image loading, stratified splitting, offline augmentation
├── graph_builder.py        # SLIC superpixel graph + Harris keypoint graph construction
├── dataset.py              # PyG dataset with manifest-validated disk caching, dual collation
├── model.py                # SuperpixelGNN (GraphSAGE), KeypointGNN (EdgeConv), DualGNN (gated fusion)
├── train.py                # FocalLoss, class weighting, ReduceLROnPlateau, early stopping
├── evaluate.py             # Metrics, confusion matrix, training curves, ablation comparison
├── visualize.py            # Graph overlay visualization on source images
├── gan_model.py            # Conditional WGAN-GP generator + critic for 256×256 synthesis
├── gan_train.py            # GAN training loop + per-class synthetic image generation
├── main.py                 # CLI entry point with 8 execution modes
├── requirements.txt        # Dependencies
└── .gitignore
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/foram-dual-gnn.git
cd foram-dual-gnn

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyTorch
- PyTorch Geometric (SAGEConv, EdgeConv, GATConv)
- OpenCV
- scikit-image (SLIC segmentation, LBP)
- scikit-learn
- Matplotlib & Seaborn
- tqdm

---

## Usage

### Dataset Format

Organize your images in a folder-per-class layout:

```
endless_forams/
├── Globigerinoides_ruber/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── Globorotalia_menardii/
│   └── ...
└── ... (35 class folders)
```

### Running the Pipeline

```bash
# Full pipeline: preprocess → train → evaluate
python main.py --mode full --epochs 50

# Individual steps
python main.py --mode preprocess
python main.py --mode train --epochs 30
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

# Ablation study (trains superpixel_only, keypoint_only, hybrid)
python main.py --mode ablation --epochs 30

# Balance dataset via offline augmentation (auto-scales to max class size)
python main.py --mode augment_offline

# Train the Conditional WGAN-GP
python main.py --mode gan_train

# Generate synthetic images for minority classes
python main.py --mode gan_generate --target_count 500

# Smoke test with synthetic data
python main.py --mode full --test --epochs 2

# Custom configuration
python main.py --mode full --data_dir /path/to/images --epochs 100 --lr 0.0005 --batch_size 16
```

### Visualizing Graphs

```bash
# Display the dual-graph overlay for a single image
python visualize.py path/to/image.jpg

# Save the visualization
python visualize.py path/to/image.jpg --save output.png
```

This produces a 3-panel figure: original image | superpixel texture graph | keypoint structure graph.

---

## Results

### Performance Progression

| Stage | Accuracy | Macro-F1 | Key Change |
|-------|----------|----------|------------|
| Baseline | 52% | 0.35 | Vanilla CrossEntropyLoss, hidden_dim=128 |
| + Focal Loss & class weights | 62% | 0.67 | Focal Loss (γ=2.0), inverse-frequency weighting |
| + Full rebalancing & EdgeConv & wider model | **89%** | **0.89** | Offline augmentation to max class, EdgeConv, hidden_dim=256, class_weight_power=1.0 |

### Outputs

The pipeline automatically generates:
- `results/training_curves.png` — Loss and accuracy curves
- `results/confusion_matrix_hybrid.png` — Per-class confusion heatmap
- `results/ablation_comparison.png` — Bar chart comparing all three model modes
- `checkpoints/best_model.pt` — Best model checkpoint (by validation accuracy)

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Graphs over CNNs** | Graphs capture irregular spatial relationships (shell boundaries, pore structures) that rigid CNN grids miss |
| **EdgeConv over GAT for keypoints** | EdgeConv computes `h(xᵢ, xⱼ − xᵢ)`, making it inherently translation-invariant — ideal for spatial keypoint coordinates |
| **Gated fusion over concatenation** | Some species are best distinguished by texture (color), others by structure (chamber shape). The gate lets the model decide per-sample |
| **Focal Loss + class weights** | Focal Loss down-weights easy high-confidence examples (γ=2.0); class weights up-weight rare classes. Together they provide sample-level and class-level rebalancing |
| **Disk-cached PyG graphs** | SLIC segmentation + feature extraction is expensive. Caching with manifest validation avoids rebuilding graphs across epochs while detecting stale caches |

---

## Configuration

All hyperparameters are centralized in `config.py`. Key settings:

```python
# GNN Architecture
hidden_dim = 256          # Embedding dimension for both branches
num_gnn_layers = 3        # Depth of GraphSAGE / EdgeConv stacks
gat_heads = 4             # (kept for API compatibility)
dropout = 0.3

# Graph Construction
n_segments = 80           # SLIC superpixel count
n_keypoints = 50          # Harris corner max count
knn_k = 5                 # k-NN neighbors for keypoint graph

# Training
lr = 1e-3
epochs = 50
patience = 10             # Early stopping patience
use_focal_loss = True     # Focal Loss (γ=2.0)
use_weighted_loss = True  # Inverse-frequency class weighting
class_weight_power = 1.0  # Weighting aggressiveness
use_lr_scheduler = True   # ReduceLROnPlateau

# GAN
gan_latent_dim = 100
gan_epochs = 100
gan_n_critic = 5          # Critic steps per generator step
gan_lambda_gp = 10.0      # Gradient penalty coefficient
```

---

## License

This project was developed as part of a Deep Learning course (6th Semester). Feel free to use and adapt for academic purposes.
