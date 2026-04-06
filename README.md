## Dual GNN Foraminifera Classification

This project builds **two graphs per image** (superpixels + keypoints) and trains a **dual-branch GNN** classifier (GraphSAGE + GAT) with a **gated fusion** head.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset format

Point `--data_dir` to a folder laid out like:

```text
data_dir/
  class_a/
    img1.jpg
    img2.png
  class_b/
    ...
```

### Run the pipeline

#### Full pipeline (preprocess → cache graphs → train → evaluate)

```bash
python main.py --mode full --data_dir endless_forams --epochs 50
```

Outputs:
- `processed/split_info.json`
- cached graphs: `processed/train/`, `processed/val/`, `processed/test/`
- checkpoint: `checkpoints/best_model.pt`
- plots: `results/`

#### Train only (uses existing split + cached graphs)

```bash
python main.py --mode train --data_dir endless_forams --epochs 50
```

#### Evaluate only

```bash
python main.py --mode evaluate --data_dir endless_forams --checkpoint checkpoints/best_model.pt
```

### GAN augmentation (optional)

Train:

```bash
python main.py --mode gan_train --data_dir endless_forams
```

Generate synthetic images until each class reaches `--target_count` images:

```bash
python main.py --mode gan_generate --data_dir endless_forams --target_count 500
```

### Smoke test (no real data required)

```bash
python main.py --mode full --test --epochs 2
```

### Visualize graphs for a single image

```bash
python visualize.py /path/to/image.jpg
python visualize.py /path/to/image.jpg --save results/graph_viz.png
```

### Notes / knobs worth tuning

- **Portability**: `processed/split_info.json` stores paths relative to `--data_dir` when possible.
- **Caching**: cache is validated via `processed/<split>/manifest.json` and rebuilt automatically when settings change.
- **Keypoints**: set `Config.keypoint_detector` to `"orb"` (default) or `"gftt"`.
- **Augmentation**: `Config.use_train_augmentation` applies train-only augmentation **once when building cached train graphs**.
