# 🛰️ AeroDamage — Building Damage Assessment

> A real-world computer vision system that classifies satellite imagery of disaster-affected buildings into four damage categories using transfer learning (ResNet50 + EfficientNet-B4).

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-xBD-orange)](https://xview2.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Damage Classes](#damage-classes)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to Run Inference](#how-to-run-inference)
- [How to Reproduce Training](#how-to-reproduce-training)
- [Model Weights](#model-weights)
- [Engineering Notes](#engineering-notes)
- [Dataset](#dataset)

---

## Overview

AeroDamage is part of a larger **AI Damage Detection System** that integrates three computer vision models:

| Model | Task | Architecture |
|-------|------|-------------|
| **AeroDamage** (this repo) | Building damage classification | ResNet50 + EfficientNet-B4 |
| Car Damage Detection | Bounding box detection | YOLOv8 |
| Road Damage Detection | Bounding box detection | YOLOv8 |

This module focuses on **post-disaster satellite imagery** — cropping individual building "chips" from full xBD scenes and classifying each building's damage level.

**Key engineering goals:**
- Maximize generalization on a severely imbalanced dataset (76.8% no-damage)
- Prove the performance ceiling is data-limited, not architecture-limited
- Apply production-grade anti-overfitting techniques

---

## Damage Classes

| Class | Description |
|-------|-------------|
| `no-damage` | Building intact, no visible structural change |
| `minor-damage` | Roof damage, minor cracks, debris around building |
| `major-damage` | Partial collapse, significant structural damage |
| `destroyed` | Complete collapse or total loss |

---

## Results

| Metric | ResNet50 | EfficientNet-B4 |
|--------|----------|-----------------|
| Test Accuracy | 64–68% | 66–70% |
| Macro F1 | ~0.55 | ~0.58 |
| Macro Precision | ~0.58 | ~0.61 |
| Macro Recall | ~0.53 | ~0.56 |

**Why not higher?** The bottleneck is the data, not the model. See [Engineering Notes](#engineering-notes).

---

## Project Structure

```
AeroDamage/
│
├── notebooks/
│   └── AeroDamage_Colab.ipynb     # Full training pipeline (run on Colab)
│
├── src/
│   ├── dataset.py                 # XBDDataset class + transforms
│   ├── model.py                   # ResNet50 model builder
│   ├── preprocess.py              # xBD → building chip extraction
│   ├── train.py                   # Training loop (CLI)
│   ├── evaluate.py                # Test set evaluation + charts
│   └── inference.py               # Single-image inference script
│
├── models/
│   ├── best_model.pth             # Trained weights (see below)
│   └── training_history.json      # Metrics from training run
│
├── data/
│   └── raw/                       # Place xBD dataset here (not tracked by git)
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
├── assets/
│   └── results/                   # Evaluation charts (auto-generated)
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       └── per_class_f1.png
│
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Setup

### Option A — pip (recommended for local dev)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/AeroDamage.git
cd AeroDamage

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate aerodamage
```

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended) — T4 or better for training
- ~50 GB disk space for the xBD dataset

---

## How to Run Inference

### On a single image

```bash
python src/inference.py \
  --image path/to/building_chip.png \
  --weights models/best_model.pth
```

**Example output:**
```
🛰️  AeroDamage Inference
──────────────────────────────
Image:       building_chip.png
Prediction:  minor-damage
Confidence:  73.4%

Class Probabilities:
  no-damage     :  18.2%
  minor-damage  :  73.4%  ◀
  major-damage  :   6.9%
  destroyed     :   1.5%
```

### On a folder of images

```bash
python src/inference.py \
  --folder path/to/chips/ \
  --weights models/best_model.pth \
  --output results.csv
```

---

## How to Reproduce Training

### Step 1 — Get the xBD dataset

1. Register at [xView2 Challenge](https://xview2.org/dataset)
2. Download the xBD dataset (train + test splits)
3. Place it in `data/raw/` following the structure above

### Step 2 — Preprocess (extract building chips)

```bash
python src/preprocess.py \
  --raw_dir data/raw \
  --out_dir data/processed
```

This reads every JSON label file, crops each building polygon, and saves chips into class folders. **Takes 15–30 minutes.** Output:

```
data/processed/
  train/  no-damage/  minor-damage/  major-damage/  destroyed/
  val/    no-damage/  minor-damage/  major-damage/  destroyed/
  test/   no-damage/  minor-damage/  major-damage/  destroyed/
```

### Step 3 — Train

```bash
python src/train.py \
  --data_dir data/processed \
  --model_dir models/ \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4
```

Or use the **Colab notebook** for GPU access:
> Open `notebooks/AeroDamage_Colab.ipynb` in Google Colab → Runtime → Change runtime type → T4 GPU → Run All

### Step 4 — Evaluate

```bash
python src/evaluate.py \
  --data_dir data/processed \
  --weights models/best_model.pth \
  --output_dir assets/results/
```

---

## Model Weights

Pre-trained weights are available here:

> 📥 **[Download best_model.pth](https://drive.google.com/YOUR_LINK_HERE)** (~100 MB)

Place the downloaded file at: `models/best_model.pth`

To use your own trained weights, follow [How to Reproduce Training](#how-to-reproduce-training).

---

## Engineering Notes

### Anti-Overfitting Techniques Applied

We applied 10 regularization strategies to push the model to its realistic performance limit:

| Technique | Details |
|-----------|---------|
| Hard class balancing | Capped no-damage class (76.8%) at 3× minority count |
| MixUp augmentation | Blended image pairs → smoother decision boundaries |
| Strong augmentation | Flips, rotations, perspective distortion, color jitter, cutout |
| Dropout (0.5 + 0.3) | Applied in classification head |
| Label smoothing (0.1) | Prevents overconfidence on training samples |
| Weight decay (AdamW) | L2 regularization, max grad norm = 1.0 |
| Early stopping | Patience = 5 epochs |
| OneCycleLR scheduler | Warmup → peak → cosine decay |
| Progressive unfreezing | Backbone unfrozen block-by-block at decreasing LRs |
| Fresh random subset/epoch | Model never saw same sample order twice |

### Why Accuracy Plateaued at 64–70%

Despite all optimizations, the ceiling is set by the **data**, not the architecture:

1. **Class imbalance** — 76.8% of chips are `no-damage`. Oversampling creates duplicates, not diversity.
2. **Domain shift** — xBD is predominantly North American disasters. Test images show different architecture and terrain.
3. **Label ambiguity** — Even human annotators only agree ~85% of the time on minor vs. major damage. This is the natural ceiling.

This is a real-world AI lesson: better architectures cannot compensate for insufficient or noisy data.

### Transfer Learning Strategy

**ResNet50:**
- Epochs 1–5: Frozen backbone → train head only
- Epoch 5+: Unfreeze `layer4` at `LR × 0.1`

**EfficientNet-B4:**
- Stage 1: Train head (highest LR)
- Stage 2: Unfreeze `blocks.6`
- Stage 3: Unfreeze `blocks.5`
- Stage 4: Unfreeze `blocks.4` (lowest LR = `LR × 0.01`)

---

## Dataset

- **Name:** xBD (xView2 Building Damage)
- **Source:** [xview2.org/dataset](https://xview2.org/dataset)
- **Size:** ~700K building chips after preprocessing
- **Classes:** 4 damage levels (severely imbalanced)
- **License:** xView2 Challenge License — see [xview2.org](https://xview2.org)

> ⚠️ The dataset is **not included** in this repository. You must register and download it separately.

---

## License

MIT License — see [LICENSE](LICENSE)
