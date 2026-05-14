"""
dataset.py — XBD Dataset loader and transforms for AeroDamage.

Usage:
    from src.dataset import XBDDataset, get_transforms

    train_ds = XBDDataset(root="data/processed", split="train",
                          transform=get_transforms("train"))
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────────────────

# Folder names match what preprocess.py creates
CLASS_DIRS = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Human-readable names used in reports and charts
CLASS_NAMES = ["no_damage", "minor_damage", "major_damage", "destroyed"]

# ImageNet normalization stats (used because we start from pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224  # Input resolution expected by ResNet50 / EfficientNet-B4


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline.

    Training augmentations are stronger to improve generalization on
    the small, imbalanced xBD dataset.

    Args:
        split: one of "train", "val", or "test"

    Returns:
        A torchvision Compose pipeline
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Validation / test: no augmentation, just resize + normalize
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class XBDDataset(Dataset):
    """
    PyTorch Dataset for the xBD building chip dataset.

    Expected folder structure (created by preprocess.py):
        root/
            train/
                no-damage/       *.png
                minor-damage/    *.png
                major-damage/    *.png
                destroyed/       *.png
            val/  (same structure)
            test/ (same structure)

    Args:
        root:      Path to data/processed/
        split:     "train", "val", or "test"
        transform: torchvision transform pipeline (use get_transforms())
    """

    def __init__(self, root: str, split: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        split_dir = os.path.join(root, split)

        for label_idx, cls in enumerate(CLASS_DIRS):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

        print(f"  {split:5s}: {len(self.samples):>8,} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
