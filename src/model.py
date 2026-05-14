"""
model.py — ResNet50 transfer learning model for AeroDamage.

Builds the model with a custom 4-class classification head.
Supports frozen backbone (warm-up phase) and progressive unfreezing.

Usage:
    from src.model import build_model, unfreeze_layer4

    model = build_model(num_classes=4, pretrained=True)
    # ... train head for 5 epochs ...
    unfreeze_layer4(model)
    # ... fine-tune with lower LR ...
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_CLASSES = 4  # no-damage / minor / major / destroyed


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet50 model with a custom classification head.

    Architecture:
        ResNet50 backbone (ImageNet pretrained, fully frozen initially)
            ↓
        Dropout(0.5)
            ↓
        Linear(2048 → 512)
            ↓
        ReLU
            ↓
        Dropout(0.3)
            ↓
        Linear(512 → num_classes)

    The backbone is frozen on creation. Call unfreeze_layer4() after
    the warm-up phase to enable fine-tuning of the deepest block.

    Args:
        num_classes: number of output classes (default 4)
        pretrained:  load ImageNet weights (strongly recommended)

    Returns:
        nn.Module ready to move to device
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze ALL backbone layers — only our new head will train initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final FC layer with our 4-class head
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=512, out_features=num_classes),
    )

    return model


def unfreeze_layer4(model: nn.Module) -> None:
    """
    Unfreeze ResNet50's layer4 (the deepest feature extractor block).

    Call this after the warm-up phase (typically epoch 5) to allow
    fine-tuning. Use a lower learning rate after calling this to avoid
    destroying the pretrained ImageNet representations.

    Args:
        model: the ResNet50 model returned by build_model()
    """
    for param in model.layer4.parameters():
        param.requires_grad = True
    print("  [model] Unfroze ResNet50 layer4 for fine-tuning.")


def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Load saved model weights from a .pth file.

    Args:
        model:        model instance (must match architecture used during training)
        weights_path: path to .pth file
        device:       torch device to map weights to

    Returns:
        model with loaded weights, set to eval mode
    """
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  [model] Loaded weights from {weights_path}")
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable vs. frozen parameters.

    Returns:
        dict with 'trainable', 'frozen', and 'total' counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
