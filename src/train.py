"""
train.py — Training script for AeroDamage ResNet50 classifier.

Implements the full transfer learning pipeline:
  1. Load pretrained ResNet50 (frozen backbone)
  2. Train classification head for UNFREEZE_EPOCH epochs
  3. Unfreeze layer4 and fine-tune at a lower LR
  4. Save best weights based on validation loss

Usage:
    python src/train.py \
        --data_dir data/processed \
        --model_dir models/ \
        --epochs 20 \
        --batch_size 64 \
        --lr 1e-4
"""

import os
import json
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import XBDDataset, get_transforms, CLASS_DIRS, CLASS_NAMES
from model import build_model, unfreeze_layer4, count_parameters


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "epochs":         20,
    "batch_size":     64,
    "lr":             1e-4,
    "weight_decay":   1e-4,
    "unfreeze_epoch": 5,
    "num_workers":    4,
    "seed":           42,
}


# ── Training / validation helpers ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """Run one full epoch of training or validation. Returns (loss, accuracy)."""
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            if is_train:
                optimizer.zero_grad()

            out  = model(imgs)
            loss = criterion(out, lbls)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(lbls).sum().item()
            total      += lbls.size(0)

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Training on: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n📂 Loading datasets...")
    train_ds = XBDDataset(args.data_dir, "train", get_transforms("train"))
    val_ds   = XBDDataset(args.data_dir, "val",   get_transforms("val"))
    test_ds  = XBDDataset(args.data_dir, "test",  get_transforms("test"))

    # ── Weighted sampler (compensate for class imbalance) ─────────────────────
    labels       = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels, minlength=4).astype(float)
    class_wts    = 1.0 / (class_counts + 1e-6)
    sample_wts   = [class_wts[l] for l in labels]
    sampler      = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)

    print("\n📊 Class distribution (train):")
    for i, (cls, cnt) in enumerate(zip(CLASS_NAMES, class_counts)):
        pct = cnt / class_counts.sum() * 100
        print(f"   {cls:20s}: {int(cnt):7,}  ({pct:.1f}%)  weight={class_wts[i]:.4f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n🧠 Building ResNet50 model...")
    model = build_model(num_classes=4, pretrained=True).to(device)
    params = count_parameters(model)
    print(f"   Trainable params : {params['trainable']:,}")
    print(f"   Frozen params    : {params['frozen']:,}")

    # ── Loss: class-weighted cross-entropy ────────────────────────────────────
    wt_tensor = torch.tensor(
        class_wts / class_wts.sum() * 4, dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=wt_tensor, label_smoothing=0.1)

    # ── Optimizer (head only initially) ───────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val     = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze layer4 after warm-up and recreate optimizer
        if epoch == args.unfreeze_epoch:
            unfreeze_layer4(model)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 0.1,
                steps_per_epoch=len(train_loader),
                epochs=args.epochs - epoch + 1
            )

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        v_loss,  v_acc  = run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(v_acc)

        star = " ⭐" if v_loss < best_val else ""
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {v_loss:>8.4f}  {v_acc:>7.4f}  {elapsed:>5.0f}s{star}")

        if v_loss < best_val:
            best_val     = v_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(args.model_dir, "best_model.pth"))

    # ── Save training history ─────────────────────────────────────────────────
    with open(os.path.join(args.model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best val loss : {best_val:.4f}")
    print(f"   Weights saved : {args.model_dir}/best_model.pth")
    print(f"   History saved : {args.model_dir}/training_history.json")
    print(f"\n→ Run evaluation: python src/evaluate.py --weights {args.model_dir}/best_model.pth")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AeroDamage ResNet50")
    parser.add_argument("--data_dir",       default="data/processed")
    parser.add_argument("--model_dir",      default="models/")
    parser.add_argument("--epochs",         default=DEFAULTS["epochs"],         type=int)
    parser.add_argument("--batch_size",     default=DEFAULTS["batch_size"],     type=int)
    parser.add_argument("--lr",             default=DEFAULTS["lr"],             type=float)
    parser.add_argument("--weight_decay",   default=DEFAULTS["weight_decay"],   type=float)
    parser.add_argument("--unfreeze_epoch", default=DEFAULTS["unfreeze_epoch"], type=int)
    parser.add_argument("--num_workers",    default=DEFAULTS["num_workers"],    type=int)
    parser.add_argument("--seed",           default=DEFAULTS["seed"],           type=int)
    main(parser.parse_args())
