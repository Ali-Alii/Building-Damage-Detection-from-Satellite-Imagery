"""
evaluate.py — Evaluate a trained AeroDamage model on the test set.

Generates:
  - Classification report (per-class precision, recall, F1)
  - Confusion matrix (saved as PNG)
  - Training curves plot (if training_history.json is available)
  - Per-class F1 bar chart

Usage:
    python src/evaluate.py \
        --data_dir data/processed \
        --weights  models/best_model.pth \
        --output_dir assets/results/
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from dataset import XBDDataset, get_transforms, CLASS_NAMES
from model import build_model, load_weights


# ── Plot helpers ──────────────────────────────────────────────────────────────

DARK_BG   = "#0a0e1a"
PANEL_BG  = "#111827"
TEXT_CLR  = "#e2e8f0"
MUTED_CLR = "#8899bb"
GRID_CLR  = "#2a3a5c"


def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   PANEL_BG,
        "text.color":       TEXT_CLR,
        "axes.labelcolor":  TEXT_CLR,
        "xtick.color":      MUTED_CLR,
        "ytick.color":      MUTED_CLR,
    })


def plot_training_curves(history: dict, out_path: str) -> None:
    """Plot loss and accuracy curves from training_history.json."""
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    epochs_x = list(range(1, len(history["train_loss"]) + 1))

    # Loss
    ax = axes[0]
    ax.plot(epochs_x, history["train_loss"], color="#3b82f6", label="Train", linewidth=2)
    ax.plot(epochs_x, history["val_loss"],   color="#f97316", label="Val",   linewidth=2)
    ax.set_title("Training & Validation Loss", color=TEXT_CLR, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(facecolor="#1a2235", edgecolor=GRID_CLR)
    ax.grid(color=GRID_CLR, alpha=0.5)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs_x, [v * 100 for v in history["train_acc"]], color="#22c55e", label="Train", linewidth=2)
    ax.plot(epochs_x, [v * 100 for v in history["val_acc"]],   color="#a78bfa", label="Val",   linewidth=2)
    ax.set_title("Training & Validation Accuracy", color=TEXT_CLR, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(facecolor="#1a2235", edgecolor=GRID_CLR)
    ax.grid(color=GRID_CLR, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Plot and save the confusion matrix."""
    _apply_dark_style()
    labels_display = ["No Damage", "Minor", "Major", "Destroyed"]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels_display, color=MUTED_CLR)
    ax.set_yticklabels(labels_display, color=MUTED_CLR)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Confusion Matrix", color=TEXT_CLR, fontsize=13, fontweight="bold")
    threshold = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else MUTED_CLR, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_class_f1(report: dict, out_path: str) -> None:
    """Plot per-class F1 scores as a bar chart."""
    _apply_dark_style()
    labels_display = ["No Damage", "Minor", "Major", "Destroyed"]
    colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
    f1s    = [report[c]["f1-score"] * 100 for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    bars = ax.bar(labels_display, f1s, color=colors, edgecolor=GRID_CLR, linewidth=1.5, width=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax.set_ylabel("F1-Score (%)")
    ax.set_title("Per-Class F1 Score", color=TEXT_CLR, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=GRID_CLR, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n🧠 Loading model...")
    model = build_model(num_classes=4, pretrained=False)
    model = load_weights(model, args.weights, device)

    # ── Load test data ────────────────────────────────────────────────────────
    print("\n📂 Loading test set...")
    test_ds     = XBDDataset(args.data_dir, "test", get_transforms("test"))
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\n🔍 Running inference on test set...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Evaluating"):
            out = model(imgs.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    # ── Metrics ───────────────────────────────────────────────────────────────
    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report   = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    cm       = confusion_matrix(all_labels, all_preds)

    print(f"\n🎯 TEST ACCURACY: {test_acc * 100:.2f}%")
    print(f"   Macro F1        : {report['macro avg']['f1-score'] * 100:.2f}%")
    print(f"   Macro Precision : {report['macro avg']['precision'] * 100:.2f}%")
    print(f"   Macro Recall    : {report['macro avg']['recall'] * 100:.2f}%")
    print("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "test_accuracy":         test_acc,
            "classification_report": report,
            "confusion_matrix":      cm.tolist(),
        }, f, indent=2)
    print(f"  Saved: {results_path}")

    # ── Generate charts ───────────────────────────────────────────────────────
    print("\n📊 Generating charts...")
    history_path = os.path.join(os.path.dirname(args.weights), "training_history.json")
    if os.path.isfile(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))
    else:
        print(f"  ⚠️  training_history.json not found at {history_path} — skipping curves plot")

    plot_confusion_matrix(cm,     os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_per_class_f1(report,     os.path.join(args.output_dir, "per_class_f1.png"))
    print("\n✅ Evaluation complete!")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AeroDamage on test set")
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--weights",     default="models/best_model.pth")
    parser.add_argument("--output_dir",  default="assets/results/")
    parser.add_argument("--batch_size",  default=64,  type=int)
    parser.add_argument("--num_workers", default=4,   type=int)
    main(parser.parse_args())
