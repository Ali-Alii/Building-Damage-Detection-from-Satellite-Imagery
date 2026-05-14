"""
inference.py — Run AeroDamage inference on a single image or a folder.

Loads a trained ResNet50 and predicts building damage class with
per-class confidence scores.

Usage (single image):
    python src/inference.py \
        --image path/to/chip.png \
        --weights models/best_model.pth

Usage (folder of images → CSV):
    python src/inference.py \
        --folder path/to/chips/ \
        --weights models/best_model.pth \
        --output results.csv
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_transforms
from model import build_model, load_weights


# ── Constants ─────────────────────────────────────────────────────────────────

LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
LABEL_EMOJIS = ["✅", "🟡", "🟠", "🔴"]
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ── Core inference ────────────────────────────────────────────────────────────

def predict_image(model: torch.nn.Module, img_path: str, device: torch.device) -> dict:
    """
    Run inference on a single image file.

    Args:
        model:    loaded and eval-mode model
        img_path: path to any image file
        device:   torch device

    Returns:
        dict with keys:
            - 'label':       predicted class name
            - 'label_idx':   predicted class index (0–3)
            - 'confidence':  confidence for predicted class (0.0–1.0)
            - 'probs':       list of 4 softmax probabilities
    """
    transform = get_transforms("test")

    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "label":      LABELS[pred_idx],
        "label_idx":  pred_idx,
        "confidence": float(probs[pred_idx]),
        "probs":      probs.tolist(),
    }


def print_result(img_path: str, result: dict) -> None:
    """Pretty-print a single inference result to the terminal."""
    print(f"\n🛰️  AeroDamage Inference")
    print("─" * 40)
    print(f"Image      : {os.path.basename(img_path)}")
    print(f"Prediction : {LABEL_EMOJIS[result['label_idx']]}  {result['label']}")
    print(f"Confidence : {result['confidence'] * 100:.1f}%")
    print()
    print("Class Probabilities:")
    for i, (label, prob) in enumerate(zip(LABELS, result["probs"])):
        bar    = "█" * int(prob * 30)
        marker = "  ◀" if i == result["label_idx"] else ""
        print(f"  {label:20s} {prob * 100:5.1f}%  {bar}{marker}")


# ── Batch inference ───────────────────────────────────────────────────────────

def run_folder(model, folder: str, output_csv: str, device: torch.device) -> None:
    """
    Run inference on all images in a folder, write results to CSV.

    Args:
        model:      loaded model
        folder:     directory containing image files
        output_csv: path to write results.csv
        device:     torch device
    """
    images = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    if not images:
        print(f"⚠️  No images found in {folder}")
        return

    print(f"\n🗂️  Processing {len(images)} images from {folder}")

    rows = []
    for i, img_path in enumerate(images, 1):
        try:
            result = predict_image(model, img_path, device)
            rows.append({
                "filename":   os.path.basename(img_path),
                "prediction": result["label"],
                "confidence": f"{result['confidence'] * 100:.2f}%",
                **{f"prob_{l}": f"{p * 100:.2f}%" for l, p in zip(LABELS, result["probs"])},
            })
            print(f"  [{i:>4}/{len(images)}] {os.path.basename(img_path):40s} → {result['label']} ({result['confidence']*100:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  Error on {img_path}: {e}")

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✅ Results saved to {output_csv} ({len(rows)} images processed)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model once
    model = build_model(num_classes=4, pretrained=False)
    model = load_weights(model, args.weights, device)

    if args.image:
        # Single image mode
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        result = predict_image(model, args.image, device)
        print_result(args.image, result)

    elif args.folder:
        # Batch folder mode
        if not os.path.isdir(args.folder):
            raise FileNotFoundError(f"Folder not found: {args.folder}")
        out = args.output or "results.csv"
        run_folder(model, args.folder, out, device)

    else:
        print("❌ Please provide --image or --folder.")
        print("   Example: python src/inference.py --image chip.png --weights models/best_model.pth")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroDamage — Building Damage Inference")
    parser.add_argument("--weights", default="models/best_model.pth",
                        help="Path to trained .pth weights file")
    parser.add_argument("--image",  default=None,
                        help="Path to a single image file")
    parser.add_argument("--folder", default=None,
                        help="Path to a folder of images (batch mode)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path for batch mode (default: results.csv)")
    main(parser.parse_args())
