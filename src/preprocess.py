"""
preprocess.py — Extract building chips from the xBD satellite dataset.

Reads full xBD scene images + JSON polygon labels, crops each building
into a small image chip, and saves chips into class-labeled folders.

Run time: 15–30 minutes on a standard laptop.

Usage:
    python src/preprocess.py --raw_dir data/raw --out_dir data/processed

Output structure:
    data/processed/
        train/  no-damage/  minor-damage/  major-damage/  destroyed/
        val/    no-damage/  minor-damage/  major-damage/  destroyed/
        test/   no-damage/  minor-damage/  major-damage/  destroyed/
"""

import os
import json
import re
import random
import shutil
import argparse
from PIL import Image
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

VAL_SPLIT   = 0.15   # Fraction of training chips moved to val/
PADDING     = 10     # Pixels of padding around each building bounding box
MIN_SIZE    = 10     # Skip chips smaller than this (pixels) in any dimension
RANDOM_SEED = 42

CLASS_DIRS = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Maps xBD JSON subtypes → output folder names (None = skip)
XBD_LABEL_MAP = {
    "no-damage":     "no-damage",
    "minor-damage":  "minor-damage",
    "major-damage":  "major-damage",
    "destroyed":     "destroyed",
    "un-classified": None,   # Ambiguous labels are discarded
}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def get_bbox(coords: list) -> tuple:
    """Compute axis-aligned bounding box from polygon coordinates."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def crop_building(img: Image.Image, coords: list) -> Image.Image | None:
    """
    Crop a building chip from a scene image using polygon coordinates.

    Adds PADDING around the bounding box and clips to image bounds.
    Returns None if the resulting chip is too small.
    """
    x0, y0, x1, y1 = get_bbox(coords)
    x0 = max(0, int(x0) - PADDING)
    y0 = max(0, int(y0) - PADDING)
    x1 = min(img.width,  int(x1) + PADDING)
    y1 = min(img.height, int(y1) + PADDING)
    if (x1 - x0) < MIN_SIZE or (y1 - y0) < MIN_SIZE:
        return None
    return img.crop((x0, y0, x1, y1))


def parse_coords(feature: dict) -> list | None:
    """
    Extract polygon coordinates from an xBD GeoJSON feature dict.

    xBD labels use either a 'geometry' GeoJSON field or a 'wkt' string.
    This function handles both formats.
    """
    geom = feature.get("geometry", {})
    if geom and geom.get("type") == "Polygon":
        return geom["coordinates"][0]

    # Fallback: parse WKT (Well-Known Text) polygon string
    wkt = feature.get("wkt", "")
    if wkt:
        match = re.search(r"POLYGON\s*\(\((.*?)\)\)", wkt)
        if match:
            return [
                [float(v) for v in pt.strip().split()]
                for pt in match.group(1).split(",")
            ]
    return None


# ── Processing ────────────────────────────────────────────────────────────────

def process_split(raw_split: str, out_split: str) -> list:
    """
    Process one xBD split (train or test): crop all building chips and save.

    Args:
        raw_split: path to raw split dir (contains images/ and labels/)
        out_split: path to processed output dir

    Returns:
        List of (chip_path, class_name) tuples for the split-val step
    """
    images_dir = os.path.join(raw_split, "images")
    labels_dir = os.path.join(raw_split, "labels")

    # Create output class folders
    for cls in CLASS_DIRS:
        os.makedirs(os.path.join(out_split, cls), exist_ok=True)

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".json"))
    saved, skipped = 0, 0
    all_chips = []

    for lf in tqdm(label_files, desc=os.path.basename(raw_split)):
        stem = lf.replace(".json", "")

        # Find the matching satellite image (xBD uses various extensions)
        img_path = None
        for ext in [".png", ".tif", ".tiff", ".jpg"]:
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if not img_path:
            continue

        try:
            scene = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        with open(os.path.join(labels_dir, lf)) as f:
            data = json.load(f)

        for i, feat in enumerate(data.get("features", {}).get("xy", [])):
            props  = feat.get("properties", {})
            damage = props.get("subtype", "un-classified")
            cls    = XBD_LABEL_MAP.get(damage)
            if cls is None:
                skipped += 1
                continue

            coords = parse_coords(feat)
            if not coords:
                skipped += 1
                continue

            chip = crop_building(scene, coords)
            if chip is None:
                skipped += 1
                continue

            fname    = f"{stem}_bld{i:04d}.png"
            out_path = os.path.join(out_split, cls, fname)
            chip.save(out_path, "PNG")
            all_chips.append((out_path, cls))
            saved += 1

    print(f"  → Saved {saved:,} chips  ({skipped:,} skipped)")
    return all_chips


def carve_val_split(train_chips: list, out_dir: str) -> None:
    """
    Move VAL_SPLIT fraction of training chips into val/ per class.

    Shuffles chips randomly before splitting to avoid order bias.
    """
    val_dir = os.path.join(out_dir, "val")
    for cls in CLASS_DIRS:
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    by_cls = {cls: [] for cls in CLASS_DIRS}
    for path, cls in train_chips:
        by_cls[cls].append(path)

    moved = 0
    for cls, paths in by_cls.items():
        random.shuffle(paths)
        n = max(1, int(len(paths) * VAL_SPLIT))
        for p in paths[:n]:
            dst = os.path.join(val_dir, cls, os.path.basename(p))
            shutil.move(p, dst)
            moved += 1

    print(f"  → Moved {moved:,} chips to val/")


def print_summary(out_dir: str) -> None:
    """Print a breakdown of chips per class per split."""
    print("\n📊 Dataset Summary:")
    for split in ["train", "val", "test"]:
        print(f"  {split.upper()}:")
        for cls in CLASS_DIRS:
            folder = os.path.join(out_dir, split, cls)
            n = len(os.listdir(folder)) if os.path.isdir(folder) else 0
            bar = "█" * min(30, n // 1000)
            print(f"    {cls:20s}: {n:7,}  {bar}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract xBD building chips")
    parser.add_argument("--raw_dir", default="data/raw",       help="Path to raw xBD dataset")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--seed",    default=RANDOM_SEED, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    # Skip if already done (re-running wastes 30 min)
    already_done = (
        os.path.isdir(os.path.join(args.out_dir, "train", "no-damage"))
        and len(os.listdir(os.path.join(args.out_dir, "train", "no-damage"))) > 100
    )
    if already_done:
        print("⏩ Preprocessing already done — skipping. Delete data/processed/ to redo.")
        print_summary(args.out_dir)
        return

    print("\n[1/3] Processing TRAIN split...")
    train_chips = process_split(
        raw_split=os.path.join(args.raw_dir, "train"),
        out_split=os.path.join(args.out_dir, "train"),
    )

    print("\n[2/3] Carving out VAL split (15% of train)...")
    carve_val_split(train_chips, args.out_dir)

    print("\n[3/3] Processing TEST split...")
    process_split(
        raw_split=os.path.join(args.raw_dir, "test"),
        out_split=os.path.join(args.out_dir, "test"),
    )

    print_summary(args.out_dir)
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
