# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build a tiny-object-only INT8 calibration list for ARD-MAV.

Only training images whose every box has a longest side <= ``--max-side`` pixels qualify, so RKNN quantization
ranges are observed on the small-target regime instead of generic background frames.

Usage:
    python tools/ard_mav/make_calib_dataset.py --num 400
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def read_labels(label_file: Path, width: int, height: int) -> list[tuple[float, float]]:
    """Return box ``(w, h)`` sizes in pixels for one YOLO label file."""
    sizes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        sizes.append((float(parts[3]) * width, float(parts[4]) * height))
    return sizes


def main() -> None:
    """Select small-target images and write the RKNN calibration list + manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--num", type=int, default=400)
    parser.add_argument("--max-side", type=float, default=16.0, help="all boxes must be <= this longest side (px)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    images = sorted((args.root / "images" / args.split).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images under {args.root / 'images' / args.split}")

    eligible, tiny_boxes, rejected = [], 0, 0
    for image in images:
        label = args.root / "labels" / args.split / f"{image.stem}.txt"
        if not label.is_file():
            continue
        sizes = read_labels(label, args.width, args.height)
        if not sizes:  # keep negatives out: calibration must see tiny targets
            rejected += 1
            continue
        if max(max(w, h) for w, h in sizes) <= args.max_side:
            eligible.append(image)
            tiny_boxes += len(sizes)
        else:
            rejected += 1

    random.seed(args.seed)
    chosen = random.sample(eligible, min(args.num, len(eligible)))
    chosen.sort()
    out_dir = args.root / "calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = out_dir / "dataset.txt"
    dataset_file.write_text("".join(f"{p.resolve().as_posix()}\n" for p in chosen), encoding="utf-8")
    data = yaml.safe_load((args.root / "data.yaml").read_text(encoding="utf-8"))
    (out_dir / "data.yaml").write_text(
        "# INT8 calibration entry: both splits are the curated tiny-target list\n"
        f"path: {args.root.as_posix()}\n"
        "train: calib/dataset.txt\n"
        "val: calib/dataset.txt\n"
        f"nc: {data['nc']}\n"
        f"names: {data['names']}\n",
        encoding="utf-8",
    )
    manifest = {
        "criterion": f"every box longest side <= {args.max_side}px, at least one box",
        "split": args.split,
        "candidates": len(eligible),
        "selected": len(chosen),
        "tiny_boxes_in_candidates": tiny_boxes,
        "rejected_images": rejected,
        "seed": args.seed,
        "images": [p.name for p in chosen],
    }
    (out_dir / "calib_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"eligible={len(eligible)} rejected={rejected} selected={len(chosen)} -> {dataset_file.as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
