# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Short-side size distribution of labelled targets, with the large end called out explicitly.

The bucket stats in `build_mixed_dataset.py` use the longest side and stop at ">32px", which cannot show whether the
large end (>200px, i.e. a drone filling 40-80% of a 1920x1080 frame) exists at all. This script reports short-side
buckets plus how many images contain a target above 200px / 400px, so oversampling or synthesis decisions are based on
counts rather than on intuition.

Usage:
    python tools/ard_mav/stat_target_sizes.py \
        --images g:/0-rk3588/train/DUT-Anti-UAV/yolo_dataset/images/train \
        --labels g:/0-rk3588/train/DUT-Anti-UAV/yolo_dataset/labels/train \
        --images g:/0-rk3588/train/ARD-MAV/mixed_r3_640/images/train \
        --labels g:/0-rk3588/train/ARD-MAV/mixed_r3_640/labels/train
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

BUCKETS = ((0, 16), (16, 32), (32, 64), (64, 128), (128, 200), (200, 400), (400, 1e9))


def label_for(image: Path, labels_dir: Path) -> Path:
    """Label path for an image inside a `images/ -> labels/` layout."""
    return labels_dir / image.with_suffix(".txt").name


def main() -> None:
    """Print per-root short-side buckets and large-target image counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, action="append", required=True)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.images) != len(args.labels):
        raise SystemExit("--images and --labels must be given in pairs")

    for images_dir, labels_dir in zip(args.images, args.labels):
        counts = Counter()
        boxes = 0
        n_img = n_200 = n_400 = 0
        sides = []
        for image in sorted(images_dir.glob("*.jpg")):
            label = label_for(image, labels_dir)
            if not label.is_file():
                continue
            rows = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not rows:
                continue
            n_img += 1
            with Image.open(image) as im:
                width, height = im.size
            img_200 = img_400 = False
            for parts in rows:
                if len(parts) != 5:
                    continue
                _, _, _, bw, bh = (float(v) for v in parts)
                short = min(bw * width, bh * height)
                sides.append(short)
                boxes += 1
                lo, hi = next((lo, hi) for lo, hi in BUCKETS if lo <= short < hi)
                counts[f"{lo}-{hi}px" if hi < 1e9 else f">{lo}px"] += 1
                img_200 |= short > 200
                img_400 |= short > 400
            n_200 += img_200
            n_400 += img_400
        sides.sort()
        med = sides[len(sides) // 2] if sides else 0.0
        print(f"\n{images_dir.as_posix()}")
        print(f"  images with labels={n_img}  boxes={boxes}  median short side={med:.1f}px")
        for lo, hi in BUCKETS:
            key = f"{lo}-{hi}px" if hi < 1e9 else f">{lo}px"
            c = counts[key]
            print(f"  {key:>10}: {c:6d}  {c / max(boxes, 1):6.2%}")
        print(f"  images containing short side >200px: {n_200} ({n_200 / max(n_img, 1):.2%})")
        print(f"  images containing short side >400px: {n_400} ({n_400 / max(n_img, 1):.2%})")


if __name__ == "__main__":
    main()
