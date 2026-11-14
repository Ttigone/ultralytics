# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generate a frame-aligned DoG channel from the sequence-level split lists (round-2 stage 2b).

The DoG operator is imported read-only from ``tools/mrf_preprocess/filters.py`` (sigma1=1.0, sigma2=3.0, the setting that
won the high-recall comparison for 7-9 px targets). Labels are never rewritten, so every box keeps its original
coordinates.

Two output kinds:

- ``dog``     : one 8-bit PNG per frame (signed DoG mapped to +128), same stem as the source image. Feed it to a 4-channel
                loader if one exists; also useful as a visual/label-audit image.
- ``compose`` : a drop-in YOLO dataset where the stored B channel is the DoG map and G/R are the original channels. The
                Ultralytics loader converts BGR -> RGB, so DoG lands in model channel 0. Useful for a 3-channel run without
                touching the network.

Usage:
    python tools/ard_mav/make_dog_channel.py --mode dog --dst G:/0-rk3588/train/ARD-MAV/ARD-MAV-dog
    python tools/ard_mav/make_dog_channel.py --mode compose --dst G:/0-rk3588/train/ARD-MAV/ARD-MAV-dog-rgb
    python tools/ard_mav/make_dog_channel.py --mode dog --dst ... --limit 120   # timing/size sample
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mrf_preprocess"))  # read-only import of the existing DoG implementation
import filters as F  # noqa: E402

SPLITS_DIR = Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/splits")


def label_path(image: Path) -> Path:
    """Map an image to its YOLO label path (last /images/ -> /labels/, extension -> .txt)."""
    return Path(image.as_posix().replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def read_split(path: Path, limit: int) -> list[Path]:
    """Read one split .txt and optionally truncate it for a timing sample."""
    images = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return images[:limit] if limit else images


def dog_u8(image: Path, s1: float, s2: float, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """Load BGR, run DoG on luma and map the signed residual to uint8 (+128)."""
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"unreadable image: {image}")
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.clip(F.dog(luma, s1=s1, s2=s2) * gain + 128.0, 0, 255).astype(np.uint8), bgr


def main() -> None:
    """Write the DoG maps (or the composed dataset) and print real timing and disk numbers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--split-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--mode", choices=("dog", "compose"), default="dog")
    parser.add_argument("--s1", type=float, default=1.0, help="inner Gaussian sigma (report optimum)")
    parser.add_argument("--s2", type=float, default=3.0, help="outer Gaussian sigma (report optimum)")
    parser.add_argument("--gain", type=float, default=1.0, help="contrast gain before uint8 mapping")
    parser.add_argument("--const", action="store_true", help="write a flat 128 channel instead of DoG (control arm)")
    parser.add_argument("--limit", type=int, default=0, help="max images per split (0 = all)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started, totals = time.time(), {"images": 0, "bytes": 0}
    manifest = []
    for split in args.splits.split(","):
        images = read_split(args.split_dir / f"{split}.txt", args.limit)
        if not images:
            raise SystemExit(f"empty split: {args.split_dir / f'{split}.txt'}")
        out_dir = args.dst / ("dog" if args.mode == "dog" else "images") / split
        label_dir = args.dst / "labels" / split
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.mode == "compose":
                label_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for image in images:
            dog, bgr = dog_u8(image, args.s1, args.s2, args.gain)
            flat = np.full(dog.shape, 128, np.uint8)
            channel = flat if args.const else dog
            if not args.dry_run:
                if args.mode == "dog":
                    out = args.dst / "dog" / split / f"{image.stem}.png"
                    cv2.imwrite(str(out), channel)
                else:
                    out = out_dir / f"{image.stem}.jpg"
                    cv2.imwrite(
                        str(out), cv2.merge([channel, bgr[:, :, 1], bgr[:, :, 2]]),
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                    )
                    label = label_path(image)
                    if label.is_file():
                        shutil.copy2(label, label_dir)
                    else:
                        raise SystemExit(f"missing label for {image}")
                totals["bytes"] += out.stat().st_size
            totals["images"] += 1
            written += 1
            manifest.append({"split": split, "stem": image.stem, "source": image.as_posix()})
            if written % 200 == 0:
                print(f"  [{split}] {written}/{len(images)}", flush=True)
        print(f"[{split}] {written} frames")

    elapsed = time.time() - started
    per_image = elapsed / max(totals["images"], 1)
    print(
        f"mode={args.mode} const={args.const} s1={args.s1} s2={args.s2} images={totals['images']} "
        f"elapsed={elapsed:.1f}s ({per_image * 1000:.1f} ms/frame) written={totals['bytes'] / 1e6:.1f} MB"
    )
    if args.limit:
        print(f"sample mode: {totals['images']} frames only; multiply by the full frame count for the full run")

    if not args.dry_run:
        args.dst.mkdir(parents=True, exist_ok=True)
        (args.dst / "dog_manifest.json").write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "const_control": args.const,
                    "s1": args.s1,
                    "s2": args.s2,
                    "gain": args.gain,
                    "frames": manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.mode == "compose":
            (args.dst / "data-dog.yaml").write_text(
                "# composed B=DoG, G/R=original; generated by tools/ard_mav/make_dog_channel.py\n"
                f"path: {args.dst.as_posix()}\n"
                "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: ['Drone']\n",
                encoding="utf-8",
            )
            print(f"saved {args.dst / 'data-dog.yaml'}")


if __name__ == "__main__":
    main()
