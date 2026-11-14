# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Synthesise the three sample types Hard1.mp4 needs and neither dataset contains.

The clip fails in two directions and both are data gaps, not architecture gaps:

    large targets     the drone sometimes covers 40-80% of the picture and is missed. DUT train holds such frames
                      (3.2%) but they appear once every three epochs. `crop` zooms into those frames so the target
                      dominates a textured background, which is the exact Hard1 failure mode.
    hard negatives    grass and treeline fire at small scale. `degrade` turns the mined Hard1 background strips grey,
                      hazy and low-contrast, so the negative set covers the lighting the clip was shot in.
    mid-size targets  the detector has only seen 8-32 px drones. `paste` copies real drone crops onto real Hard1
                      background strips at mid size, filling the 32-160 px hole between "tiny" and "huge".

Only `paste` and `crop` carry labels; `degrade` is negatives by construction, which is why it is applied to the mined
background strips and never to a labelled frame.

Usage:
    python tools/ard_mav/synth_hard_examples.py
    python tools/ard_mav/synth_hard_examples.py --modes paste,crop --count 300
    python tools/ard_mav/synth_hard_examples.py --crop-scale-mode area --crop-frac 0.6
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--dut-root", type=Path, default=Path("g:/0-rk3588/train/DUT-Anti-UAV/dut_seq"))
    p.add_argument("--bg", type=Path, default=Path("g:/0-rk3588/train/hard1_neg/images/bg"))
    p.add_argument("--out", type=Path, default=Path("g:/0-rk3588/train/hard1_neg/synth"))
    p.add_argument("--modes", default="paste,crop,degrade", help="any of paste, crop, degrade")
    p.add_argument("--count", type=int, default=200, help="samples per mode")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--lg-px", type=float, default=200.0, help="short side that marks a large-target frame")
    p.add_argument("--paste-min-px", type=float, default=32.0, help="smallest drone crop worth pasting")
    p.add_argument("--paste-max-px", type=float, default=160.0, help="largest drone crop worth pasting")
    p.add_argument("--paste-factor", type=float, default=0.25, help="random resize range for a pasted crop (+/-)")
    p.add_argument(
        "--crop-scale-mode",
        choices=("edge", "area"),
        default="area",
        help="how 'the target occupies 40-80%% of the frame' is measured: bbox area or bbox edge length",
    )
    p.add_argument(
        "--crop-fracs",
        default="0.4,0.6,0.8",
        help="area shares the synthesised crop should give the target; one batch of --crop-count is made per value",
    )
    p.add_argument("--crop-count", type=int, default=100, help="crops per --crop-fracs value")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality")
    return p.parse_args()


def read_label(path: Path) -> list[list[float]]:
    """Read YOLO rows, dropping malformed ones."""
    if not path.is_file():
        return []
    return [[float(v) for v in ln.split()] for ln in path.read_text(encoding="utf-8").splitlines() if len(ln.split()) == 5]


def label_path(image: Path) -> Path:
    """Mirror Ultralytics' `/images/` -> `/labels/` swap."""
    parts = list(image.parts)
    parts[len(parts) - 3] = "labels"
    return Path(*parts).with_suffix(".txt")


def collect(dut_root: Path, lo: float, hi: float, lg_px: float) -> tuple[list, list]:
    """Return (paste_sources, crop_sources) as (image, (x1, y1, x2, y2), width, height)."""
    paste, crop = [], []
    for image in sorted((dut_root / "images/train").glob("*.jpg")):
        rows = read_label(label_path(image))
        if not rows:
            continue
        h, w = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE).shape[:2]
        for _, xc, yc, bw, bh in rows:
            box = (
                (xc - bw / 2) * w,
                (yc - bh / 2) * h,
                (xc + bw / 2) * w,
                (yc + bh / 2) * h,
            )
            short = min(box[2] - box[0], box[3] - box[1])
            if lo <= short <= hi:
                paste.append((image, box, w, h))
            if short >= lg_px:
                crop.append((image, box, w, h))
    return paste, crop


def save(out: Path, name: str, image: np.ndarray, boxes: list[tuple[float, float, float, float]], quality: int) -> None:
    """Write one synthetic image and its YOLO label."""
    h, w = image.shape[:2]
    (out / "images/train").mkdir(parents=True, exist_ok=True)
    (out / "labels/train").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images/train" / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    rows = [f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}" for x1, y1, x2, y2 in boxes]
    (out / "labels/train" / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def make_paste(sources: list, backgrounds: list[Path], args, rng: random.Random) -> int:
    """Copy a real drone crop onto a real Hard1 strip, feathering the seam."""
    made = 0
    while made < args.count:
        image, box, _, _ = sources[rng.randrange(len(sources))]
        src = cv2.imread(str(image))
        if src is None:
            continue
        mx, my = (box[2] - box[0]) * 0.25, (box[3] - box[1]) * 0.25
        x1, y1 = max(0, int(box[0] - mx)), max(0, int(box[1] - my))
        x2, y2 = min(src.shape[1], int(box[2] + mx)), min(src.shape[0], int(box[3] + my))
        patch = src[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        factor = 1.0 + rng.uniform(-args.paste_factor, args.paste_factor)
        patch = cv2.resize(patch, (max(4, int(patch.shape[1] * factor)), max(4, int(patch.shape[0] * factor))))
        bg = cv2.imread(str(backgrounds[rng.randrange(len(backgrounds))]))
        if bg is None or patch.shape[0] >= bg.shape[0] or patch.shape[1] >= bg.shape[1]:
            continue
        px = rng.randrange(0, bg.shape[1] - patch.shape[1])
        py = rng.randrange(0, bg.shape[0] - patch.shape[0])
        # feather the rectangular seam so the crop does not teach "visible border == object"
        mask = np.zeros(patch.shape[:2], np.float32)
        pad = max(2, min(patch.shape[:2]) // 8)
        mask[pad : patch.shape[0] - pad, pad : patch.shape[1] - pad] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, pad / 2))[:, :, None]
        roi = bg[py : py + patch.shape[0], px : px + patch.shape[1]].astype(np.float32)
        bg[py : py + patch.shape[0], px : px + patch.shape[1]] = (roi * (1 - mask) + patch * mask).astype(np.uint8)
        # label the drone itself, not the feathered rectangle: map the source box through crop and resize
        scale = patch.shape[1] / (x2 - x1)
        new_box = (
            px + (box[0] - x1) * scale,
            py + (box[1] - y1) * scale,
            px + (box[2] - x1) * scale,
            py + (box[3] - y1) * scale,
        )
        save(args.out, f"paste_{made:05d}", bg, [new_box], args.quality)
        made += 1
    return made


def make_crop(sources: list, args, rng: random.Random) -> dict:
    """Crop around a large target until it occupies the requested share of the crop area.

    `--crop-scale-mode area` is the confirmed definition: a target covering `p` of the crop area has a linear size of
    `sqrt(p) * crop_side`, so the crop is `sqrt(area / p)`. p=0.4 -> 1.58x the target, p=0.8 -> 1.12x, which is the
    1.12-1.58x band and not the 1.25-2.5x the edge reading would give. `edge` is kept for comparison only.

    Boxes that would be cut by the image border are skipped rather than clipped: a half-drone teaches the detector that
    half a drone is a whole one.
    """
    counts = {}
    for frac in [float(f) for f in args.crop_fracs.split(",") if f.strip()]:
        tag = f"p{round(frac * 100):02d}"
        idx = list(range(len(sources)))
        rng.shuffle(idx)
        made = 0
        for i in idx:
            if made >= args.crop_count:
                break
            image, box, w, h = sources[i]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if args.crop_scale_mode == "area":
                # a rectangular crop is the only way to hit an exact area share: with a square crop an elongated box
                # can never exceed 1/1.25 = 80% (its long edge alone would be longer than the crop side)
                cw, ch = bw / math.sqrt(frac), bh / math.sqrt(frac)
            else:
                cw = ch = max(bw, bh) / frac
            if min(cw, ch) < 96 or cw > w or ch > h:
                continue
            cw, ch = int(round(cw)), int(round(ch))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            x0 = int(round(min(max(cx - cw / 2, 0), w - cw)))
            y0 = int(round(min(max(cy - ch / 2, 0), h - ch)))
            if box[0] < x0 or box[1] < y0 or box[2] > x0 + cw or box[3] > y0 + ch:
                continue  # would chop the target
            src = cv2.imread(str(image))
            if src is None:
                continue
            patch = src[y0 : y0 + ch, x0 : x0 + cw]
            kept = []
            for _, xc, yc, lbw, lbh in read_label(label_path(image)):
                bx1, by1 = (xc - lbw / 2) * w, (yc - lbh / 2) * h
                bx2, by2 = (xc + lbw / 2) * w, (yc + lbh / 2) * h
                if not (x0 < (bx1 + bx2) / 2 < x0 + cw and y0 < (by1 + by2) / 2 < y0 + ch):
                    continue
                kept.append((max(bx1 - x0, 0), max(by1 - y0, 0), min(bx2 - x0, cw), min(by2 - y0, ch)))
            if not kept:
                continue
            save(args.out, f"crop_{tag}_{made:05d}", patch, kept, args.quality)
            made += 1
        counts[tag] = made
    return counts


def make_degrade(backgrounds: list[Path], args, rng: random.Random) -> int:
    """Produce grey-sky, hazy and low-contrast negatives from the mined strips."""
    made = 0
    while made < args.count:
        bg = cv2.imread(str(backgrounds[rng.randrange(len(backgrounds))]))
        if bg is None:
            continue
        kind = made % 3
        img = bg.astype(np.float32)
        if kind == 0:  # overcast: desaturate and lift the floor
            gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)[:, :, None]
            img = gray * 0.75 + img * 0.25 + 30
        elif kind == 1:  # haze: blend towards a bright grey, i.e. low contrast at distance
            img = img * 0.7 + 190 * 0.3
        else:  # flat light: compress the histogram around its mean
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = mean + (img - mean) * 0.55
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 3.0, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        save(args.out, f"deg_{made:05d}", img, [], args.quality)
        made += 1
    return made


def main() -> None:
    """Build every requested synthetic set and report what was written."""
    args = parse_args()
    rng = random.Random(args.seed)
    backgrounds = sorted(args.bg.glob("*.jpg"))
    if not backgrounds:
        raise SystemExit(f"No background strips under {args.bg}; run mine_hard_negatives.py first")
    out = args.out.resolve()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    # clear only the batches being rebuilt, so regenerating `crop` keeps the paste/degrade batches on disk
    prefixes = {"paste": "paste_", "crop": "crop_", "degrade": "deg_"}
    for mode in modes:
        for path in list((out / "images/train").glob(prefixes[mode] + "*.jpg")) if (out / "images/train").is_dir() else []:
            path.unlink()
            (out / "labels/train" / (path.stem + ".txt")).unlink(missing_ok=True)

    paste_src, crop_src = collect(args.dut_root, args.paste_min_px, args.paste_max_px, args.lg_px)
    print(f"sources: paste={len(paste_src)} crop={len(crop_src)} backgrounds={len(backgrounds)}")
    counts = {}
    if "paste" in modes:
        counts["paste"] = make_paste(paste_src, backgrounds, args, rng)
    if "crop" in modes:
        counts["crop"] = make_crop(crop_src, args, rng)
        print(f"crop-scale-mode={args.crop_scale_mode} fracs={args.crop_fracs}")
    if "degrade" in modes:
        counts["degrade"] = make_degrade(backgrounds, args, rng)
    print(f"out={out} {counts}")


if __name__ == "__main__":
    main()
