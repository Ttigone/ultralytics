# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build the round-3 mixed dataset: DUT-Anti-UAV as the base domain, ARD-MAV restricted to its larger targets.

Why the filter exists (measured, see ARD-MAV-ROUND2-PLAN.md / the Hard1 diagnosis):
    ARD-MAV labels have a median longest side of 12.0 px and 50.9% of boxes are 8-16 px in the original frame, which
    collapses to 4-8 px inside a 960-letterboxed network input - the same scale as grass/foliage texture. Training on
    those positives teaches "any small blob is a drone", and that is exactly what fires on Hard1. DUT-Anti-UAV labels
    have a median of 31.4 px and 71.9% of boxes are >32 px, so its positives are structured objects.

    The filter is applied per IMAGE, not per box: an image is kept when at least one of its boxes reaches
    `--min-px`, and then all of its boxes are kept. Dropping the sub-threshold boxes instead would leave a drone in the
    picture with an empty label file, which teaches the classifier to suppress it.

Images and labels are hardlinked (copy fallback) into a self-contained `images/{train,val,test}` + `labels/...` tree
under `--out`, because the ARD-MAV split is sequence-level and keeps val frames inside `images/train`: a `.txt` list
would make train and val resolve to the same label directory and overwrite each other's `.cache`. Hardlinks cost no
extra disk, and the source trees stay untouched.

Usage:
    python tools/ard_mav/build_mixed_dataset.py --bg g:/0-rk3588/train/hard1_neg/images/bg
    python tools/ard_mav/build_mixed_dataset.py --ard-share 0.35 --min-px 20 --out g:/0-rk3588/train/ARD-MAV/mixed_r3b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

BUCKETS = ((0, 4, "<4px"), (4, 8, "4-8px"), (8, 16, "8-16px"), (16, 32, "16-32px"), (32, 1e9, ">32px"))


def bucket_of(size: float) -> str:
    """Return the size-bucket label for a box's longest side in original-image pixels."""
    return next(name for lo, hi, name in BUCKETS if lo <= size < hi)


def image_size(path: Path) -> tuple[int, int]:
    """Read (width, height) from the JPEG/PNG header without decoding pixels."""
    with Image.open(path) as im:
        return im.size


def read_txt_list(path: Path) -> list[Path]:
    """Read an Ultralytics image list (one absolute path per line)."""
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_path(image: Path) -> Path:
    """Mirror Ultralytics' `img2label_paths`: swap the `/images/` segment for `/labels/`."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise SystemExit(f"Image path has no 'images' segment, cannot locate its label: {image}")


def scan(images: list[Path], min_px: float) -> tuple[list[Path], list[Path], dict]:
    """Split images into (kept, dropped) by the per-image max-box-size rule and collect size statistics."""
    kept, dropped = [], []
    buckets = dict.fromkeys([name for _, _, name in BUCKETS], 0)
    boxes_total = 0
    for image in images:
        label = label_path(image)
        if not label.is_file():
            continue
        rows = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows:
            kept.append(image)  # genuine background image, no target to filter
            continue
        width, height = image_size(image)
        sides, rows_ok = [], []
        for parts in rows:
            if len(parts) != 5:
                continue
            _, _, _, bw, bh = (float(v) for v in parts)
            sides.append(max(bw * width, bh * height))
            rows_ok.append(parts)
        if not rows_ok:
            kept.append(image)
            continue
        boxes_total += len(rows_ok)
        for side in sides:
            buckets[bucket_of(side)] += 1
        (kept if max(sides) >= min_px else dropped).append(image)
    stats = {"images_in": len(images), "images_kept": len(kept), "images_dropped": len(dropped), "buckets": buckets}
    stats["boxes_in"] = boxes_total
    return kept, dropped, stats


def subsample(images: list[Path], target: int, seed: int) -> list[Path]:
    """Round-robin over sequences so a shrunken ARD set keeps every flight recording represented."""
    if target >= len(images):
        return list(images)
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[image.stem.rpartition("_")[0]].append(image)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    for name in names:
        groups[name].sort()
    picked: list[Path] = []
    depth = 0
    while len(picked) < target:
        added = False
        for name in names:
            if depth < len(groups[name]):
                picked.append(groups[name][depth])
                added = True
                if len(picked) == target:
                    break
        if not added:
            break
        depth += 1
    return picked


def large_images(images: list[Path], lg_px: float) -> list[Path]:
    """Return the images holding at least one target whose short side reaches `lg_px`.

    These are the frames the deployment clip fails on (a drone filling 40-80% of the picture) and they are only 3.2% of
    DUT train, i.e. one occurrence every three epochs. Repeating them is the cheapest way to raise their weight without
    synthesising anything.
    """
    out = []
    for image in images:
        label = label_path(image)
        if not label.is_file():
            continue
        rows = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows:
            continue
        width, height = image_size(image)
        sides = [min(float(r[3]) * width, float(r[4]) * height) for r in rows if len(r) == 5]
        if sides and max(sides) >= lg_px:
            out.append(image)
    return out


def bucket_dist(images: list[Path]) -> dict:
    """Compute the size-bucket distribution of the boxes referenced by an image list."""
    buckets = dict.fromkeys([name for _, _, name in BUCKETS], 0)
    boxes = 0
    for image in images:
        label = label_path(image)
        if not label.is_file():
            continue
        rows = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows:
            continue
        width, height = image_size(image)
        for parts in rows:
            if len(parts) != 5:
                continue
            _, _, _, bw, bh = (float(v) for v in parts)
            buckets[bucket_of(max(bw * width, bh * height))] += 1
            boxes += 1
    dist = {k: round(v / max(boxes, 1), 4) for k, v in buckets.items()}
    return {"images": len(images), "boxes": boxes, "buckets_pct": dist}


def link(src: Path, dst: Path) -> None:
    """Hardlink `src` to `dst`, falling back to a copy when the two paths live on different volumes."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialise(out: Path, splits: dict) -> dict:
    """Build `out/images/<split>` + `out/labels/<split>` so every split owns a separate label directory.

    Ultralytics resolves a label by swapping the `/images/` segment for `/labels/`, and takes the dataset cache
    location from the first label's parent. ARD-MAV keeps its val frames under `images/train` (its split is
    sequence-level, expressed as `.txt` lists), so pointing `train:` and `val:` at `.txt` lists makes both resolve
    to `labels/train` and both write `labels/train.cache` - the val pass then silently overwrites the train cache.
    Materialising per-split directories removes that collision; ARD val frames get their own `labels/val`.

    Args:
        out (Path): dataset root.
        splits (dict[str, list[tuple[Path, str, int]]]): split name -> (source image, name prefix, repeat index).
            A repeat index of -1 means "no repeat suffix".

    Returns:
        (dict[str, list[str]]): split name -> file names written under `images/<split>`.
    """
    result = {}
    for split, entries in splits.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        for d in (img_dir, lbl_dir):
            if d.is_dir():
                shutil.rmtree(d)  # also drops a stale .cache from an earlier build
            d.mkdir(parents=True)
        names = []
        for src, prefix, rep in entries:
            name = f"{prefix}_{src.stem}_r{rep}{src.suffix}" if rep >= 0 else f"{prefix}_{src.stem}{src.suffix}"
            link(src, img_dir / name)
            label = label_path(src)
            if label.is_file():
                link(label, (lbl_dir / name).with_suffix(".txt"))
            else:
                (lbl_dir / name).with_suffix(".txt").write_text("", encoding="utf-8")
            names.append(name)
        result[split] = names
    return result


def main() -> None:
    """Assemble the mixed dataset and print the size-bucket shift it produces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dut-root", type=Path, default=Path("g:/0-rk3588/train/DUT-Anti-UAV/yolo_dataset"))
    parser.add_argument("--ard-root", type=Path, default=Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo"))
    parser.add_argument("--bg", type=Path, default=None, help="background patch image dir from mine_hard_negatives.py")
    parser.add_argument("--out", type=Path, default=Path("g:/0-rk3588/train/ARD-MAV/mixed_r3"))
    parser.add_argument("--min-px", type=float, default=16.0, help="keep an ARD image only if one box reaches this size")
    parser.add_argument(
        "--ard-share",
        type=float,
        default=0.25,
        help="target ARD frame share of the train split (background patches excluded from the denominator)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--bg-repeat",
        type=int,
        default=3,
        help="repeat each background strip N times in the train list; they are few (one static scene) but each carries "
        "only negatives, so repeating raises the per-epoch negative pressure without touching the positive set",
    )
    parser.add_argument(
        "--synth",
        type=Path,
        default=None,
        help="synthetic sample tree from synth_hard_examples.py (images/train + labels/train); all of it goes to train",
    )
    parser.add_argument("--lg-px", type=float, default=200.0, help="short side that makes a frame a large-target frame")
    parser.add_argument(
        "--lg-repeat",
        type=int,
        default=3,
        help="extra copies of each large-target DUT frame in the train list (0 disables the oversampling)",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="splits to materialise; `val` alone builds an eval-only set (ARD val + --val-dut-share DUT val) for "
        "re-scoring a finished checkpoint without touching the training set",
    )
    parser.add_argument(
        "--val-dut-share",
        type=float,
        default=0.0,
        help="fraction of DUT val added to the val split; with --dut-root dut_seq (rebuilt split) 0.15 makes the val "
        "cover both domains instead of ARD alone",
    )
    args = parser.parse_args()

    ard_splits = args.ard_root / "splits"
    dut_train = sorted((args.dut_root / "images/train").glob("*.jpg"))
    dut_val = sorted((args.dut_root / "images/val").glob("*.jpg"))
    ard_train_all = read_txt_list(ard_splits / "train.txt")
    ard_val_all = read_txt_list(ard_splits / "val.txt")
    ard_test_all = read_txt_list(ard_splits / "test.txt")
    bg = sorted(args.bg.glob("*.jpg")) if args.bg and args.bg.is_dir() else []
    synth = sorted((args.synth / "images/train").glob("*.jpg")) if args.synth and args.synth.is_dir() else []
    dut_large = large_images(dut_train, args.lg_px)

    ard_train, ard_train_drop, ard_stats = scan(ard_train_all, args.min_px)
    ard_val, _, ard_val_stats = scan(ard_val_all, args.min_px)
    _, _, ard_test_stats = scan(ard_test_all, args.min_px)

    # ARD share is measured against the labelled frames only, so adding background strips cannot silently dilute it.
    target_ard = len(ard_train) if args.ard_share <= 0 else round(len(dut_train) * args.ard_share / (1 - args.ard_share))
    ard_train = subsample(ard_train, target_ard, args.seed)

    splits = {
        "train": [(p, "dut", -1) for p in dut_train]
        + [(p, "dut", r) for r in range(max(args.lg_repeat, 0)) for p in dut_large]
        + [(p, "ard", -1) for p in ard_train]
        + [(p, "bg", r) for r in range(max(args.bg_repeat, 1)) for p in bg]
        + [(p, "syn", -1) for p in synth],
        "val": [(p, "ard", -1) for p in ard_val]
        + (
            [(p, "dut", -1) for p in random.Random(args.seed).sample(dut_val, round(len(dut_val) * args.val_dut_share))]
            if args.val_dut_share > 0
            else []
        ),
        # test stays unfiltered: it is only used for reporting, and keeping it identical to the round-2 split
        # (`splits/test.txt`) is what makes the pre/post retraining comparison meaningful.
        "test": [(p, "ard", -1) for p in ard_test_all],
    }
    if not splits["val"]:
        raise SystemExit("Empty val split; check --ard-root / --min-px")
    keep = [s.strip() for s in args.splits.split(",") if s.strip()]
    splits = {k: v for k, v in splits.items() if k in keep}
    if "train" in splits and not splits["train"]:
        raise SystemExit("Empty train split; check --dut-root / --ard-root / --min-px")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    counts = {split: len(entries) for split, entries in splits.items()}
    materialise(out, splits)
    (out / "data.yaml").write_text(
        "\n".join(
            [
                "# Round-3 mixed dataset: DUT-Anti-UAV base + ARD-MAV (>=%.0f px) + Hard1 background strips" % args.min_px,
                "# generated by tools/ard_mav/build_mixed_dataset.py",
                f"path: {out.as_posix()}",
                "train: images/train" if "train" in splits else "train: images/val",
                "val: images/val",
                *(["test: images/test"] if "test" in splits else []),
                "nc: 1",
                "names: ['drone']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "dut_root": args.dut_root.as_posix(),
        "ard_root": args.ard_root.as_posix(),
        "bg_dir": args.bg.as_posix() if args.bg else None,
        "min_px": args.min_px,
        "ard_share_requested": args.ard_share,
        "seed": args.seed,
        "counts": {
            "dut_train": len(dut_train),
            "ard_train_available_after_filter": ard_stats["images_kept"],
            "ard_train_used": len(ard_train),
            "ard_train_dropped_by_size": len(ard_train_drop),
            "background_strips": len(bg),
            "background_repeat": args.bg_repeat,
            "synthetic_samples": len(synth),
            "dut_large_frames": len(dut_large),
            "dut_large_repeat": args.lg_repeat,
            **{f"{split}_total": n for split, n in counts.items()},
        },
        "ard_filter": {"train": ard_stats, "val": ard_val_stats, "test": ard_test_stats},
        "ard_share_actual": round(len(ard_train) / max(len(dut_train) + len(ard_train), 1), 4),
        "bg_share_of_train": round(len(bg) / max(counts.get("train", 0), 1), 4),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"out={out}")
    if "train" in counts:
        print(
            f"  train={counts['train']} (dut {len(dut_train)} + dut-large extras {len(dut_large) * max(args.lg_repeat, 0)}"
            f" + ard {len(ard_train)} + bg {len(bg)} x{args.bg_repeat} + synth {len(synth)})"
        )
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  ARD filtered to >= {args.min_px:.0f} px: {ard_stats['images_kept']}/{ard_stats['images_in']} images kept")
    print(f"  ARD actual share of labelled train frames: {manifest['ard_share_actual']:.1%}")
    print("  ARD train boxes before filter:", ard_stats["buckets"])
    if "train" not in splits:
        return
    print("  size buckets  ARD-orig -> mixed-train:")
    before = bucket_dist(ard_train_all)
    after = bucket_dist([p for p, _, _ in splits["train"]])
    for name in before["buckets_pct"]:
        print(f"    {name:>7}: {before['buckets_pct'][name]:6.1%} -> {after['buckets_pct'][name]:6.1%}")
    print(f"  boxes: ARD {before['boxes']} -> mixed {after['boxes']}")


if __name__ == "__main__":
    main()
