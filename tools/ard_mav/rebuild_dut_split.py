# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Rebuild the DUT-Anti-UAV split so validation frames no longer come from training videos.

The published split cannot be used as-is. Measured on `yolo_dataset` (10,000 frames):

    dHash of every val frame against every train frame: 64.9% match at 0 bits and 81.8% at <=2 bits.
    Byte-level md5 says 0 copies, so these are the same frames re-encoded/rescaled, not shared files.

Every val frame therefore has a visually identical twin in train, which is why the DUT baseline reports mAP50 0.919
while the same checkpoint misses a third of the frames on Hard1.mp4. Optimising against that val set rewards
memorisation, not generalisation.

The filenames are flat (`00001.jpg` ...) and carry no sequence id, so the grouping is reconstructed from pixels:

    duplicate edge   two frames within `--dup-bits` of each other are the same frame, so they must never straddle two
                     splits. All copies are kept (they differ in resolution, which is useful scale variety) but they
                     always land in the same split.
    temporal edge    two frames that are adjacent in the published ordering, share a resolution and stay within
                     `--run-bits` belong to the same take.

Connected components of those two edge sets are the unit that gets assigned to train/val/test, which makes a leak
between train and val impossible by construction. Thresholds are deliberately conservative: a false merge only costs a
slightly imbalanced split, while a false split re-introduces the leak.

Usage:
    python tools/ard_mav/rebuild_dut_split.py
    python tools/ard_mav/rebuild_dut_split.py --out g:/0-rk3588/train/DUT-Anti-UAV/dut_seq --val-share 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
MASK64 = (1 << 64) - 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("g:/0-rk3588/train/DUT-Anti-UAV/yolo_dataset"))
    p.add_argument("--out", type=Path, default=Path("g:/0-rk3588/train/DUT-Anti-UAV/dut_seq"))
    p.add_argument("--val-share", type=float, default=0.15)
    p.add_argument("--test-share", type=float, default=0.15)
    p.add_argument("--dup-bits", type=int, default=3, help="hamming distance treated as the same frame")
    p.add_argument("--run-bits", type=int, default=18, help="hamming distance that still chains adjacent frames")
    p.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def dhash(path: Path) -> int:
    """Return the 64-bit difference hash of one image."""
    with Image.open(path) as im:
        gray = im.convert("L").resize((9, 8), Image.BILINEAR)
    bits = (np.asarray(gray, dtype=np.int16)[:, 1:] > np.asarray(gray, dtype=np.int16)[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def scan(root: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Return (images, hashes, wh) for every frame in the published train/val/test split."""
    images = []
    for split in ("train", "val", "test"):
        files = sorted((root / "images" / split).glob("*.jpg"))
        print(f"  {split}: {len(files)}")
        images.extend(files)
    hashes = np.zeros(len(images), np.uint64)
    wh = np.zeros((len(images), 2), np.int32)
    for i, path in enumerate(images):
        hashes[i] = np.uint64(dhash(path) & MASK64)
        with Image.open(path) as im:
            wh[i] = im.size
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(images)}")
    return images, hashes, wh


def popcount(x: np.ndarray) -> np.ndarray:
    """Per-element popcount of a uint64 array."""
    return POPCOUNT[x.view(np.uint8)].reshape(-1, 8).sum(axis=1, dtype=np.uint8)


class UnionFind:
    """Disjoint-set forest with path halving."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_units(images: list[Path], hashes: np.ndarray, wh: np.ndarray, dup_bits: int, run_bits: int) -> np.ndarray:
    """Group frame indices into sequences using duplicate and temporal edges."""
    n = len(images)
    uf = UnionFind(n)

    # duplicate edges: the same frame saved twice, possibly at two resolutions
    dup_edges = 0
    for start in range(0, n, 500):
        chunk = hashes[start : start + 500]
        dist = popcount(chunk[:, None] ^ hashes[None, :]).reshape(len(chunk), n)
        rows, cols = np.nonzero(dist <= dup_bits)
        for a, b in zip(rows + start, cols):
            if a < b:
                uf.union(int(a), int(b))
                dup_edges += 1

    # temporal edges: consecutive published filenames inside one split, same take
    temporal_edges = 0
    offset = 0
    for split in ("train", "val", "test"):
        count = 0
        while offset + count < n and images[offset + count].parent.name == split:
            count += 1
        for i in range(offset, offset + count - 1):
            a, b = images[i], images[i + 1]
            if int(b.stem) != int(a.stem) + 1 or tuple(wh[i]) != tuple(wh[i + 1]):
                continue
            dist = int(popcount(np.array([hashes[i] ^ hashes[i + 1]], np.uint64))[0])
            if dist <= run_bits:
                uf.union(i, i + 1)
                temporal_edges += 1
        offset += count
    print(f"  duplicate edges={dup_edges} temporal edges={temporal_edges}")

    roots = np.array([uf.find(i) for i in range(n)])
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots.tolist()))}
    units = np.array([remap[r] for r in roots], dtype=np.int32)
    print(f"  units={len(remap)} sizes: {sorted(Counter(units.tolist()).values(), reverse=True)[:10]}")
    return units


def assign(units: np.ndarray, val_share: float, test_share: float, seed: int) -> dict[str, list[int]]:
    """Fill train/val/test with whole units, keeping both the frame ratio and the unit order unbiased.

    Units are visited in a seeded random order rather than largest-first. Largest-first looks harmless but is not:
    the 2,000 single-frame units are also the ones holding the large targets, so they all end up filling whichever
    split is still short, which moved the val median short side from 21 px to 44 px in the first build.
    """
    members: dict[int, list[int]] = defaultdict(list)
    for idx, unit in enumerate(units):
        members[int(unit)].append(idx)
    targets = {"train": 1 - val_share - test_share, "val": val_share, "test": test_share}
    total = len(units)
    assigned = {"train": [], "val": [], "test": []}
    counts = dict.fromkeys(assigned, 0)
    order = list(members)
    random.Random(seed).shuffle(order)
    for unit in order:
        split = max(targets, key=lambda s: targets[s] * total - counts[s])
        assigned[split].extend(members[unit])
        counts[split] += len(members[unit])
    return {k: sorted(v) for k, v in assigned.items()}


def link(src: Path, dst: Path, mode: str) -> None:
    """Hardlink (or copy) one file into place, refusing to overwrite: a duplicate name means a bug upstream."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"Refusing to overwrite {dst}; output names must be unique")
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def label_of(image: Path) -> Path:
    """Mirror Ultralytics' `/images/` -> `/labels/` swap."""
    parts = list(image.parts)
    parts[len(parts) - 3] = "labels"
    return Path(*parts).with_suffix(".txt")


def stats(images: list[Path]) -> dict:
    """Frame count, box count and short-side sizes for one split."""
    boxes, short_sides, big_frames = 0, [], 0
    for image in images:
        rows = [ln.split() for ln in label_of(image).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows:
            continue
        with Image.open(image) as im:
            w, h = im.size
        sides = [min(float(r[3]) * w, float(r[4]) * h) for r in rows if len(r) == 5]
        boxes += len(sides)
        short_sides.extend(sides)
        if sides and max(sides) > 200:
            big_frames += 1
    return {
        "frames": len(images),
        "boxes": boxes,
        "median_short_side": round(float(np.median(short_sides)), 1) if short_sides else 0.0,
        "frames_with_box_gt200px": big_frames,
    }


def leakage(train_hashes: np.ndarray, val_hashes: np.ndarray) -> dict:
    """Nearest-neighbour hamming distance of every held-out frame against the train pool."""
    best = np.full(len(val_hashes), 64, np.uint8)
    for start in range(0, len(val_hashes), 500):
        chunk = val_hashes[start : start + 500]
        dist = popcount(chunk[:, None] ^ train_hashes[None, :]).reshape(len(chunk), len(train_hashes))
        best[start : start + len(chunk)] = dist.min(axis=1)
    return {f"<={t}": round(float(np.mean(best <= t)), 4) for t in (0, 2, 4, 6, 10, 16)}


def main() -> None:
    """Build the sequence-level split and report the leak it removes."""
    args = parse_args()
    print(f"scanning {args.root}")
    images, hashes, wh = scan(args.root)

    print("building units")
    units = build_units(images, hashes, wh, args.dup_bits, args.run_bits)
    splits = assign(units, args.val_share, args.test_share, args.seed)

    out = args.out.resolve()
    if out.is_dir():
        shutil.rmtree(out)
    report, hash_of = {}, {}
    for split, idxs in splits.items():
        chosen = [images[i] for i in idxs]
        for image in chosen:
            # the three published splits restart numbering, so the source split has to be part of the name
            name = f"{image.parent.name}_{image.name}"
            link(image, out / "images" / split / name, args.link_mode)
            link(label_of(image), out / "labels" / split / name.replace(".jpg", ".txt"), args.link_mode)
        report[split] = stats(chosen)
        hash_of[split] = hashes[idxs]
        print(f"{split}: {report[split]}")

    report["leak_val_to_train"] = leakage(hash_of["train"], hash_of["val"])
    report["leak_test_to_train"] = leakage(hash_of["train"], hash_of["test"])
    report["units"] = int(units.max()) + 1
    report["duplicate_copies_dropped"] = len(images) - sum(len(v) for v in splits.values())
    report["source_root"] = args.root.as_posix()
    report["thresholds"] = {"dup_bits": args.dup_bits, "run_bits": args.run_bits}

    (out / "data.yaml").write_text(
        "\n".join(
            [
                "# Sequence-level DUT-Anti-UAV split rebuilt by tools/ard_mav/rebuild_dut_split.py",
                f"path: {out.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names: ['drone']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nleak val->train: {report['leak_val_to_train']}")
    print(f"leak test->train: {report['leak_test_to_train']}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
