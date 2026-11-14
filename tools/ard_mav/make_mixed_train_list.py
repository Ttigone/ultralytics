# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build a size-filtered, multi-source training list for the ARD-MAV aerial tiny-UAV detector.

ARD-MAV is the only in-domain source, so it stays the backbone of every split. Auxiliary RGB sources are
mixed into ``train`` only, and only images whose every box is smaller than ``--max-net-side`` in the
network input space qualify -- a ground-to-air frame with a 200 px airliner must never teach the tiny-target
prior. Validation and test stay ARD-MAV-only so the reported metrics keep their meaning.

Sources and what each contributes:

- ARD-MAV: in-domain empty-to-air frames, all of them; empty-label frames become negatives.
- DUT-Anti-UAV: ground-to-air RGB, single class, already YOLO.
- UAV-Anti-UAV: ground-to-air RGB, mostly large aircraft, so only tiny frames survive the filter.
- VisioDECT: ground-to-air RGB, 17 classes remapped to 0. Its labels live in a parallel tree, so
  ``--materialize-visiodect`` copies them next to the images where Ultralytics looks for them.
- bird archive: hard negatives (sky birds), unzipped with an empty label per image by ``--materialize-negatives``.

Everything is referenced through absolute image-list txt, so nothing is moved or duplicated except the two
materialized sources above.

Usage:
    python tools/ard_mav/make_mixed_train_list.py --dry-run
    python tools/ard_mav/make_mixed_train_list.py --write
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

BUCKETS = ((0, 8, "<=8px"), (8, 16, "8-16px"), (16, 32, "16-32px"), (32, 1e9, ">32px"))

ARDMAV = Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo-s5")
ARDMAV_LEGACY = Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo")
DUT = Path("g:/0-rk3588/train/DUT-Anti-UAV/yolo_dataset")
UAV = Path("g:/0-rk3588/train/uav-anti-uav/UAV-Anti-UAV/yolo_dataset")
VISIO = Path("g:/0-rk3588/train/visiodet/VisioDECT Dataset Upload")
NEG_ZIP = Path("g:/0-rk3588/train/other/archive.zip")
NEG_OUT = Path("g:/0-rk3588/train/ARD-MAV/negatives")


def bucket_of(size: float) -> str:
    """Return the network-space size bucket for one longest-side value."""
    return next(name for lo, hi, name in BUCKETS if lo <= size < hi)


def read_boxes(label: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse one YOLO label file into ``(cls, cx, cy, w, h)`` rows, skipping malformed lines."""
    rows = []
    if label.is_file():
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return rows


def image_size(path: Path) -> tuple[int, int] | None:
    """Read image dimensions from the JPEG/PNG header, or None if unreadable."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def label_of(image: Path) -> Path:
    """Map an image path to the label path Ultralytics will look for."""
    return Path(image.as_posix().replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def scan(paths: list[Path], imgsz: int, max_net_side: float) -> tuple[list[Path], dict]:
    """Keep images whose every box is <= ``max_net_side`` in network pixels, and report the size mix.

    Sizes are converted per image (``min(imgsz/W, imgsz/H)``), so a 1920x1080 and a 856x480 frame are
    compared in the space the network actually sees rather than in their own pixels.
    """
    kept, rec = [], {"scanned": 0, "no_label": 0, "no_image": 0, "empty": 0, "too_big": 0, "buckets": Counter()}
    for image in paths:
        rec["scanned"] += 1
        size = image_size(image)
        if size is None:
            rec["no_image"] += 1
            continue
        label = label_of(image)
        if not label.is_file():
            rec["no_label"] += 1
            continue
        boxes = read_boxes(label)
        if not boxes:
            rec["empty"] += 1
            kept.append(image)
            continue
        width, height = size
        scale = min(imgsz / width, imgsz / height)
        longest = max(max(b[3] * width, b[4] * height) * scale for b in boxes)
        if longest <= max_net_side:
            kept.append(image)
            rec["buckets"][bucket_of(longest)] += 1
        else:
            rec["too_big"] += 1
    return kept, rec


def ard_split(root: Path, split: str) -> list[Path]:
    """Resolve one ARD-MAV split: prefer the sequence-level image list over the raw directory."""
    listing = root / "splits" / f"{split}.txt"
    if listing.is_file():
        return [Path(line.strip()) for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted((root / "images" / split).glob("*.jpg"))


def materialize_visiodect(root: Path, imgsz: int, max_net_side: float) -> list[Path]:
    """Copy VisioDECT labels next to their images so Ultralytics resolves them, remapping every class to 0.

    The dataset ships ``<model>/labels/<light>/txt/<stem>.txt`` while the image sits in
    ``<model>/images/<Light>/<stem>.jpg``; Ultralytics only looks for ``<model>/labels/<Light>/<stem>.txt``.
    Label files are tiny, so copying is cheap and leaves the original tree untouched.
    """
    sources = sorted(root.rglob("labels/*/txt/*.txt"))
    written = 0
    for label in sources:
        stem = label.stem
        model_dir = label.parents[3]
        light = label.parents[1].name
        image = model_dir / "images" / light.capitalize() / f"{stem}.jpg"
        if not image.is_file():
            matches = list((model_dir / "images").rglob(f"{stem}.jpg"))
            if not matches:
                continue
            image = matches[0]
        # Ultralytics maps /images/<Dir>/ -> /labels/<Dir>/, so the label must mirror the image directory
        light = image.parent.name
        size = image_size(image)
        if size is None:
            continue
        rows = read_boxes(label)
        if not rows:
            continue
        width, height = size
        scale = min(imgsz / width, imgsz / height)
        if max(max(b[3] * width, b[4] * height) * scale for b in rows) > max_net_side:
            continue
        target = model_dir / "labels" / light / f"{stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(f"0 {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in rows),
            encoding="utf-8",
        )
        written += 1
    print(f"visiodect: materialized {written} labels under {root}")
    return sorted(root.rglob("labels/*/*.txt"))


def materialize_negatives(zip_path: Path, out: Path) -> list[Path]:
    """Unzip sky/bird frames and write one empty label per image so they act as hard negatives."""
    images = out / "images"
    labels = out / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    kept = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [n for n in archive.namelist() if n.lower().rsplit(".", 1)[-1] in {"jpg", "jpeg", "png"}]
        for name in members:
            target = images / Path(name).name
            if not target.is_file():
                with archive.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            (labels / f"{target.stem}.txt").write_text("", encoding="utf-8")
            kept.append(target)
    print(f"negatives: extracted {len(kept)} frames -> {images}")
    return kept


def sample(images: list[Path], quota: int, rng: random.Random) -> list[Path]:
    """Randomly pick up to ``quota`` images; ``quota <= 0`` keeps everything."""
    if quota <= 0 or len(images) <= quota:
        return images
    return sorted(rng.sample(images, quota))


def main() -> None:
    """Filter each source, apply quotas and write the mixed image list plus its data YAML."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardmav", type=Path, default=ARDMAV)
    parser.add_argument("--fallback-ardmav", type=Path, default=ARDMAV_LEGACY)
    parser.add_argument("--dut", type=Path, default=DUT)
    parser.add_argument("--uav", type=Path, default=UAV)
    parser.add_argument("--visiodect", type=Path, default=VISIO)
    parser.add_argument("--negatives-zip", type=Path, default=NEG_ZIP)
    parser.add_argument("--negatives-out", type=Path, default=NEG_OUT)
    parser.add_argument("--imgsz", type=int, default=960, help="network input size used for the size filter")
    parser.add_argument("--max-net-side", type=float, default=32.0, help="keep images whose every box is <= this")
    parser.add_argument(
        "--quota",
        default="dut=1400,uav=1800,visio=800,neg=1000",
        help="per-source image caps; 0 keeps every qualifying frame",
    )
    parser.add_argument("--materialize-visiodect", action="store_true", help="copy VisioDECT labels next to images")
    parser.add_argument("--materialize-negatives", action="store_true", help="unzip the bird archive as negatives")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--write", action="store_true", help="write the list and data YAML (default: dry run)")
    args = parser.parse_args()

    quota = {k: int(v) for k, v in (item.split("=") for item in args.quota.split(",") if item)}
    rng = random.Random(args.seed)

    root = args.ardmav if (args.ardmav / "splits" / "train.txt").is_file() else args.fallback_ardmav
    if not (root / "splits" / "train.txt").is_file() and not (root / "images" / "train").is_dir():
        raise SystemExit(f"No ARD-MAV training split under {args.ardmav} or {args.fallback_ardmav}")

    sources: dict[str, list[Path]] = {}

    # In-domain source: keep every frame, empty-label ones included, because sky negatives are scarce here.
    train_images = ard_split(root, "train")
    empty = sum(1 for p in train_images if not read_boxes(label_of(p)))
    sources["ardmav"] = train_images
    print(f"ardmav: {len(train_images)} frames from {root} (empty-label negatives: {empty})")

    for name, base in (("dut", args.dut), ("uav", args.uav)):
        images = sorted((base / "images" / "train").glob("*.jpg"))
        if not images:
            print(f"{name}: SKIP, no images under {base}")
            continue
        kept, rec = scan(images, args.imgsz, args.max_net_side)
        print(f"{name}: {len(images)} scanned -> {len(kept)} kept {dict(rec['buckets'])} (too_big={rec['too_big']})")
        sources[name] = kept

    visio_images: list[Path] = []
    if args.visiodect.is_dir():
        if args.materialize_visiodect:
            materialize_visiodect(args.visiodect, args.imgsz, args.max_net_side)
        candidates = sorted(p for p in args.visiodect.rglob("images/**/*.jpg"))
        kept, rec = scan(candidates, args.imgsz, args.max_net_side)
        print(f"visio: {len(candidates)} scanned -> {len(kept)} kept {dict(rec['buckets'])} (no_label={rec['no_label']})")
        if rec["no_label"] and not args.materialize_visiodect:
            print("visio: run --materialize-visiodect first, otherwise these frames are dropped")
        visio_images = kept
        sources["visio"] = kept

    if args.negatives_zip.is_file():
        if args.materialize_negatives:
            materialize_negatives(args.negatives_zip, args.negatives_out)
        neg_images = sorted((args.negatives_out / "images").glob("*"))
        if neg_images:
            sources["neg"] = neg_images
            print(f"neg: {len(neg_images)} hard negatives from {args.negatives_out}")
        else:
            print("neg: run --materialize-negatives first to extract the bird archive")

    mixed: list[Path] = []
    report = {}
    for name, images in sources.items():
        picked = sample(images, quota.get(name, 0), rng)
        mixed.extend(picked)
        report[name] = {"available": len(images), "used": len(picked), "quota": quota.get(name, 0)}

    total = len(mixed)
    print("\n--- mix ---")
    for name, rec in report.items():
        print(f"  {name:>7}: {rec['used']:6d} / {rec['available']:6d}  ({rec['used'] / max(total, 1):.1%})")
    print(f"  {'total':>7}: {total:6d}")

    if not args.write:
        print("\ndry run: pass --write to emit the image list and data YAML")
        return

    out = root / "splits"
    out.mkdir(parents=True, exist_ok=True)
    listing = out / "train_mixed.txt"
    listing.write_text("".join(f"{p.as_posix()}\n" for p in mixed), encoding="utf-8")
    (out / "data_mixed.yaml").write_text(
        "\n".join(
            [
                "# Mixed training split: in-domain ARD-MAV + size-filtered auxiliary RGB sources",
                f"# generated by tools/ard_mav/make_mixed_train_list.py (max_net_side={args.max_net_side} @ {args.imgsz})",
                f"path: {root.as_posix()}",
                "train: splits/train_mixed.txt",
                "val: splits/val.txt",
                "test: splits/test.txt",
                "nc: 1",
                "names: ['Drone']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "mixed_manifest.json").write_text(
        json.dumps({"imgsz": args.imgsz, "max_net_side": args.max_net_side, "sources": report, "total": total}, indent=2),
        encoding="utf-8",
    )
    print(f"\nlist   -> {listing.as_posix()}")
    print(f"data   -> {(out / 'data_mixed.yaml').as_posix()}")


if __name__ == "__main__":
    main()
