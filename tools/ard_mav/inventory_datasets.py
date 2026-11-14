# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Sample target-size distributions of the candidate RGB UAV datasets before any merge decision.

Merging datasets changes the validation distribution, so this script only measures: image count, annotation format,
classes and the longest-side size buckets of a random label sample. It never writes to the datasets.

Usage:
    python tools/ard_mav/inventory_datasets.py --sample 400
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

BUCKETS = ((0, 4, "<4px"), (4, 8, "4-8px"), (8, 16, "8-16px"), (16, 32, "16-32px"), (32, 1e9, ">32px"))
YOLO_DATASETS = {
    "ARD-MAV": Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo"),
    "DUT-Anti-UAV": Path("g:/0-rk3588/train/DUT-Anti-UAV/yolo_dataset"),
    "3rd_anti_uav": Path("g:/0-rk3588/train/3rd_anti_uav/yolo_dataset"),
    "uav-anti-uav": Path("g:/0-rk3588/train/uav-anti-uav/UAV-Anti-UAV/yolo_dataset"),
}
VISIODECT = Path("g:/0-rk3588/train/visiodet/VisioDECT Dataset Upload")
VISIODECT_MODELS = ("Anafi-Extended", "DJIFPV", "DJIPhantom", "EFT-E410S", "Mavic_Air", "Mavic_Enterprise")
VISIODECT_LIGHTS = ("cloudy", "evening", "sunny")


def bucket_of(size: float) -> str:
    """Return the size-bucket label for a box's longest side in original-image pixels."""
    return next(name for lo, hi, name in BUCKETS if lo <= size < hi)


def image_size(path: Path) -> tuple[int, int]:
    """Read (width, height) without decoding pixels beyond the header."""
    with Image.open(path) as im:
        return im.size


def dominant_size(image_dir: Path, labels: list[Path]) -> tuple[dict[str, int], list[tuple[Path, int, int]]]:
    """Pair each sampled label with its image size, returning (resolution counts, [(label, w, h)])."""
    counts, pairs = Counter(), []
    for label in labels:
        image = image_dir / f"{label.stem}.jpg"
        if not image.is_file():
            continue
        width, height = image_size(image)
        counts[f"{width}x{height}"] += 1
        pairs.append((label, width, height))
    return dict(counts), pairs


def add_yolo_labels(pairs: list[tuple[Path, int, int]], rec: dict) -> None:
    """Accumulate YOLO label rows into bucket and per-image box counters."""
    for label, width, height in pairs:
        rec["sampled_images"] += 1
        lines = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        boxes = [ln for ln in lines if len(ln) == 5]
        rec["boxes"] += len(boxes)
        rec["empty_images"] += int(not boxes)
        rec["class_ids"].update(int(b[0]) for b in boxes)
        for _, _, _, bw, bh in boxes:
            rec["buckets"][bucket_of(max(float(bw) * width, float(bh) * height))] += 1


def add_voc_labels(files: list[Path], rec: dict) -> None:
    """Accumulate VOC annotations (bbox in absolute pixels) into bucket counters."""
    for xml in files:
        rec["sampled_images"] += 1
        root = ET.parse(xml).getroot()
        objects = root.findall("object")
        rec["boxes"] += len(objects)
        rec["empty_images"] += int(not objects)
        for obj in objects:
            rec["names"].update({obj.findtext("name") or "unknown": 1})
            box = obj.find("bndbox")
            x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            rec["buckets"][bucket_of(max(x2 - x1, y2 - y1))] += 1


def empty_rec() -> dict:
    """Fresh accumulator for one split or one dataset."""
    return {
        "sampled_images": 0,
        "boxes": 0,
        "empty_images": 0,
        "class_ids": Counter(),
        "names": Counter(),
        "buckets": defaultdict(int),
    }


def finish(rec: dict) -> dict:
    """Convert counters into report-friendly numbers."""
    total = max(rec["boxes"], 1)
    return {
        "sampled_images": rec["sampled_images"],
        "boxes": rec["boxes"],
        "empty_image_share": round(rec["empty_images"] / max(rec["sampled_images"], 1), 4),
        "boxes_per_image": round(rec["boxes"] / max(rec["sampled_images"], 1), 3),
        "buckets_pct": {b[2]: round(100 * rec["buckets"].get(b[2], 0) / total, 1) for b in BUCKETS},
        "tiny_le16_pct": round(
            100 * sum(rec["buckets"].get(b[2], 0) for b in BUCKETS[:3]) / total,
            1,
        ),
        "class_ids": dict(sorted(rec["class_ids"].items())),
        "names": dict(rec["names"].most_common(12)),
    }


def scan_yolo(name: str, root: Path, sample: int, seed: int) -> dict:
    """Inventory one YOLO-format dataset directory."""
    out = {"exists": root.is_dir(), "root": root.as_posix(), "format": "yolo txt"}
    if not root.is_dir():
        return out
    yaml = root / "data.yaml"
    out["data_yaml"] = yaml.read_text(encoding="utf-8").strip() if yaml.is_file() else None
    out["splits"] = {}
    for split in ("train", "val", "test"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir():
            continue
        images = list(image_dir.glob("*.jpg"))
        labels = list(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        resolutions, pairs = dominant_size(image_dir, labels)
        rng = random.Random(f"{seed}-{name}-{split}")
        picked = rng.sample(pairs, min(sample, len(pairs))) if pairs else []
        rec = empty_rec()
        add_yolo_labels(picked, rec)
        out["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "resolution": max(resolutions, key=resolutions.get) if resolutions else "",
            "resolution_sample": resolutions,
            **finish(rec),
        }
    return out


def scan_visiodect(sample: int, seed: int) -> dict:
    """Inventory VisioDECT, which stores YOLO txt and VOC xml under per-model, per-weather folders."""
    out = {"exists": VISIODECT.is_dir(), "root": VISIODECT.as_posix(), "format": "VOC xml + yolo txt"}
    if not VISIODECT.is_dir():
        return out
    pairs = []
    for model in VISIODECT_MODELS:
        for light in VISIODECT_LIGHTS:
            xml_dir = VISIODECT / model / "labels" / light / "voc"
            if xml_dir.is_dir():
                pairs.append((model, light, xml_dir, VISIODECT / model / "images" / light.capitalize()))
    total_images = sum(len(list(p.glob("*.jpg"))) for _, _, _, p in pairs if p.is_dir())
    rng = random.Random(f"{seed}-visiodect")
    per_dir = max(1, sample // max(len(pairs), 1))
    rec, resolutions = empty_rec(), Counter()
    out["models"] = {}
    for model, light, xml_dir, image_dir in pairs:
        images = list(image_dir.glob("*.jpg")) if image_dir.is_dir() else []
        out["models"][model] = out["models"].get(model, 0) + len(images)
        if images:
            resolutions.update(f"{w}x{h}" for w, h in (image_size(p) for p in rng.sample(images, min(8, len(images)))))
        xml_files = sorted(xml_dir.glob("*.xml"))
        picked = rng.sample(xml_files, min(per_dir, len(xml_files)))
        add_voc_labels(picked, rec)
    out["images_total"] = total_images
    out["resolution_sample"] = dict(resolutions)
    out.update(finish(rec))
    return out


def main() -> None:
    """Print (and optionally save) the inventory as a table plus JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=400, help="label files sampled per split")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    inventory = {name: scan_yolo(name, root, args.sample, args.seed) for name, root in YOLO_DATASETS.items()}
    inventory["visiodet"] = scan_visiodect(args.sample, args.seed)
    inventory["ant_uav"] = {
        "exists": Path("g:/0-rk3588/train/ant_uav").is_dir(),
        "format": "mp4 + json attribute labels (no bounding boxes)",
    }

    for name, rec in inventory.items():
        if not rec.get("exists"):
            print(f"{name:>14}: MISSING")
            continue
        if "splits" not in rec:
            extra = f" images_total={rec['images_total']}" if "images_total" in rec else ""
            print(f"{name:>14}: exists, format={rec['format']}{extra}")
            if "buckets_pct" in rec:
                print(
                    f"  sample: boxes={rec['boxes']} tiny<=16px={rec['tiny_le16_pct']}% "
                    f"buckets={rec['buckets_pct']} names={rec['names']}"
                )
            continue
        print(f"{name:>14}: format={rec['format']}")
        for split, s in rec["splits"].items():
            print(
                f"  {split:>5}: images={s['images']:6d} labels={s['labels']:6d} res={s['resolution']} "
                f"b/i={s['boxes_per_image']:.2f} empty={s['empty_image_share']:.1%} tiny<=16px={s['tiny_le16_pct']}% "
                f"buckets={s['buckets_pct']}"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
