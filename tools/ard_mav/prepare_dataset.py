# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the ARD-MAV VOC-style annotations + videos into a YOLO detection dataset.

The dataset ships only ``Annotations/<seq>/*.xml`` and ``videos/<seq>.mp4``; every XML
names the frame (``<seq>_<idx>.jpg``) that was annotated. Frames are decoded from the
matching video in one sequential pass and written as JPEGs, so no unrelated frames are
extracted. Splits are made at sequence level, which keeps train/val leak-free.

Usage:
    python tools/ard_mav/prepare_dataset.py --stride 10
    python tools/ard_mav/prepare_dataset.py --stride 1 --out g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo-full
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

BUCKETS = ((0, 4, "<4px"), (4, 8, "4-8px"), (8, 16, "8-16px"), (16, 1e9, ">16px"))


def bucket_of(size: float) -> str:
    """Return the size bucket label for a box's longest side."""
    for lo, hi, name in BUCKETS:
        if lo <= size < hi:
            return name
    return ">16px"


def parse_annotation(xml_file: Path) -> dict:
    """Parse one VOC-style XML into image size + a list of ``(cls, x1, y1, x2, y2)`` boxes."""
    root = ET.parse(xml_file).getroot()
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        boxes.append(
            (
                name,
                float(bb.findtext("xmin")),
                float(bb.findtext("ymin")),
                float(bb.findtext("xmax")),
                float(bb.findtext("ymax")),
            )
        )
    return {"filename": root.findtext("filename"), "width": width, "height": height, "boxes": boxes}


def to_yolo_line(box: tuple, width: int, height: int, cls_id: int) -> str | None:
    """Convert a pixel xyxy box to a normalized YOLO label line, keeping all valid boxes."""
    _, x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def collect(ann_root: Path) -> tuple[dict, dict]:
    """Parse every XML once and return per-sequence frame records plus aggregate statistics."""
    sequences = {}
    stats = {
        "images": 0,
        "objects": 0,
        "empty_images": 0,
        "classes": Counter(),
        "resolutions": Counter(),
        "buckets": Counter(),
        "boxes_per_image": Counter(),
        "min_box_wh": [1e9, 1e9],
        "sequences": {},
    }
    for seq_dir in sorted(p for p in ann_root.iterdir() if p.is_dir()):
        seq = seq_dir.name
        frames = {}
        for xml_file in sorted(seq_dir.glob("*.xml")):
            rec = parse_annotation(xml_file)
            if rec["filename"] is None:  # frame id comes from the file name when <filename> is empty
                rec["filename"] = f"{xml_file.stem}.jpg"
            try:
                idx = int(Path(rec["filename"]).stem.split("_")[-1])
            except ValueError:
                raise ValueError(f"Cannot parse frame index from {rec['filename']} ({xml_file})") from None
            frames[idx] = rec

            stats["images"] += 1
            stats["objects"] += len(rec["boxes"])
            stats["buckets"]["no-object"] += int(not rec["boxes"])
            stats["boxes_per_image"][len(rec["boxes"])] += 1
            stats["resolutions"][(rec["width"], rec["height"])] += 1
            if not rec["boxes"]:
                stats["empty_images"] += 1
            for name, x1, y1, x2, y2 in rec["boxes"]:
                stats["classes"][name] += 1
                w, h = abs(x2 - x1), abs(y2 - y1)
                stats["buckets"][bucket_of(max(w, h))] += 1
                stats["min_box_wh"][0] = min(stats["min_box_wh"][0], w)
                stats["min_box_wh"][1] = min(stats["min_box_wh"][1], h)
        sequences[seq] = frames
        stats["sequences"][seq] = {
            "frames": len(frames),
            "first": min(frames),
            "last": max(frames),
            "objects": sum(len(f["boxes"]) for f in frames.values()),
        }
    return sequences, stats


def make_splits(sequences: list[str], val_every: int) -> tuple[list[str], list[str]]:
    """Split sequences deterministically so no sequence appears in both train and val."""
    if val_every <= 0:
        raise ValueError("--val-every must be >= 1")
    val = [s for i, s in enumerate(sequences) if i % val_every == 0]
    train = [s for s in sequences if s not in val]
    return train, val


def extract_sequence(
    seq: str,
    frames: dict,
    video: Path,
    image_dir: Path,
    label_dir: Path,
    stride: int,
    jpeg_quality: int,
    cls_map: dict,
) -> tuple[int, int]:
    """Decode the annotated frames of one video and write JPEGs + YOLO labels."""
    wanted = sorted(idx for idx in frames if (idx - 1) % stride == 0)
    wanted_set = set(wanted)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    written = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_idx = max(wanted)
    if max_idx > frame_count:
        raise RuntimeError(f"{seq}: annotation index {max_idx} exceeds video frames {frame_count}")
    idx = 0
    while idx < max_idx:
        ok = cap.grab()
        if not ok:
            break
        idx += 1  # cv2 frame 0 corresponds to annotation index 1
        if idx not in wanted_set:
            continue
        ok, img = cap.retrieve()
        if not ok or img is None:
            raise RuntimeError(f"{seq}: cannot decode frame {idx}")
        rec = frames[idx]
        if img.shape[1] != rec["width"] or img.shape[0] != rec["height"]:
            raise RuntimeError(
                f"{seq} frame {idx}: video {img.shape[1]}x{img.shape[0]} != annotation {rec['width']}x{rec['height']}"
            )
        name = f"{seq}_{idx:04d}"
        cv2.imwrite(str(image_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        lines = []
        for box in rec["boxes"]:
            line = to_yolo_line(box, rec["width"], rec["height"], cls_map[box[0]])
            if line:
                lines.append(line)
        (label_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    cap.release()
    return written, len(wanted)


def main() -> None:
    """Run the ARD-MAV to YOLO conversion and write ``data.yaml`` plus ``stats.json``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g:/0-rk3588/train/ARD-MAV"))
    parser.add_argument("--out", type=Path, default=Path("g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo"))
    parser.add_argument("--stride", type=int, default=10, help="keep 1 of every N annotated frames")
    parser.add_argument("--val-every", type=int, default=5, help="every N-th sequence goes to val")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit-sequences", type=int, default=0, help="debug: only convert the first N sequences")
    parser.add_argument(
        "--stats-only", action="store_true", help="only write data.yaml/stats.json, skip frame extraction"
    )
    args = parser.parse_args()

    ann_root, video_root = args.root / "Annotations", args.root / "videos"
    sequences, stats = collect(ann_root)
    seq_names = sorted(sequences)
    if args.limit_sequences:
        seq_names = seq_names[: args.limit_sequences]
    names = sorted(stats["classes"])
    cls_map = {name: i for i, name in enumerate(names)}
    train_seqs, val_seqs = make_splits(seq_names, args.val_every)

    out = args.out
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = {"train": [0, 0, len(train_seqs)], "val": [0, 0, len(val_seqs)]}
    if args.stats_only:
        for split in ("train", "val"):
            split_counts[split][0] = len(list((out / "images" / split).glob("*.jpg")))
    for i, seq in enumerate([] if args.stats_only else train_seqs + val_seqs):
        split = "train" if seq in train_seqs else "val"
        video = video_root / f"{seq}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"Missing video for {seq}: {video}")
        done, planned = extract_sequence(
            seq,
            sequences[seq],
            video,
            out / "images" / split,
            out / "labels" / split,
            args.stride,
            args.jpeg_quality,
            cls_map,
        )
        split_counts[split][0] += done
        split_counts[split][1] += planned
        print(f"[{i + 1}/{len(seq_names)}] {seq} -> {split}: {done}/{planned} frames", flush=True)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "# ARD-MAV tiny UAV detection dataset, generated by tools/ard_mav/prepare_dataset.py\n"
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )

    stats["resolutions"] = {f"{w}x{h}": c for (w, h), c in stats["resolutions"].items()}
    stats["split"] = {
        "train_sequences": train_seqs,
        "val_sequences": val_seqs,
        "train_images": split_counts["train"][0],
        "val_images": split_counts["val"][0],
        "frame_stride": args.stride,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(
        f"\nDone: {split_counts['train'][0]} train / {split_counts['val'][0]} val images"
        f" | classes={names} | data={data_yaml}"
    )


if __name__ == "__main__":
    main()
