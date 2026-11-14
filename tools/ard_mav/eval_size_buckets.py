# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate an ARD-MAV checkpoint per target-size bucket, confidence threshold and sequence continuity.

mAP does not expose tiny-target behaviour, so this script matches predictions to ground truth at IoU 0.5 and reports,
for each size bucket:

- AP50 (average precision of the detections that belong to the bucket),
- recall (bucketed by ground-truth size),
- precision (bucket of a TP from its GT size, bucket of an FP from its predicted size),
- FP/frame, so precision is readable at the operating point the detector will actually run at,
- sequence continuity: adjacent-frame co-detection rate and detected-segment breaks, a tracking-free proxy for track
  fragmentation.

Two size metrics are available. `--size-metric side` measures the box's longest side on the original image (the
historical default). `--size-metric sqrt_area --size-space input` measures `sqrt(w*h)` after the letterbox that the
network actually saw, which is the only口径 comparable to a 640 px network input: a 30 px box on a 1920 px frame
becomes `30 * 640/1920 = 10 px`.

Inference runs once at a low confidence floor; every threshold in the sweep is applied to those boxes, so one pass
produces the full table. Rows are appended to one CSV so several experiments compare side by side.

Usage:
    python tools/ard_mav/eval_size_buckets.py --weights g:/.../runs/r3_dut640-2/weights/best.pt --split val
    python tools/ard_mav/eval_size_buckets.py --weights ... --size-metric sqrt_area --size-space input \
        --data g:/0-rk3588/train/ARD-MAV/mixed_r4_640/data.yaml --experiment r4_640
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

DEFAULT_EDGES = {"side": (4, 8, 16, 32), "sqrt_area": (8, 16, 32, 64, 128)}
CONF_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def make_buckets(edges: tuple[float, ...]) -> list[tuple[float, float, str]]:
    """Turn ascending edges into (low, high, label) buckets."""
    buckets, low = [], 0.0
    for edge in edges:
        buckets.append((low, edge, f"<{edge:g}px" if low == 0 else f"{low:g}-{edge:g}px"))
        low = edge
    buckets.append((low, float("inf"), f">={low:g}px"))
    return buckets


def bucket_of(size: float, buckets: list[tuple[float, float, str]]) -> str:
    """Return the bucket label for a box size."""
    return next(name for lo, hi, name in buckets if lo <= size < hi)


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized IoU between one box (4,) and N boxes (N, 4), all in xyxy."""
    lt = torch.maximum(a[:2], b[:, :2])
    rb = torch.minimum(a[2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def average_precision(scores: list[float], flags: list[int], n_gt: int) -> float:
    """101-point interpolated AP from confidence-ordered (score, is_true_positive) pairs."""
    if n_gt == 0 or not scores:
        return 0.0
    order = torch.argsort(torch.tensor(scores, dtype=torch.float32), descending=True)
    tp = torch.cumsum(torch.tensor(flags, dtype=torch.float32)[order], 0)
    fp = torch.cumsum(1 - torch.tensor(flags, dtype=torch.float32)[order], 0)
    rec = tp / n_gt
    prec = tp / (tp + fp).clamp(min=1e-9)
    for i in range(len(prec) - 2, -1, -1):  # precision envelope, right to left
        prec[i] = torch.maximum(prec[i], prec[i + 1])
    idx = torch.searchsorted(rec, torch.linspace(0, 1, 101))
    return float(torch.tensor([prec[int(i)] if int(i) < len(prec) else 0.0 for i in idx]).mean())


def local_path(p) -> str:
    """Map a Windows drive path to ``/mnt/<drive>/...`` on POSIX so one dataset YAML works on Windows and in WSL.

    The ARD-MAV YAMLs and image lists are authored on Windows (``g:/0-rk3588/...``). Running the same evaluation
    inside WSL otherwise resolves them as relative paths and fails with FileNotFoundError.
    """
    s = str(p)
    if os.name != "nt" and len(s) > 2 and s[1] == ":" and s[0].isalpha():
        rest = s[2:].lstrip("/\\")
        return f"/mnt/{s[0].lower()}/{rest}"
    return s


def read_split(data_yaml: Path, split: str) -> list[Path]:
    """Resolve the image list of one split from a data YAML (`.txt` list or image directory)."""
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(local_path(cfg.get("path", Path(data_yaml).parent)))
    entry = cfg[split]
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for item in entries:
        path = Path(local_path(item))
        path = path if path.is_absolute() else root / path
        if path.suffix.lower() == ".txt":
            images += [
                Path(local_path(line.strip()))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            images += sorted(path.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images resolved for split '{split}' in {data_yaml}")
    return images


def label_path(image: Path) -> Path:
    """Map an image path to its YOLO label path (last /images/ -> /labels/, .jpg -> .txt)."""
    return Path(image.as_posix().replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def read_gt(image: Path, width: int, height: int, gain: float, metric: str) -> tuple[torch.Tensor, list[float]]:
    """Return GT boxes (N, 4) in xyxy pixels and their sizes after the letterbox `gain`."""
    path = label_path(image)
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append([float(v) for v in parts[1:]])
    boxes = torch.tensor(
        [[(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height]
         for cx, cy, bw, bh in rows],
        dtype=torch.float32,
    ).reshape(-1, 4)
    sizes = [box_size(b, gain, metric) for b in boxes]
    return boxes, sizes


def box_size(box: torch.Tensor, gain: float, metric: str) -> float:
    """Measure one box: longest side or sqrt(area), multiplied by the letterbox gain."""
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    return (max(w, h) if metric == "side" else (w * h) ** 0.5) * gain


def match(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> list[int]:
    """Greedy confidence-ordered matching; returns the GT index of each matched prediction (-1 = FP).

    Args:
        pred (torch.Tensor): Predictions (N, 6) as xyxy, conf, cls, already filtered by confidence.
        gt (torch.Tensor): Ground-truth boxes (M, 4) in xyxy.
        threshold (float): IoU threshold for a match.

    Returns:
        (list[int]): One GT index per prediction, -1 when unmatched.
    """
    assigned = [-1] * len(pred)
    if not len(pred) or not len(gt):
        return assigned
    free = torch.ones(len(gt), dtype=torch.bool)
    for i in torch.argsort(pred[:, 4], descending=True).tolist():
        scores = iou(pred[i, :4], gt)
        scores[~free] = 0
        j = int(scores.argmax())
        if float(scores[j]) >= threshold:
            assigned[i] = j
            free[j] = False
    return assigned


def main() -> None:
    """Run prediction once and report the bucket (AP, recall, precision), threshold and continuity tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", default="g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/splits/data.yaml")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a TP")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU used during prediction")
    parser.add_argument("--conf-floor", type=float, default=0.001, help="lowest conf kept for the sweep")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--experiment", default="", help="label written to the CSV; defaults to the run name")
    parser.add_argument("--continuity-conf", type=float, default=0.15, help="threshold used for sequence metrics")
    parser.add_argument("--size-metric", choices=("side", "sqrt_area"), default="side")
    parser.add_argument(
        "--size-space",
        choices=("orig", "input"),
        default="orig",
        help="orig: size on the original image; input: size after the letterbox to --imgsz",
    )
    parser.add_argument("--edges", default="", help="comma-separated bucket edges; defaults depend on --size-metric")
    parser.add_argument("--csv", type=Path, help="append one row per threshold and bucket to this CSV")
    args = parser.parse_args()

    edges = tuple(float(e) for e in args.edges.split(",") if e.strip()) or DEFAULT_EDGES[args.size_metric]
    buckets = make_buckets(edges)
    bucket_names = [name for _, _, name in buckets] + ["all"]

    images = read_split(Path(args.data), args.split)
    experiment = args.experiment or args.weights.parent.parent.name
    model = YOLO(str(args.weights))

    # threshold -> bucket -> counters
    stats = {t: defaultdict(lambda: {"gt": 0, "tp": 0, "pred": 0, "fp": 0}) for t in CONF_SWEEP}
    ap_scores = defaultdict(list)  # bucket -> confidences of every prediction assigned to it
    ap_flags = defaultdict(list)  # bucket -> 1 when that prediction matched a GT
    frame_hits = {t: {} for t in CONF_SWEEP}  # threshold -> {image stem: bool}
    for start in range(0, len(images), args.batch):
        batch = images[start : start + args.batch]
        results = model.predict(
            [str(p) for p in batch],
            imgsz=args.imgsz,
            conf=args.conf_floor,
            iou=args.nms_iou,
            max_det=args.max_det,
            verbose=False,
            device=args.device,
        )
        for image, result in zip(batch, results):
            h, w = result.orig_shape
            gain = 1.0 if args.size_space == "orig" else min(args.imgsz / h, args.imgsz / w)
            gt, gt_sizes = read_gt(image, w, h, gain, args.size_metric)
            gt_buckets = [bucket_of(s, buckets) for s in gt_sizes]
            pred = result.boxes.data.cpu()
            pred_buckets = [bucket_of(box_size(b, gain, args.size_metric), buckets) for b in pred]
            matched_all = match(pred, gt, args.iou)
            for t in CONF_SWEEP:
                keep = pred[:, 4] >= t if len(pred) else torch.zeros(0, dtype=torch.bool)
                kept = torch.nonzero(keep).flatten().tolist()
                assigned = {i: matched_all[i] for i in kept}
                for name in bucket_names:
                    stats[t][name]["gt"] += sum(b == name or name == "all" for b in gt_buckets)
                for i in kept:
                    name = pred_buckets[i]
                    if assigned[i] >= 0:
                        stats[t][gt_buckets[assigned[i]]]["tp"] += 1
                        stats[t][gt_buckets[assigned[i]]]["pred"] += 1
                        stats[t]["all"]["tp"] += 1
                        stats[t]["all"]["pred"] += 1
                    else:
                        stats[t][name]["fp"] += 1
                        stats[t][name]["pred"] += 1
                        stats[t]["all"]["fp"] += 1
                        stats[t]["all"]["pred"] += 1
                frame_hits[t][image.stem] = any(v >= 0 for v in assigned.values())
            for i in range(len(pred)):  # AP uses every prediction at the conf floor, in its own bucket
                name = gt_buckets[matched_all[i]] if matched_all[i] >= 0 else pred_buckets[i]
                ap_scores[name].append(float(pred[i, 4]))
                ap_flags[name].append(int(matched_all[i] >= 0))
                ap_scores["all"].append(float(pred[i, 4]))
                ap_flags["all"].append(int(matched_all[i] >= 0))

    ap = {name: average_precision(ap_scores[name], ap_flags[name], stats[CONF_SWEEP[0]][name]["gt"]) for name in bucket_names}

    rows = []
    for t in CONF_SWEEP:
        for name in bucket_names:
            s = stats[t][name]
            recall = s["tp"] / max(s["gt"], 1)
            precision = s["tp"] / max(s["pred"], 1)
            rows.append(
                {
                    "experiment": experiment,
                    "weights": args.weights.as_posix(),
                    "imgsz": args.imgsz,
                    "split": args.split,
                    "conf": t,
                    "iou": args.iou,
                    "size_metric": f"{args.size_metric}@{'input' if args.size_space == 'input' else 'orig'}",
                    "bucket": name,
                    "images": len(images),
                    "gt": s["gt"],
                    "tp": s["tp"],
                    "pred": s["pred"],
                    "fp": s["fp"],
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "ap50": round(ap[name], 4),
                    "fp_per_frame": round(s["fp"] / max(len(images), 1), 4),
                }
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv.is_file()
        with args.csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    print(
        f"experiment={experiment} weights={args.weights} split={args.split} images={len(images)} imgsz={args.imgsz} "
        f"size={args.size_metric}@{args.size_space}"
    )
    print("bucket        gt     AP50   recall@0.15  precision@0.15  fp/frame@0.15")
    for name in bucket_names:
        s = stats[0.15][name]
        print(
            f"{name:<12}{s['gt']:>6}{ap[name]:>9.4f}{s['tp'] / max(s['gt'], 1):>15.4f}"
            f"{s['tp'] / max(s['pred'], 1):>16.4f}{s['fp'] / max(len(images), 1):>15.3f}"
        )
    print("  conf |  recall  precision  fp/frame")
    for t in CONF_SWEEP:
        s = stats[t]["all"]
        print(
            f"  {t:4.2f} |  {s['tp'] / max(s['gt'], 1):.4f}   {s['tp'] / max(s['pred'], 1):.4f}    "
            f"{s['fp'] / max(len(images), 1):7.3f}"
        )

    # sequence continuity: a frame "hits" when any prediction matches a GT at `continuity-conf`
    seqs = defaultdict(list)
    for image in images:
        seqs[image.stem.rpartition("_")[0]].append(image.stem)
    hits = frame_hits[args.continuity_conf]
    pairs = both = frames = hit_frames = segments = sequences_with_hits = 0
    for stems in seqs.values():
        ordered = sorted(stems, key=lambda s: int(s.rpartition("_")[2]) if s.rpartition("_")[2].isdigit() else 0)
        flags = [hits[s] for s in ordered]
        frames += len(flags)
        hit_frames += sum(flags)
        pairs += max(len(flags) - 1, 0)
        both += sum(flags[i] and flags[i - 1] for i in range(1, len(flags)))
        runs = sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))
        segments += runs
        sequences_with_hits += int(runs > 0)
    print(
        f"continuity(conf={args.continuity_conf}): sequences={len(seqs)} frame_hit={hit_frames / max(frames, 1):.3f} "
        f"adjacent_co_hit={both / max(pairs, 1):.3f} detected_segments={segments} "
        f"breaks={max(segments - sequences_with_hits, 0)}"
    )
    print("definitions: frame_hit = frame with >=1 IoU-matched GT; adjacent_co_hit = hit frame whose previous frame also hit;")
    print("             breaks = detected_segments - sequences_with_hits (one continuous run per sequence is 0 breaks)")


if __name__ == "__main__":
    main()
