# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Audit and evaluate a detector on the UAV-Anti-UAV single-object-tracking test split.

The split ships one directory per sequence holding `groundtruth_rect.txt` (`x,y,w,h` per frame), `absent.txt` (one
binary flag per frame), `attributes.txt` (14 unnamed binary flags) and one mp4. It is an SOT benchmark, so it can
measure "does the box land on the one known drone" but it cannot measure precision in the detection sense: there are
no negative images, no multi-object frames and no class labels. What it can measure is recall-style hit rate, centre
error, success AUC and behaviour on `absent` frames (any box there is a false positive).

Two decode modes are reported. `maxconf` takes the highest-confidence box of every frame independently, which is what
the board runs today. `sot` adds a one-step motion gate: the highest-confidence box that overlaps the previous pick
(IoU >= --gate) wins, falling back to the global maximum when nothing overlaps. That gate is a proxy for the online
tracker, not a tracker.

Usage:
    python tools/ard_mav/eval_sot_sequences.py --audit-only
    python tools/ard_mav/eval_sot_sequences.py --weights g:/.../r4_dut640_synth/weights/best.pt \
        --weights g:/.../r3_dut640-2/weights/best.pt --stride 10 --conf 0.15
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

ROOT = Path("G:/0-rk3588/train/uav-anti-uav/UAV-Anti-UAV/Test")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--weights", type=Path, action="append", default=None)
    p.add_argument("--imgsz", action="append", type=int, default=None)
    p.add_argument("--stride", type=int, default=10, help="evaluate every Nth frame")
    p.add_argument("--conf", type=float, default=0.15, help="working point for hit / FP counting")
    p.add_argument("--gate", type=float, default=0.2, help="IoU with the previous pick that keeps a box in `sot` mode")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--max-seq", type=int, default=0)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--csv", type=Path, default=None)
    return p.parse_args()


def read_gt(path: Path) -> list[tuple[float, float, float, float]]:
    """Read `x,y,w,h` rows, keeping zero rows as absent markers."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.replace("\t", ",").split(",")
        if len(parts) >= 4:
            try:
                rows.append(tuple(float(v) for v in parts[:4]))
            except ValueError:
                continue
    return rows


def read_flags(path: Path) -> list[int]:
    """Read a one-value-per-line binary flag file."""
    if not path.is_file():
        return []
    return [int(float(v)) for v in path.read_text(encoding="utf-8", errors="ignore").split() if v.strip()]


def video_meta(path: Path) -> tuple[int, int, int]:
    """Return (frames, width, height) without decoding the clip."""
    cap = cv2.VideoCapture(str(path))
    meta = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return meta


def audit(root: Path, max_seq: int) -> list[dict]:
    """Check that GT, absent flags and the video agree, and collect size/attribute statistics."""
    seqs = sorted(d for d in root.iterdir() if d.is_dir())
    if max_seq:
        seqs = seqs[:max_seq]
    out, attrs = [], defaultdict(int)
    for seq in seqs:
        gt = read_gt(seq / "groundtruth_rect.txt")
        absent = read_flags(seq / "absent.txt")
        attr = read_flags(seq / "attributes.txt")
        video = next(iter(seq.glob("*.mp4")), None)
        frames, w, h = video_meta(video) if video else (0, 0, 0)
        present = [b for b in gt if b[2] > 0 and b[3] > 0]
        sides = [float(np.sqrt(b[2] * b[3])) for b in present]
        for i, v in enumerate(attr):
            attrs[i] += int(v == 1)
        out.append(
            {
                "seq": seq.name,
                "gt_rows": len(gt),
                "absent_rows": len(absent),
                "absent_ones": sum(absent),
                "video_frames": frames,
                "width": w,
                "height": h,
                "present": len(present),
                "gt_zero_rows": sum(1 for b in gt if b[2] <= 0 or b[3] <= 0),
                "side_med": round(float(np.median(sides)), 1) if sides else 0.0,
                "side_p10": round(float(np.percentile(sides, 10)), 1) if sides else 0.0,
                "side_p90": round(float(np.percentile(sides, 90)), 1) if sides else 0.0,
                "attrs": "".join(str(v) for v in attr),
                "aligned": int(len(gt) == frames),
                "absent_matches_zero": int(sum(absent) == sum(1 for b in gt if b[2] <= 0 or b[3] <= 0)),
            }
        )
    print(f"sequences={len(out)}")
    if out:
        print(
            f"gt rows total={sum(r['gt_rows'] for r in out)} mean={np.mean([r['gt_rows'] for r in out]):.1f} "
            f"min={min(r['gt_rows'] for r in out)} max={max(r['gt_rows'] for r in out)}"
        )
        print(
            f"aligned(gt==video frames)={sum(r['aligned'] for r in out)}/{len(out)}  "
            f"absent_ones==gt_zero_rows={sum(r['absent_matches_zero'] for r in out)}/{len(out)}"
        )
        total = sum(r["gt_rows"] for r in out)
        print(f"absent frames={sum(r['absent_ones'] for r in out)}/{total} = {sum(r['absent_ones'] for r in out) / max(total, 1):.3f}")
        print(f"resolutions={sorted({(r['width'], r['height']) for r in out})}")
        print(f"median target side per sequence: p10={np.percentile([r['side_med'] for r in out], 10):.1f} "
              f"median={np.median([r['side_med'] for r in out]):.1f} p90={np.percentile([r['side_med'] for r in out], 90):.1f}")
        print("attribute positive rate:", {i: round(v / len(out), 2) for i, v in sorted(attrs.items())})
    return out


def iou_xywh(a, b) -> float:
    """IoU between two (x, y, w, h) boxes."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def iou_xyxy_xywh(p, g) -> float:
    """IoU between a predicted xyxy box and a GT (x, y, w, h) box."""
    iw = max(0.0, min(p[2], g[0] + g[2]) - max(p[0], g[0]))
    ih = max(0.0, min(p[3], g[1] + g[3]) - max(p[1], g[1]))
    inter = iw * ih
    return inter / ((p[2] - p[0]) * (p[3] - p[1]) + g[2] * g[3] - inter + 1e-9)


def evaluate_sequence(model: YOLO, seq: Path, args, imgsz: int) -> dict:
    """Run one sequence and return hit / centre-error / success / absent-frame statistics."""
    gt = read_gt(seq / "groundtruth_rect.txt")
    absent = read_flags(seq / "absent.txt")
    video = next(iter(seq.glob("*.mp4")), None)
    cap = cv2.VideoCapture(str(video))
    stats = {
        "frames": 0,
        "present": 0,
        "absent": 0,
        "hit25": 0,
        "hit50": 0,
        "hit25_sot": 0,
        "hit50_sot": 0,
        "fp_absent": 0,
        "box_sum": 0,
        "iou_sum": 0.0,
        "success": np.zeros(21),
        "cerr": [],
        "cerr_norm": [],
    }
    prev = None
    chunk, chunk_idx = [], []
    idx = 0

    def flush() -> None:
        nonlocal chunk, chunk_idx, prev
        if not chunk:
            return
        results = model.predict(chunk, imgsz=imgsz, conf=min(0.001, args.conf), max_det=300, device=args.device, verbose=False)
        for local, result in enumerate(results):
            i = chunk_idx[local]
            boxes = result.boxes.data.cpu().numpy() if len(result.boxes) else np.zeros((0, 6), np.float32)
            if len(boxes):
                boxes = boxes[boxes[:, 4] >= args.conf]
            is_present = i < len(gt) and gt[i][2] > 0 and gt[i][3] > 0 and not (i < len(absent) and absent[i])
            if is_present:
                stats["present"] += 1
                if len(boxes):
                    j = int(np.argmax(boxes[:, 4]))
                    best = boxes[j]
                    prev_xywh = (best[0], best[1], best[2] - best[0], best[3] - best[1])
                    iou = iou_xyxy_xywh(best, gt[i])
                    stats["hit25"] += int(iou >= 0.25)
                    stats["hit50"] += int(iou >= 0.5)
                    stats["iou_sum"] += iou
                    stats["success"] += (iou > np.arange(21) * 0.05).astype(float)
                    cx, cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
                    stats["cerr"].append(float(np.hypot(cx - (gt[i][0] + gt[i][2] / 2), cy - (gt[i][1] + gt[i][3] / 2))))
                    stats["cerr_norm"].append(
                        float(np.hypot(cx - (gt[i][0] + gt[i][2] / 2), cy - (gt[i][1] + gt[i][3] / 2)))
                        / np.hypot(result.orig_shape[1], result.orig_shape[0])
                    )
                    # motion gate: prefer the box that overlaps the previous pick
                    if prev is not None:
                        cand = [k for k in range(len(boxes)) if iou_xywh(prev_xywh, (
                            boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1])) >= args.gate]
                        if cand:
                            j2 = max(cand, key=lambda k: boxes[k][4])
                            gated = boxes[j2]
                            iou2 = iou_xyxy_xywh(gated, gt[i])
                            stats["hit25_sot"] += int(iou2 >= 0.25)
                            stats["hit50_sot"] += int(iou2 >= 0.5)
                    prev = prev_xywh
                else:
                    prev = None
                stats["box_sum"] += len(boxes)
            elif i < len(gt):
                stats["absent"] += 1
                stats["fp_absent"] += int(len(boxes) > 0)
                prev = None
        chunk, chunk_idx = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            chunk.append(frame)
            chunk_idx.append(idx)
        idx += 1
        stats["frames"] += 1
        if len(chunk) >= args.batch:
            flush()
    flush()
    cap.release()

    stats["hit25"] /= max(stats["present"], 1)
    stats["hit50"] /= max(stats["present"], 1)
    stats["hit25_sot"] /= max(stats["present"], 1)
    stats["hit50_sot"] /= max(stats["present"], 1)
    stats["success_auc"] = float(np.mean(stats["success"] / max(stats["present"], 1)))
    stats["iou_mean"] = stats["iou_sum"] / max(stats["present"], 1)
    stats["cerr_med"] = float(np.median(stats["cerr"])) if stats["cerr"] else 0.0
    stats["cerr_norm_med"] = float(np.median(stats["cerr_norm"])) if stats["cerr_norm"] else 0.0
    stats["fp_absent_rate"] = stats["fp_absent"] / max(stats["absent"], 1)
    stats["box_per_frame"] = stats["box_sum"] / max(stats["present"] + stats["absent"], 1)
    return stats


def main() -> None:
    """Audit the split, then evaluate every checkpoint over every sequence."""
    args = parse_args()
    rows = audit(args.root, args.max_seq)
    if args.audit_only or not args.weights:
        return

    sizes = args.imgsz or [640] * len(args.weights)
    if len(sizes) != len(args.weights):
        raise SystemExit("--imgsz must be given once per --weights")
    out = []
    for weights, imgsz in zip(args.weights, sizes):
        model = YOLO(str(weights))
        name = weights.parent.parent.name
        print(f"\n== {name} @ {imgsz} stride={args.stride} conf>={args.conf} ==")
        print(f"{'seq':<32}{'present':>8}{'hit@.25':>9}{'hit@.5':>8}{'sot@.25':>9}{'sot@.5':>8}{'AUC':>7}{'cErr_px':>9}{'FP_abs':>8}{'box/f':>7}")
        acc = defaultdict(list)
        for row in rows:
            stats = evaluate_sequence(model, args.root / row["seq"], args, imgsz)
            if not stats["present"]:
                continue
            print(
                f"{row['seq']:<32}{stats['present']:>8}{stats['hit25']:>9.3f}{stats['hit50']:>8.3f}"
                f"{stats['hit25_sot']:>9.3f}{stats['hit50_sot']:>8.3f}{stats['success_auc']:>7.3f}"
                f"{stats['cerr_med']:>9.1f}{stats['fp_absent_rate']:>8.2f}{stats['box_per_frame']:>7.2f}"
            )
            for key in ("hit25", "hit50", "hit25_sot", "hit50_sot", "success_auc", "iou_mean", "cerr_med",
                        "cerr_norm_med", "fp_absent_rate", "box_per_frame"):
                acc[key].append(stats[key])
            out.append({"model": name, "weights": weights.as_posix(), "imgsz": imgsz, "seq": row["seq"],
                        "stride": args.stride, "conf": args.conf, **{k: round(v, 4) for k, v in stats.items()
                        if isinstance(v, (int, float))}})
        print("-" * 100)
        for key in ("hit25", "hit50", "hit25_sot", "hit50_sot", "success_auc", "iou_mean", "cerr_med", "fp_absent_rate", "box_per_frame"):
            v = np.array(acc[key])
            print(f"{key:<14} mean={v.mean():.4f} median={np.median(v):.4f} p10={np.percentile(v, 10):.4f} p90={np.percentile(v, 90):.4f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out[0]))
            writer.writeheader()
            writer.writerows(out)
        print(f"wrote {len(out)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
