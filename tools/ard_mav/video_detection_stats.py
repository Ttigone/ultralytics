# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-video detection statistics on a real deployment clip: boxes per frame and detection persistence.

mAP cannot be measured on an unlabelled video, but two numbers are still comparable across checkpoints and they are
exactly the ones that separate the DUT model from the ARD-MAV model on Hard1.mp4:

    boxes/frame  how many boxes the detector emits per frame at a given confidence. A model that fires on grass and
                 treeline raises this while its real hits stay the same.
    persistence  of the frames that contain at least one box, the fraction whose previous sampled frame also had one.
                 A model seeing a real drone keeps it frame after frame; texture false positives flicker in and out.
                 This is the same definition used by `eval_size_buckets.py`.

Both are reported per confidence threshold from a single inference pass, because the boxes at the lowest threshold are
superset of every higher threshold. Recall and precision are NOT computed - pass annotated frames to
`eval_size_buckets.py` for that.

Every row is printed twice: `raw` is the detector alone, `post` adds the temporal stage a tracker performs online
(M-of-N confirmation, then interpolation of 1-2 frame gaps). The delta between the two is the ceiling of what moving
that confirmation into the deployment pipeline can recover, measured without touching board code.

Usage:
    python tools/ard_mav/video_detection_stats.py --video Hard1.mp4 \
        --weights G:/0-rk3588/train/DUT-Anti-UAV/runs/yolo26n_p2_p5_dut_640/weights/best.pt --imgsz 640 \
        --weights G:/0-rk3588/train/ARD-MAV/runs/r2_base_960-2/weights/best.pt --imgsz 960 \
        --conf 0.05,0.15,0.30 --stride 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

CONF_DEFAULT = (0.05, 0.15, 0.30)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, action="append", required=True, help="one checkpoint per --weights")
    p.add_argument("--imgsz", action="append", type=int, default=None, help="one input size per --weights")
    p.add_argument("--conf", default=",".join(str(c) for c in CONF_DEFAULT))
    p.add_argument("--stride", type=int, default=5, help="evaluate every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--temporal-m", type=int, default=3, help="M-of-N confirmation: minimum hits inside the window")
    p.add_argument("--temporal-n", type=int, default=5, help="M-of-N window length in sampled frames")
    p.add_argument("--max-gap", type=int, default=2, help="fill gaps of at most this many sampled frames")
    p.add_argument("--csv", type=Path, default=None, help="append one row per model/conf/stage to this CSV")
    return p.parse_args()


def read_frames(video: Path, stride: int, max_frames: int) -> tuple[list[np.ndarray], list[int]]:
    """Decode every `stride`-th frame, returning the frames and their absolute indices."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    frames, indices = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
            indices.append(idx)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, indices


def predict_confs(model: YOLO, frames: list[np.ndarray], imgsz: int, conf_floor: float, args) -> list[np.ndarray]:
    """Return per frame the (N,) confidence vector and (N,4) xyxy boxes at the lowest threshold."""
    per_frame = [np.zeros((0, 5), np.float32) for _ in frames]  # x1 y1 x2 y2 conf
    for start in range(0, len(frames), args.batch):
        chunk = frames[start : start + args.batch]
        results = model.predict(
            chunk, imgsz=imgsz, conf=conf_floor, max_det=300, device=args.device, verbose=False
        )
        for i, r in enumerate(results):
            boxes = r.boxes
            if boxes is None or not len(boxes):
                continue
            per_frame[start + i] = np.concatenate(
                [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None]], axis=1
            )
    return per_frame


def series(per_frame: list[np.ndarray], conf: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Per frame: hit flag, box count and mean box side at one confidence threshold."""
    hits = np.array([int((f[:, 4] >= conf).sum() > 0) for f in per_frame])
    counts = np.array([int((f[:, 4] >= conf).sum()) for f in per_frame])
    sides = [
        float(np.sqrt(np.maximum((f[:, 2] - f[:, 0]) * (f[:, 3] - f[:, 1]), 0)).mean())
        for f in per_frame
        if len(f) and (f[:, 4] >= conf).any()
    ]
    return hits.astype(bool), counts, sides


def temporal_post(hits: np.ndarray, max_gap: int, m: int, n: int) -> np.ndarray:
    """M-of-N confirmation followed by short-gap interpolation, mirroring what a tracker does online.

    The window is centred because this is an offline evaluation (the online tracker can only look backwards). A frame
    is confirmed when at least `m` of the `n` samples around it are hits, which removes isolated single-frame boxes;
    gaps of at most `max_gap` samples that are bracketed by confirmed hits are then filled, which recovers dropouts the
    detector itself never produced.
    """
    pad = n // 2
    padded = np.concatenate([np.zeros(pad, bool), hits, np.zeros(pad, bool)])
    window = np.array([padded[i : i + n].sum() for i in range(len(hits))])
    out = hits & (window >= m)
    for _ in range(max_gap):  # each pass fills gaps of length up to the current pass index + 1
        filled = out.copy()
        for i in range(1, len(out) - 1):
            if not out[i] and out[i - 1] and out[i + 1]:
                filled[i] = True
        if filled.sum() == out.sum():
            break
        out = filled
    return out


def summarise(hits: np.ndarray, counts: np.ndarray, sides: list[float], indices: list[int], conf: float) -> dict:
    """Compute boxes/frame, frame-hit rate, persistence and median box size for one series."""
    # persistence: hit frames whose previous *sampled* frame was consecutive in time and also hit
    step = indices[1] - indices[0] if len(indices) > 1 else 0
    consecutive = np.diff(indices) == step
    pairs = hits[1:] & consecutive & hits[:-1]
    previous_hit = hits[:-1] & consecutive
    persistence = float(pairs.sum() / max(previous_hit.sum(), 1))
    runs = int(np.sum(hits & ~np.concatenate([[False], hits[:-1]])))
    ends = np.flatnonzero(hits & ~np.concatenate([hits[1:], [False]]))
    starts = np.flatnonzero(hits & ~np.concatenate([[False], hits[:-1]]))
    lengths = ends - starts + 1 if len(ends) else np.zeros(0, int)
    return {
        "conf": conf,
        "frames": len(hits),
        "boxes_per_frame": round(float(counts.mean()), 3),
        "frame_hit": round(float(hits.mean()), 3),
        "persistence": round(persistence, 3),
        "runs": runs,
        "mean_run_len": round(float(hits.sum() / max(runs, 1)), 1),
        "blip_frac": round(float((lengths == 1).sum() / max(len(lengths), 1)), 3),
        "median_side_px": round(float(np.median(sides)), 1) if sides else 0.0,
    }


def main() -> None:
    """Run every checkpoint over the video and print the per-threshold table."""
    args = parse_args()
    confs = [float(c) for c in args.conf.split(",")]
    sizes = args.imgsz or [640] * len(args.weights)
    if len(sizes) != len(args.weights):
        raise SystemExit("--imgsz must be given once per --weights")
    frames, indices = read_frames(args.video, args.stride, args.max_frames)
    print(f"video={args.video} frames={len(frames)} stride={args.stride} size={frames[0].shape[1]}x{frames[0].shape[0]}")

    rows = []
    header = (
        f"{'model':<22}{'imgsz':>6}{'conf':>6}{'stage':>6}{'box/frame':>11}{'frame_hit':>11}"
        f"{'persist':>9}{'runs':>6}{'run_len':>9}{'blip':>7}{'side_px':>9}"
    )
    print(header)
    print("-" * len(header))
    for weights, imgsz in zip(args.weights, sizes):
        model = YOLO(str(weights))
        per_frame = predict_confs(model, frames, imgsz, min(confs), args)
        for conf in confs:
            hits, counts, sides = series(per_frame, conf)
            post_hits = temporal_post(hits, args.max_gap, args.temporal_m, args.temporal_n)
            post_counts = np.where(post_hits & ~hits, 1, np.where(post_hits, counts, 0))
            label = weights.parent.parent.name[:21]
            for stage, stage_hits, stage_counts in (("raw", hits, counts), ("post", post_hits, post_counts)):
                row = summarise(stage_hits, stage_counts, sides, indices, conf)
                print(
                    f"{label:<22}{imgsz:>6}{conf:>6.2f}{stage:>6}{row['boxes_per_frame']:>11.3f}"
                    f"{row['frame_hit']:>11.3f}{row['persistence']:>9.3f}{row['runs']:>6}{row['mean_run_len']:>9.1f}"
                    f"{row['blip_frac']:>7.3f}{row['median_side_px']:>9.1f}"
                )
                rows.append(
                    {
                        "model": label,
                        "weights": str(weights),
                        "imgsz": imgsz,
                        "video": Path(args.video).name,
                        "stage": stage,
                        "m": args.temporal_m,
                        "n": args.temporal_n,
                        "max_gap": args.max_gap,
                        **row,
                    }
                )

    if args.csv:
        new = not args.csv.is_file()
        if not new:
            existing = next(csv.reader(args.csv.open(encoding="utf-8")), [])
            if existing != list(rows[0]):
                args.csv = args.csv.with_name(f"{args.csv.stem}_v2{args.csv.suffix}")
                print(f"existing CSV has a different schema, writing to {args.csv}")
                new = not args.csv.is_file()
        with args.csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)
        print(f"appended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
