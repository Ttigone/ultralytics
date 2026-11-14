# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render detection videos and a per-frame CSV for one or more checkpoints, streaming frame by frame.

`video_detection_stats.py` buffers every sampled frame in RAM and only reports aggregate numbers, which is fine for
stride-5 sampling but impossible for full-rate rendering of a 1080p clip. This script decodes once, runs every
checkpoint on the same batch, draws one video per (model, confidence) at the source resolution and frame rate, and
writes a per-frame row so a specific dropout can be located by frame number. The proxy metrics reuse that script's
definitions (`series` / `summarise` / `temporal_post`) instead of re-deriving them.

Usage:
    python tools/ard_mav/render_detect_video.py --video Hard1.mp4 \
        --weights g:/.../r4_dut640_synth/weights/best.pt --imgsz 640 \
        --weights g:/.../r3_dut640-2/weights/best.pt --imgsz 640 \
        --conf 0.15,0.30 --out-dir g:/0-rk3588/train/ARD-MAV/viz_r4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics import YOLO
from video_detection_stats import summarise, temporal_post

COLORS = ((120, 200, 60), (60, 160, 255), (200, 80, 220), (0, 220, 220))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, action="append", required=True)
    p.add_argument("--imgsz", action="append", type=int, default=None, help="one input size per --weights")
    p.add_argument("--conf", default="0.15,0.30")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def draw(frame: np.ndarray, boxes: np.ndarray, conf: float, label: str, color, line: int) -> np.ndarray:
    """Draw every kept box with its confidence and a corner banner."""
    out = frame.copy()
    for row in boxes:
        x1, y1, x2, y2, c = row[0], row[1], row[2], row[3], row[4]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, line)
        cv2.putText(
            out, f"{c:.2f}", (p1[0], max(p1[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    cv2.putText(
        out,
        f"{label}  conf>={conf:.2f}  n={len(boxes)}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    """Stream the clip once, writing rendered videos, a per-frame CSV and the proxy table."""
    args = parse_args()
    confs = [float(c) for c in args.conf.split(",")]
    sizes = args.imgsz or [640] * len(args.weights)
    if len(sizes) != len(args.weights):
        raise SystemExit("--imgsz must be given once per --weights")
    names = [w.parent.parent.name for w in args.weights]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = [YOLO(str(w)) for w in args.weights]
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: dict[tuple[int, float], cv2.VideoWriter] = {}

    rows = []
    hits = {i: {c: [] for c in confs} for i in range(len(models))}
    counts = {i: {c: [] for c in confs} for i in range(len(models))}
    sides = {i: {c: [] for c in confs} for i in range(len(models))}
    indices: list[int] = []
    frame_idx, done, shape = 0, 0, None
    chunk, chunk_idx = [], []

    def flush() -> None:
        """Run every model on the buffered chunk and write one frame per writer."""
        nonlocal chunk, chunk_idx, shape
        if not chunk:
            return
        for i, model in enumerate(models):
            results = model.predict(
                chunk, imgsz=sizes[i], conf=min(confs), max_det=300, device=args.device, verbose=False
            )
            for local, result in enumerate(results):
                data = result.boxes.data.cpu().numpy() if len(result.boxes) else np.zeros((0, 6), np.float32)
                for conf in confs:
                    kept = data[data[:, 4] >= conf] if len(data) else data
                    hits[i][conf].append(bool(len(kept)))
                    counts[i][conf].append(len(kept))
                    side = (
                        float(np.sqrt((kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])).mean()) if len(kept) else 0.0
                    )
                    if len(kept):
                        sides[i][conf].append(side)
                    rows.append(
                        {
                            "frame": chunk_idx[local],
                            "t_sec": round(chunk_idx[local] / fps, 3),
                            "model": names[i],
                            "conf": conf,
                            "boxes": len(kept),
                            "max_conf": round(float(kept[:, 4].max()), 4) if len(kept) else 0.0,
                            "sqrt_area_px": round(
                                float(np.sqrt(((kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])).max())), 2
                            )
                            if len(kept)
                            else 0.0,
                            "hit": int(len(kept) > 0),
                        }
                    )
                    w = writers[(i, conf)]
                    canvas = frame_of[local]
                    w.write(draw(canvas, kept, conf, names[i], COLORS[i % len(COLORS)], args.line_width))
        chunk, chunk_idx = [], []

    frame_of: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and done >= args.max_frames:
            break
        if shape is None:
            shape = (frame.shape[1], frame.shape[0])
            for i in range(len(models)):
                for conf in confs:
                    path = args.out_dir / f"{args.video.stem}_{names[i]}_c{int(round(conf * 100)):02d}.mp4"
                    writers[(i, conf)] = cv2.VideoWriter(str(path), fourcc, fps, shape)
                    print(f"writing {path}")
        frame_of.append(frame)
        chunk.append(frame)
        chunk_idx.append(frame_idx)
        indices.append(frame_idx)
        frame_idx += 1
        done += 1
        if len(chunk) >= args.batch:
            flush()
            frame_of = []
    flush()
    cap.release()
    for w in writers.values():
        w.release()

    print(f"\nvideo={args.video} frames={done} fps={fps:.2f} size={shape[0]}x{shape[1]}")
    header = f"{'model':<22}{'conf':>6}{'box/frame':>11}{'frame_hit':>11}{'persist':>9}{'runs':>6}{'blip':>7}{'side_px':>9}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        for conf in confs:
            h = np.array(hits[i][conf])
            row = summarise(h, np.array(counts[i][conf]), sides[i][conf], indices, conf)
            post = temporal_post(h, 2, 3, 5)
            print(
                f"{name:<22}{conf:>6.2f}{row['boxes_per_frame']:>11.3f}{row['frame_hit']:>11.3f}"
                f"{row['persistence']:>9.3f}{row['runs']:>6}{row['blip_frac']:>7.3f}{row['median_side_px']:>9.1f}"
                f"   post: hit={post.mean():.3f} runs={int(np.sum(post & ~np.concatenate([[False], post[:-1]])))}"
            )

    csv_path = args.out_dir / f"{args.video.stem}_per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-frame rows={len(rows)} -> {csv_path}")


if __name__ == "__main__":
    main()
