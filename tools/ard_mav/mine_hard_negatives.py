# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Mine hard-negative background patches from a real deployment video, and dump a frame list for GT annotation.

Two jobs in one pass over the video, because both need the same per-frame detections:

1. Hard negatives. Hard1.mp4 has a drone in essentially every frame (the DUT model fires 1.07 boxes/frame at conf 0.05),
   so a whole frame can never be used as a negative image - an empty label file would teach the classifier to suppress
   the real target. Instead the frame is tiled into patches and a patch is kept only when it does not touch any detection
   from EITHER model, dilated by `--margin`. Patches are ranked by Laplacian variance so the output is dominated by the
   grass / conifer-band texture that produced 97% of the ARD-MAV false positives on this video, not by flat sky.

2. Evaluation frames. `--eval-frames N` writes N evenly spaced full-resolution frames plus an empty label directory and a
   format note, so precision/recall can be measured before and after retraining instead of guessing from box counts.

One more constraint decides the crop geometry: scale. A standalone square patch is always resized to fill `imgsz`, so a
320 px patch lands in the network at 320 px (2x the deployment scale) while the real video is letterboxed 1920 -> 640
(1/3). Texture blobs would then be learned at the wrong size. Full-width strips (1920 x H) keep the exact deployment
letterbox gain, and they also preserve the real sky/treeline/ground composition instead of inventing a crop that never
occurs in the video.

The detector used for exclusion is the DUT-Anti-UAV checkpoint (it is the cleaner model on this video). Pass a second
checkpoint with `--weights2` (the ARD one) and the union of both exclusions is used: a drone missed by both models is
below the detectability of the current pipeline, so no patch containing it can teach anything harmful.

Usage:
    python tools/ard_mav/mine_hard_negatives.py --video Hard1.mp4 \
        --weights G:/0-rk3588/train/DUT-Anti-UAV/runs/yolo26n_p2_p5_dut_640/weights/best.pt \
        --weights2 G:/0-rk3588/train/ARD-MAV/runs/r2_base_960-2/weights/best.pt \
        --out G:/0-rk3588/train/hard1_neg --limit 2000 --eval-frames 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO
from ultralytics.utils import LOGGER

EVAL_NOTE = """Hard1.mp4 evaluation frames - annotation format
===============================================
Images : eval/images/XXXXXX.jpg  (frame index in the file name)
Labels : eval/labels/XXXXXX.txt  (one file per image; create it even if the frame has no drone)

Each line is one target, YOLO detection format, 5 whitespace-separated columns:
    class  cx  cy  w  h
    * class : 0 (single class, "drone")
    * cx, cy: box centre, normalised to [0, 1] by image width / height
    * w, h  : box size, normalised to [0, 1]
Example for a 20x24 px drone centred at (960, 540) in a 1920x1080 frame:
    0 0.500000 0.500000 0.010417 0.022222

An empty .txt marks a frame with no drone. If a frame has a drone but you are unsure of the exact box, annotate the
tightest box you can - the same label is used for both recall and precision, so rough boxes bias both metrics equally.
Report: precision@conf, recall@conf, false-positives per frame (see tools/ard_mav/eval_size_buckets.py).
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True, help="primary exclusion model (DUT checkpoint)")
    p.add_argument("--weights2", type=Path, default=None, help="secondary exclusion model (ARD checkpoint)")
    p.add_argument("--out", type=Path, default=Path("g:/0-rk3588/train/hard1_neg"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05, help="exclusion threshold; low on purpose")
    p.add_argument("--margin", type=int, default=40, help="pixels added around every detection before excluding patches")
    p.add_argument("--strip-h", type=int, default=320, help="strip height; the width is always the full frame width")
    p.add_argument("--stride", type=int, default=160, help="vertical stride between strips; 160 = 50%% overlap")
    p.add_argument("--step", type=int, default=5, help="sample every Nth frame")
    p.add_argument("--limit", type=int, default=2000, help="max patches to write")
    p.add_argument("--max-per-frame", type=int, default=6)
    p.add_argument("--easy-frac", type=float, default=0.15, help="share of output drawn from low-texture (sky/flat) patches")
    p.add_argument("--min-var", type=float, default=25.0, help="Laplacian variance floor for a hard-negative patch")
    p.add_argument("--dedup-mad", type=float, default=0.01, help="drop patches closer than this to an already-saved one")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-frames", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def detect_frames(models: list[YOLO], frames: list[np.ndarray], args: argparse.Namespace) -> list[np.ndarray]:
    """Return one (N,4) int array of exclusion boxes per frame, the union over all models."""
    per_frame = [np.zeros((0, 4), np.int32) for _ in frames]
    for model in models:
        results = model.predict(
            frames, imgsz=args.imgsz, conf=args.conf, max_det=300, device=args.device, verbose=False
        )
        for i, r in enumerate(results):
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes):
                per_frame[i] = np.concatenate([per_frame[i], np.round(boxes).astype(np.int32)])
    return per_frame


def blocked_mask(boxes: np.ndarray, height: int, width: int, margin: int) -> np.ndarray:
    """Rasterise the dilated exclusion boxes into a uint8 mask."""
    mask = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in boxes:
        m = margin + int(0.25 * max(x2 - x1, y2 - y1))
        cv2.rectangle(mask, (max(0, x1 - m), max(0, y1 - m)), (min(width - 1, x2 + m), min(height - 1, y2 + m)), 1, -1)
    return mask


def strip_origins(height: int, strip_h: int, stride: int) -> list[int]:
    """Top edge of every full-width strip, clamped so the last strip stays inside the frame."""
    ys = list(range(0, max(height - strip_h, 0) + 1, stride)) or [0]
    if ys[-1] != height - strip_h and height > strip_h:
        ys.append(height - strip_h)
    return ys


def thumb(patch: np.ndarray) -> np.ndarray:
    """Small grayscale descriptor used for near-duplicate rejection."""
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (16, 9), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def main() -> None:
    """Scan the video, select patches, then write the chosen negatives and the annotation frame list."""
    args = parse_args()
    out = args.out.resolve()
    (out / "images/bg").mkdir(parents=True, exist_ok=True)
    (out / "labels/bg").mkdir(parents=True, exist_ok=True)

    models = [YOLO(str(args.weights))]
    if args.weights2:
        models.append(YOLO(str(args.weights2)))
    LOGGER.info(f"exclusion models: {[str(m.ckpt_path) for m in models]}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    LOGGER.info(f"{args.video} frames={n_frames} size={width}x{height} step={args.step}")

    # pass 1: candidate geometry + texture score
    candidates: list[dict] = []
    batch_frames: list[np.ndarray] = []
    batch_idx: list[int] = []
    idx = 0

    def flush() -> None:
        if not batch_frames:
            return
        boxes_list = detect_frames(models, batch_frames, args)
        for frame, fidx, boxes in zip(batch_frames, batch_idx, boxes_list):
            h, w = frame.shape[:2]
            mask = blocked_mask(boxes, h, w, args.margin)
            for y in strip_origins(h, args.strip_h, args.stride):
                if mask[y : y + args.strip_h].any():
                    continue
                crop = frame[y : y + args.strip_h]
                var = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                candidates.append({"frame": fidx, "y": y, "var": var, "cy": (y + args.strip_h / 2) / h})
        batch_frames.clear()
        batch_idx.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            batch_frames.append(frame)
            batch_idx.append(idx)
            if len(batch_frames) >= args.batch:
                flush()
        idx += 1
    flush()
    cap.release()
    LOGGER.info(f"candidate patches after exclusion: {len(candidates)}")

    # selection: per-frame top by texture for the hard set, plus a random low-texture slice for easy negatives
    rng = np.random.default_rng(args.seed)
    by_frame: dict[int, list[dict]] = {}
    for c in candidates:
        by_frame.setdefault(c["frame"], []).append(c)
    hard: list[dict] = []
    for fidx in sorted(by_frame):
        tiles = sorted(by_frame[fidx], key=lambda c: -c["var"])
        hard += [c for c in tiles[: args.max_per_frame] if c["var"] >= args.min_var]
    hard.sort(key=lambda c: -c["var"])
    n_easy = int(args.limit * args.easy_frac)
    easy_pool = [c for c in candidates if c["var"] < args.min_var]
    easy = list(rng.choice(len(easy_pool), size=min(n_easy, len(easy_pool)), replace=False)) if easy_pool else []
    easy = [easy_pool[i] for i in easy]

    ordered = hard[: max(args.limit - len(easy), 0)] + easy

    # pass 2: write selected patches, rejecting near-duplicates
    wanted: dict[int, list[dict]] = {}
    for c in ordered:
        wanted.setdefault(c["frame"], []).append(c)
    cap = cv2.VideoCapture(str(args.video))
    saved_thumbs: list[np.ndarray] = []
    kept: list[dict] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            for c in wanted[idx]:
                crop = frame[c["y"] : c["y"] + args.strip_h]
                t = thumb(crop).ravel()
                if saved_thumbs:
                    mad = np.abs(np.asarray(saved_thumbs) - t).mean(1)
                    if mad.min() < args.dedup_mad:
                        continue
                saved_thumbs.append(t)
                name = f"hard1_f{c['frame']:06d}_y{c['y']}"
                cv2.imwrite(str(out / "images/bg" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (out / "labels/bg" / f"{name}.txt").write_text("", encoding="utf-8")
                kept.append({"name": name, "frame": c["frame"], "y": c["y"], "var": round(c["var"], 1)})
                if len(kept) >= args.limit:
                    break
        if len(kept) >= args.limit:
            break
        idx += 1
    cap.release()

    # evaluation frame list
    eval_meta = None
    if args.eval_frames:
        (out / "eval/images").mkdir(parents=True, exist_ok=True)
        (out / "eval/labels").mkdir(parents=True, exist_ok=True)
        (out / "eval/README.txt").write_text(EVAL_NOTE, encoding="utf-8")
        picks = np.linspace(0, max(n_frames - 1, 0), args.eval_frames).round().astype(int)
        cap = cv2.VideoCapture(str(args.video))
        idx, lines = 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in set(picks.tolist()):
                name = f"{idx:06d}"
                cv2.imwrite(str(out / "eval/images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / "eval/labels" / f"{name}.txt").write_text("", encoding="utf-8")
                lines.append(name)
            idx += 1
        cap.release()
        (out / "eval/eval_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        eval_meta = {"frames": len(lines), "dir": str(out / "eval"), "first": lines[0], "last": lines[-1]}

    manifest = {
        "video": str(args.video),
        "frames_total": n_frames,
        "size": [width, height],
        "step": args.step,
        "strip_h": args.strip_h,
        "strip_w": width,
        "letterbox_gain_at_imgsz": round(min(args.imgsz / width, args.imgsz / height), 4),
        "stride": args.stride,
        "margin": args.margin,
        "conf": args.conf,
        "candidates": len(candidates),
        "saved": len(kept),
        "saved_hard_texture": sum(1 for k in kept if k["var"] >= args.min_var),
        "saved_easy_flat": sum(1 for k in kept if k["var"] < args.min_var),
        "y_band": {
            "upper(<0.45)": round(sum(1 for k in kept if (k["y"] + args.strip_h / 2) / height < 0.45) / max(len(kept), 1), 3),
            "mid(0.45-0.75)": round(
                sum(1 for k in kept if 0.45 <= (k["y"] + args.strip_h / 2) / height < 0.75) / max(len(kept), 1), 3
            ),
            "lower(>=0.75)": round(
                sum(1 for k in kept if (k["y"] + args.strip_h / 2) / height >= 0.75) / max(len(kept), 1), 3
            ),
        },
        "eval": eval_meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
