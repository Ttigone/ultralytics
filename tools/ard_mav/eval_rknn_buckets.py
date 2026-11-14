# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate an exported RKNN detector per target-size bucket, on the x86 simulator or a Rockchip board.

``eval_size_buckets.py`` only understands Ultralytics checkpoints, so a quantized build had no way to answer
the one question that decides whether INT8 ships: how much recall the small-target buckets lose. This script
reuses the exact preprocessing, raw-output decode, NMS and GT matching of the PyTorch path, so the only
variable left is the RKNN build itself.

The RKNN graph is exported with mean=0 / std=255 baked in, therefore images are fed as uint8 RGB NHWC and
must not be normalized a second time.

Usage (WSL, rknn-toolkit2 installed):
    ~/venv_rknn/bin/python tools/ard_mav/eval_rknn_buckets.py \
        --rknn /mnt/g/.../best_sr_960_8outputs_rk3576_i8.rknn \
        --data .../data_leakfree.yaml --split test --limit 400
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_pt_onnx_video import Preproc, detections  # noqa: E402
from eval_size_buckets import BUCKET_NAMES, CONF_SWEEP, bucket_of, match, read_gt, read_split  # noqa: E402
from ultralytics.utils.tal import dist2bbox, make_anchors  # noqa: E402


def raw_maps(outputs: list[np.ndarray], imgsz: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split the flat RKNN outputs into per-scale regression and classification maps.

    The exporter emits ``reg_p2, cls_p2, reg_p3, cls_p3, ...`` as ``(1, C, H*W)``. Some toolkit versions hand
    the tensors back reshaped to ``(1, C, H, W)``, so both layouts are accepted.
    """
    tensors = []
    for out in outputs:
        t = torch.from_numpy(np.asarray(out)).float()
        if t.ndim == 4:  # (1, C, H, W) -> (1, C, H*W)
            t = t.flatten(2)
        if t.ndim == 2:  # (C, H*W) -> (1, C, H*W)
            t = t.unsqueeze(0)
        tensors.append(t)
    if len(tensors) % 2:
        raise ValueError(f"expected an even number of raw outputs, got {len(tensors)}")
    n = len(tensors) // 2
    return tensors[0 : 2 * n : 2], tensors[1 : 2 * n : 2]


def decode(reg: list[torch.Tensor], cls: list[torch.Tensor], imgsz: int) -> torch.Tensor:
    """Turn flat per-scale maps into the (1, 4+nc, anchors) xywh tensor the repo NMS expects."""
    feats = []
    for r in reg:
        hw = int(round(r.shape[-1] ** 0.5))
        if hw * hw != r.shape[-1]:
            raise ValueError(f"non-square reg map: {tuple(r.shape)}")
        feats.append(r.view(r.shape[0], r.shape[1], hw, hw))
    strides = [imgsz / f.shape[2] for f in feats]
    anchors, stride_tensor = make_anchors(feats, strides)
    boxes = torch.cat([f.flatten(2) for f in feats], 2).transpose(1, 2)
    boxes = dist2bbox(boxes, anchors, xywh=True) * stride_tensor
    scores = torch.cat([c.float().flatten(2) for c in cls], 2).transpose(1, 2).sigmoid()
    return torch.cat((boxes, scores), 2).transpose(1, 2)


def main() -> None:
    """Run RKNN inference once per image and report the bucket table."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rknn", type=Path, help="a built .rknn; only loadable on Rockchip hardware")
    group.add_argument("--onnx", type=Path, help="raw 8-output ONNX; built in-process so the x86 simulator works")
    parser.add_argument("--calib", type=Path, help="INT8 calibration image list (required with --onnx --dtype i8)")
    parser.add_argument("--target", default="rk3576")
    parser.add_argument("--dtype", choices=("i8", "fp"), default="i8")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a TP")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--workdir", type=Path, help="writable cwd for rknn-toolkit2 intermediate files")
    args = parser.parse_args()

    # rknn-toolkit2 writes check*.onnx next to the process cwd; a scheduler/system cwd is not writable.
    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=True)
        os.chdir(args.workdir)

    from rknn.api import RKNN

    images = read_split(args.data, args.split)
    if args.limit:
        images = images[: args.limit]
    experiment = args.experiment or (args.rknn or args.onnx).stem

    rknn = RKNN(verbose=False)
    if args.rknn:
        # A .rknn loaded back from disk cannot run on the x86 simulator; this path needs a board.
        if rknn.load_rknn(str(args.rknn)) != 0:
            raise RuntimeError(f"load_rknn failed: {args.rknn}")
        ret = rknn.init_runtime()
    else:
        from eval_size_buckets import local_path

        if rknn.config(target_platform=args.target, mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]]) != 0:
            raise RuntimeError("rknn.config failed")
        if rknn.load_onnx(model=str(args.onnx)) != 0:
            raise RuntimeError(f"load_onnx failed: {args.onnx}")
        build = {"do_quantization": args.dtype == "i8"}
        if args.dtype == "i8":
            if not args.calib:
                raise SystemExit("--calib is required for an INT8 build")
            lines = [local_path(x) for x in args.calib.read_text(encoding="utf-8").splitlines() if x.strip()]
            calib = args.calib.with_name(f"{args.calib.stem}_wsl.txt")
            calib.write_text("\n".join(lines) + "\n", encoding="utf-8")
            build["dataset"] = str(calib)
            print(f"calibration images: {len(lines)}", flush=True)
        if rknn.build(**build) != 0:
            raise RuntimeError("rknn.build failed")
        ret = rknn.init_runtime(target=None)  # simulator
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    preproc = Preproc(args.imgsz)
    stats = {t: defaultdict(lambda: {"gt": 0, "tp": 0, "pred": 0, "fp": 0}) for t in CONF_SWEEP}
    started = time.time()

    for index, image in enumerate(images, 1):
        frame = cv2_imread(image)
        if frame is None:
            continue
        x, gain, pad = preproc(frame)
        nhwc = (x[0].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)[None, ...]
        outputs = rknn.inference(inputs=[nhwc])
        reg, cls = raw_maps(outputs, args.imgsz)
        pred = detections(decode(reg, cls, args.imgsz), 0.001, args.nms_iou, args.max_det, nc=1)
        pred = pred[pred[:, 4] >= 0] if len(pred) else pred

        h, w = frame.shape[:2]
        pred[:, :4] = preproc.restore(pred[:, :4], gain, pad, (h, w))
        gt, gt_buckets = read_gt(image, w, h)
        pred_buckets = [bucket_of(float(max(b[2] - b[0], b[3] - b[1]))) for b in pred]
        matched = match(pred, gt, args.iou)

        for t in CONF_SWEEP:
            kept = [i for i in range(len(pred)) if pred[i, 4] >= t]
            for name in BUCKET_NAMES:
                stats[t][name]["gt"] += sum(b == name or name == "all" for b in gt_buckets)
            for i in kept:
                label = pred_buckets[i]
                if matched[i] >= 0:
                    hit = gt_buckets[matched[i]]
                    stats[t][hit]["tp"] += 1
                    stats[t][hit]["pred"] += 1
                    stats[t]["all"]["tp"] += 1
                    stats[t]["all"]["pred"] += 1
                else:
                    stats[t][label]["fp"] += 1
                    stats[t][label]["pred"] += 1
                    stats[t]["all"]["fp"] += 1
                    stats[t]["all"]["pred"] += 1

        if index % 25 == 0:
            rate = index / max(time.time() - started, 1e-9)
            print(f"[{index}/{len(images)}] {rate:.2f} img/s", flush=True)

    rows = []
    for t in CONF_SWEEP:
        for name in BUCKET_NAMES:
            s = stats[t][name]
            rows.append(
                {
                    "experiment": experiment,
                    "rknn": (args.rknn or args.onnx).as_posix(),
                    "imgsz": args.imgsz,
                    "split": args.split,
                    "images": len(images),
                    "conf": t,
                    "iou": args.iou,
                    "bucket": name,
                    "gt": s["gt"],
                    "tp": s["tp"],
                    "pred": s["pred"],
                    "fp": s["fp"],
                    "recall": round(s["tp"] / max(s["gt"], 1), 4),
                    "precision": round(s["tp"] / max(s["pred"], 1), 4),
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

    print(f"\nexperiment={experiment} images={len(images)} imgsz={args.imgsz}")
    print("  conf |  recall  precision  fp/frame | <4px R 4-8px R 8-16px R 16-32px R >32px R")
    for t in CONF_SWEEP:
        r = {row["bucket"]: row for row in rows if row["conf"] == t}
        per = " ".join(f"{r[b]['recall']:.3f}" for b in BUCKET_NAMES[:-1])
        print(
            f"  {t:4.2f} |  {r['all']['recall']:.4f}   {r['all']['precision']:.4f}    "
            f"{r['all']['fp_per_frame']:7.3f} | {per}"
        )
    rknn.release()


def cv2_imread(path: Path):
    """Read a BGR uint8 image, or None when unreadable."""
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_COLOR)


if __name__ == "__main__":
    main()
