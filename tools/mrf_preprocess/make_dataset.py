"""Turn an Ultralytics YOLO dataset into a preprocessed copy.

The training scripts are NOT modified: the output directory is a drop-in
dataset with the same layout (images/{train,val}, labels/{train,val}) plus a
``data-mrf.yaml`` you can point ``yolo train data=...`` at.

Example
  python make_dataset.py --src G:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo \
      --dst G:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo-mrf-gate \
      --method mrf_gate --set tau=8,iters=6,scale=4

Mapping to uint8 (``--map``):
  direct : clip(out, 0, 255)      image-like outputs (gate / bgsub_ms)
  signed : clip(out + 128, 0, 255) residual outputs (mrf_res / dog / tophat / lcn)

Colour is preserved: the operator is applied per channel (or to luma only with
``--gray``, which makes the result 3 identical channels).
"""

import argparse
import os
import shutil
from glob import glob

import cv2
import numpy as np
import yaml

import filters as F


def parse_set(items):
    kw = {}
    for line in items or []:
        for item in line.split(","):
            k, v = item.split("=")
            kw[k] = float(v) if v.replace(".", "", 1).replace("-", "").isdigit() else v
    return kw


def to_uint8(out, mode, gray_floor=0.0):
    if mode == "auto":
        mode = "signed" if out.min() < -1.0 else "direct"
    if mode == "signed":
        out = out + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--method", default="mrf_gate")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--map", default="auto", choices=["auto", "direct", "signed"])
    ap.add_argument("--gray", action="store_true", help="luma only, replicated to 3 channels")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--limit", type=int, default=0, help="max images per split (0 = all)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = F.build_methods()
    if args.method not in methods:
        raise SystemExit("unknown method %s (have: %s)" % (args.method, ", ".join(methods)))
    fn, kw = methods[args.method]
    kw = {**kw, **parse_set(args.set)}
    print("method=%s params=%s map=%s gray=%s" % (args.method, kw, args.map, args.gray))

    with open(os.path.join(args.src, "data.yaml"), encoding="utf-8") as f:
        src_yaml = yaml.safe_load(f)

    stats = []
    for split in args.splits.split(","):
        img_in = os.path.join(args.src, "images", split)
        files = sorted(glob(os.path.join(img_in, "*")))
        if args.limit:
            files = files[: args.limit]
        out_img = os.path.join(args.dst, "images", split)
        out_lbl = os.path.join(args.dst, "labels", split)
        if not args.dry_run:
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_lbl, exist_ok=True)
        print("[%s] %d images" % (split, len(files)))
        for i, fp in enumerate(files):
            bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if bgr is None:
                print("  skip unreadable", fp)
                continue
            if args.gray:
                g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                o = to_uint8(fn(g, **kw), args.map)
                out = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
            else:
                chans = []
                for c in range(3):
                    chans.append(to_uint8(fn(bgr[:, :, c].astype(np.float32), **kw), args.map))
                out = cv2.merge(chans)
            stats.append(out.astype(np.float32))
            if not args.dry_run:
                cv2.imwrite(os.path.join(out_img, os.path.basename(fp)), out,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                lb = os.path.join(args.src, "labels", split,
                                  os.path.splitext(os.path.basename(fp))[0] + ".txt")
                if os.path.exists(lb):
                    shutil.copy2(lb, out_lbl)
            if (i + 1) % 200 == 0:
                print("  %d/%d" % (i + 1, len(files)), flush=True)

    if stats:
        s = np.concatenate([x.ravel() for x in stats[:: max(1, len(stats) // 200)]])
        print("uint8 output: min=%.0f p50=%.0f p99=%.0f max=%.0f  frac@0=%.4f frac@255=%.4f std=%.2f" % (
            s.min(), np.median(s), np.percentile(s, 99), s.max(),
            float(np.mean(s == 0)), float(np.mean(s == 255)), float(s.std())))

    if not args.dry_run:
        dst_yaml = dict(path=args.dst.replace("\\", "/"),
                        train="images/train", val="images/val",
                        nc=src_yaml.get("nc", 1), names=src_yaml.get("names", ["Drone"]),
                        preprocess=dict(method=args.method, params=kw,
                                        map=args.map, gray=bool(args.gray)))
        with open(os.path.join(args.dst, "data-mrf.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dst_yaml, f, allow_unicode=True, sort_keys=False)
        print("saved", os.path.join(args.dst, "data-mrf.yaml"))


if __name__ == "__main__":
    main()
