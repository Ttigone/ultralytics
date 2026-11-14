# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reproducible ARD-MAV training entry point for the baseline, ablation and round-2 sweeps.

Every augmentation value that matters for 4-16 px targets is an explicit flag, so an A/B only differs in the flag under
test. Defaults are the round-2 recipe: sequence-level split, `patience=20`, half-strength mosaic that closes 20 epochs
early, no mixup, no multi-scale and a conservative scale jitter.

Usage:
    python tools/ard_mav/train.py --variant baseline --epochs 100 --imgsz 640 --batch 16
    python tools/ard_mav/train.py --variant baseline --imgsz 960 --batch 8        # resolution sweep
    python tools/ard_mav/train.py --variant baseline --weights weights/yolo26n.pt  # pretrained A/B (True)
    python tools/ard_mav/train.py --variant dwr_nwd --weights weights/yolo26n.pt --mixup 0.1  # augmentation control
    python tools/ard_mav/train.py --resume g:/0-rk3588/train/ARD-MAV/runs/baseline_p2_r2/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

VARIANTS = {
    # name: (model yaml, nwd, sr_loss)
    "baseline": ("yolo26n-p2.yaml", False, 0.0),
    "dwr": ("yolo26n-p2-dwr.yaml", False, 0.0),
    "nwd": ("yolo26n-p2.yaml", True, 0.0),
    "dwr_nwd": ("yolo26n-p2-dwr.yaml", True, 0.0),
    "dwr_nwd_sr": ("yolo26n-p2-dwr-sr.yaml", True, 0.05),
}


def main() -> None:
    """Train one variant with the shared ARD-MAV recipe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    parser.add_argument(
        "--data",
        default="g:/0-rk3588/train/ARD-MAV/ARD-MAV-yolo/splits/data.yaml",
        help="sequence-level split data.yaml; the legacy data.yaml (images/train + images/val) also works",
    )
    parser.add_argument("--out", default="g:/0-rk3588/train/ARD-MAV/runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="640 | 960 | 1280 on a 10GB RTX 3080")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20, help="early-stop after N epochs without val improvement")
    parser.add_argument("--nwd-threshold", type=float, default=16.0)
    parser.add_argument("--nwd-c", type=float, default=12.0)
    parser.add_argument("--nwd-weight", type=float, default=1.0)
    parser.add_argument(
        "--sr-loss",
        type=float,
        default=0.0,
        help="training-only super-resolution auxiliary loss gain; needs a DetectSR head (yolo26n-p2-sr.yaml)",
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="train on a subset for smoke tests")
    parser.add_argument("--quantize", type=int, default=None, help="8 enables modelopt QAT preparation")
    parser.add_argument(
        "--calib-data", default=None, help="image-list .txt for QAT calibration instead of the train split"
    )
    parser.add_argument("--model", default="", help="model source override: .yaml architecture or .pt checkpoint")
    parser.add_argument("--weights", default="", help="pretrained .pt loaded into the selected .yaml architecture")
    parser.add_argument(
        "--pretrained",
        choices=("auto", "true", "false"),
        default="auto",
        help="auto: True when --weights/.pt is used; true: requires --weights; false: force from-scratch",
    )
    parser.add_argument("--mosaic", type=float, default=0.5, help="0.0 disables; 0.5 halves the default 1.0")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp probability; 0.0 keeps tiny targets intact")
    parser.add_argument("--close-mosaic", type=int, default=20, help="disable mosaic for the final N epochs")
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="scale jitter; 0.5 is the value the round-3 recipe uses, 0.3 keeps an 8px target from shrinking to 5px",
    )
    parser.add_argument("--multi-scale", type=float, default=0.0, help="0.0 keeps every epoch at --imgsz")
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--erasing", type=float, default=0.0, help="random erasing can delete a tiny target entirely")
    parser.add_argument(
        "--cos-lr",
        choices=("true", "false"),
        default="false",
        help="cosine LR schedule; true keeps early stopping honest by decaying lr before patience expires",
    )
    parser.add_argument("--cls", type=float, default=0.5, help="classification loss gain; >0.5 suppresses high-score FP")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="saturation jitter; the default 0.7 is large for grey sky")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="value jitter")
    parser.add_argument("--cache", choices=("none", "ram", "disk"), default="none", help="image caching mode")
    parser.add_argument("--amp", choices=("true", "false", "bf16"), default="true")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--lr0", type=float, default=None, help="override the trainer's optimizer default")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--plots", action="store_true", help="save results.png, confusion matrix and training batches")
    parser.add_argument("--save-period", type=int, default=-1, help="save a checkpoint every N epochs (-1 disables)")
    parser.add_argument("--name", default="")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", type=Path, help="resume from a run's last.pt (keeps its args and epoch counter)")
    parser.add_argument("--no-val", action="store_true", help="skip validation to shorten CPU smoke tests")
    args = parser.parse_args()

    if args.resume:
        YOLO(str(args.resume)).train(resume=True)
        return

    if args.pretrained == "true" and not args.weights:
        parser.error("--pretrained true requires --weights <checkpoint.pt>")

    yaml_name, nwd, sr_loss = VARIANTS[args.variant]
    model_path = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/26" / yaml_name
    name = args.name or f"{args.variant}_p2_r2_e{args.epochs}_{args.imgsz}"
    model_src = args.model or str(model_path)
    if args.weights and model_src.lower().endswith(".pt"):
        parser.error("--weights loads into a .yaml model; pass a .pt checkpoint to --model instead")
    model = YOLO(model_src)
    if args.weights:
        model.load(args.weights)
    has_weights = bool(args.weights) or model_src.lower().endswith(".pt")
    pretrained = has_weights if args.pretrained == "auto" else args.pretrained == "true"

    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=pretrained,
        patience=args.patience,
        seed=0,
        project=args.out,
        name=name,
        exist_ok=args.exist_ok,
        plots=args.plots,
        save_period=args.save_period,
        val=not args.no_val,
        fraction=args.fraction,
        quantize=args.quantize,
        calib_data=args.calib_data,
        nwd=nwd,
        nwd_threshold=args.nwd_threshold,
        nwd_c=args.nwd_c,
        nwd_weight=args.nwd_weight,
        sr_loss=args.sr_loss or sr_loss,
        # augmentation: explicit so an A/B only changes the flag under test
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        multi_scale=args.multi_scale,
        perspective=args.perspective,
        fliplr=args.fliplr,
        erasing=args.erasing,
        cos_lr=args.cos_lr == "true",
        cls=args.cls,
        cls_pw=0.0,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        max_det=args.max_det,
        amp={"true": True, "bf16": "bf16", "false": False}[args.amp],
        cache=args.cache if args.cache != "none" else False,
    )
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0
    if args.weight_decay is not None:
        train_args["weight_decay"] = args.weight_decay
    model.train(**train_args)


if __name__ == "__main__":
    main()
