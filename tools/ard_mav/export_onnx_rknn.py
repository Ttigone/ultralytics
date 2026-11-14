# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export a YOLO26 detector to the raw multi-output ONNX used by RKNN, optionally building the INT8 RKNN model.

The raw layout keeps the Detect head undecoded: for every scale it emits ``reg_<scale>`` and ``cls_<scale>`` tensors,
which the RK3576 runtime decodes after inference. Class count, scale count and strides come from the checkpoint, so
ARD-MAV (nc=1, 4 scales) and VisDrone (nc=10) use the same code path; only the profile values differ.

Usage:
    python tools/ard_mav/export_onnx_rknn.py --profile ard_mav_rk3576 --weights path/best.pt
    python tools/ard_mav/export_onnx_rknn.py --profile visdrone_rk3576 --weights path/best.pt --rknn
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx
import torch
import yaml

# Run against this checkout even when the script directory (sys.path[0]) sits outside it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

SCALE_NAMES = {4: "p2", 8: "p3", 16: "p4", 32: "p5", 64: "p6"}


def load_profile(path: Path, name: str) -> dict:
    """Return one named export profile from a YAML file."""
    profiles = yaml.safe_load(path.read_text(encoding="utf-8"))["profiles"]
    if name not in profiles:
        raise KeyError(f"Unknown profile '{name}', available: {sorted(profiles)}")
    return profiles[name]


class FlatOutputs(torch.nn.Module):
    """Wrap the detector so every per-scale tensor is flattened to (B, C, H*W), matching the legacy RKNN decoder."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(t.flatten(2) for t in self.model(x))


def export_raw_onnx(weights: Path, imgsz: int, opset: int, output: Path, layout: str = "flat") -> None:
    """Export the Detect head's undecoded per-scale regression/classification tensors to ONNX."""
    model = YOLO(str(weights)).model.float().eval()
    head = model.model[-1]
    missing = [n for n in ("cv2", "cv3", "nl", "nc", "reg_max") if not hasattr(head, n)]
    if missing:
        raise RuntimeError(f"Last layer is not a Detect head, missing: {missing}")

    head.export = True
    head.format = "rknn"
    head.dynamic = False
    head.shape = None
    print(f"head={type(head).__name__} export={head.export} format={head.format} layout={layout}")

    dummy = torch.zeros((1, 3, imgsz, imgsz))
    with torch.inference_mode():
        raw = model(dummy)
    raw = tuple(raw) if isinstance(raw, (tuple, list)) else (raw,)
    expected = head.nl * 2
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} raw outputs, got {len(raw)}; check the RKNN export branch")

    strides = [int(s) for s in head.stride]
    output_names = [f"{kind}_{SCALE_NAMES.get(s, f's{s}')}" for s in strides for kind in ("reg", "cls")]
    print(f"weights={weights} nc={head.nc} scales={head.nl} strides={strides} reg_max={head.reg_max}")
    for name, tensor in zip(output_names, raw):
        print(f"  {name}: {tuple(tensor.shape)}")

    export_model = FlatOutputs(model) if layout == "flat" else model
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy,
        str(output),
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    ops = Counter(node.op_type for node in exported.graph.node)
    shapes = [tuple(d.dim_value for d in o.type.tensor_type.shape.dim) for o in exported.graph.output]
    print(f"ONNX ok: {output} | outputs={shapes} | ops={dict(ops.most_common(8))}")


def build_rknn(weights: Path, profile: dict, calib_yaml: Path) -> None:
    """Delegate the INT8 build to the repository's native ``format=rknn`` exporter (rknn-toolkit2 required)."""
    try:
        import rknn  # noqa: F401
    except ImportError:
        raise SystemExit(
            "rknn-toolkit2>=2.3.2 is not installed; run `yolo export format=rknn` on Linux/WSL with "
            "`pip install rknn-toolkit2` from the Rockchip wheel repository."
        ) from None
    model = YOLO(str(weights))
    model.export(
        format="rknn",
        name=profile["target_platform"],
        quantize=8 if profile["dtype"] == "i8" else 16,
        data=str(calib_yaml),
        imgsz=profile["imgsz"],
        batch=1,
        simplify=True,
    )


def main() -> None:
    """Run the raw ONNX export and an optional RKNN INT8 build."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profiles-file", type=Path, default=Path(__file__).with_name("export_profiles.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--opset", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calib-data", type=Path, help="YOLO yaml whose val split is the tiny-target calibration list")
    parser.add_argument("--layout", choices=("flat", "2d"), help="flat=(B,C,H*W) legacy decoder, 2d=(B,C,H,W)")
    parser.add_argument("--rknn", action="store_true", help="also build the INT8 RKNN model (requires rknn-toolkit2)")
    args = parser.parse_args()

    profile = load_profile(args.profiles_file, args.profile)
    imgsz = args.imgsz or profile["imgsz"]
    opset = args.opset or profile["opset"]
    layout = args.layout or profile.get("output_layout", "flat")
    output = args.output or args.weights.with_name(f"{args.weights.stem}_raw_{imgsz}.onnx")
    calib = args.calib_data or Path(profile["calib_data"])

    export_raw_onnx(args.weights, imgsz, opset, output, layout)
    if args.rknn:
        build_rknn(args.weights, profile, calib)


if __name__ == "__main__":
    main()
