# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build the RKNN INT8 calibration set from the deployment video itself.

Calibration decides the per-tensor activation ranges that INT8 quantization clamps to, so it has to see the pixel
statistics the model meets at inference time. `make_calib_dataset.py` samples ARD-MAV training frames, which is the
wrong domain for a model deployed on Hard1: an ARD-based calibration re-introduces exactly the grass / treeline
appearance that produced the false positives.

Frames are sampled evenly over the whole clip (not the first N seconds) so all lighting and treeline/sky/ground
compositions are covered. Labels are written empty on purpose - calibration only runs forward passes, and empty files
keep `YOLODataset` from warning about missing labels.

Usage:
    python tools/ard_mav/make_calib_video.py --video Hard1.mp4 --num 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    """Sample frames from the video and write the calibration dataset + yaml."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("g:/0-rk3588/train/hard1_neg/calib"))
    parser.add_argument("--num", type=int, default=400, help="target image count; >=300 is the INT8 recommendation")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality; keep high, quantization sees these pixels")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(total // max(args.num, 1), 1)
    out = args.out.resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    written, idx = [], 0
    while len(written) < args.num:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            name = f"{idx:06d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            (out / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
            written.append(idx)
        idx += 1
    cap.release()
    if not written:
        raise SystemExit("No frames written")

    (out / "data.yaml").write_text(
        "\n".join(
            [
                "# INT8 calibration set sampled from the deployment video (see tools/ard_mav/make_calib_video.py)",
                f"path: {out.as_posix()}",
                "train: images",
                "val: images",
                "nc: 1",
                "names: ['drone']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "calib_manifest.json").write_text(
        json.dumps(
            {
                "video": args.video.as_posix(),
                "video_frames": total,
                "size": f"{width}x{height}",
                "stride": stride,
                "requested": args.num,
                "written": len(written),
                "first": written[0],
                "last": written[-1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(written)} frames (stride {stride}, {width}x{height}) -> {out.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
