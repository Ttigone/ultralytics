# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare a YOLO `.pt` checkpoint with a raw 8-output ONNX export on the same video.

Both paths share exactly one preprocessing (letterbox), one box decode (`dist2bbox`) and one NMS
(`ultralytics.utils.nms.non_max_suppression`), so any difference comes from the numeric backend
(PyTorch FP32 vs onnxruntime CPU FP32), not from a different pipeline.

The ONNX here is the RKNN-style raw export used by `Detect.forward(..., format="rknn")`: eight flat
outputs `reg_p2, cls_p2, reg_p3, cls_p3, ...` (raw ltrb distances and raw class logits), so
`YOLO("x.onnx")` cannot be used directly -- `DetectionPredictor.postprocess` would treat the first
raw map as a dense `[1, 4+nc, anchors]` tensor. The host decode below mirrors
`ultralytics/nn/backends/hailo.py::_decode_raw`.

NOTE: the input video has NO ground-truth labels. This script can only compare detection counts,
confidence distributions, frame-to-frame stability and pt/onnx agreement. Recall and precision are
NOT computed, and neither path can be called "more accurate" from these numbers.

Usage:
    python tools/ard_mav/compare_pt_onnx_video.py \
        --video runs/pt_onnx_compare/inputs/Hard1.mp4 \
        --pt  G:/0-rk3588/train/ARD-MAV/runs/r2_base_960-2/weights/best.pt \
        --onnx runs/pt_onnx_compare/inputs/best_8outputs.onnx \
        --out runs/pt_onnx_compare/Hard1_960 --conf 0.05,0.15,0.30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import nms
from ultralytics.utils.tal import dist2bbox, make_anchors

NO_GT_NOTE = (
    "NO GROUND TRUTH in this video: only detection counts, confidence distributions, temporal stability "
    "and pt/onnx agreement are reported. Recall / precision / mAP are NOT computable here."
)


# --------------------------------------------------------------------------- #
# Preprocessing (identical for both paths)
# --------------------------------------------------------------------------- #
class Preproc:
    """Letterbox to a square model input, then BGR->RGB uint8 CHW float [0, 1]."""

    def __init__(self, imgsz: int):
        self.imgsz = int(imgsz)
        self.lb = LetterBox((self.imgsz, self.imgsz), auto=False, scale_fill=False, scaleup=True, stride=32)

    def __call__(self, frame: np.ndarray) -> tuple[torch.Tensor, float, tuple[int, int]]:
        im = self.lb(image=frame)
        gain = min(self.imgsz / frame.shape[0], self.imgsz / frame.shape[1])
        pad = ((self.imgsz - round(frame.shape[1] * gain)) / 2.0, (self.imgsz - round(frame.shape[0] * gain)) / 2.0)
        x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))
        return torch.from_numpy(x).float().div(255.0).unsqueeze(0), gain, pad

    def restore(self, boxes: torch.Tensor, gain: float, pad: tuple[int, int], shape: tuple[int, int]) -> torch.Tensor:
        """Map letterboxed xyxy boxes back to original-frame pixels."""
        if not len(boxes):
            return boxes
        b = boxes.clone()
        b[:, [0, 2]] -= pad[0]
        b[:, [1, 3]] -= pad[1]
        b /= gain
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, shape[1])
        b[:, [1, 3]] = b[:, [1, 3]].clamp(0, shape[0])
        return b


# --------------------------------------------------------------------------- #
# Model wrappers, both returning the same dense (B, 4+nc, A) xywh tensor
# --------------------------------------------------------------------------- #
class PTModel:
    """PyTorch checkpoint run in export mode so the head returns the dense pre-NMS tensor."""

    def __init__(self, weights: str, imgsz: int, device: str):
        from ultralytics import YOLO

        self.yolo = YOLO(weights, task="detect")
        self.net = self.yolo.model.eval().to(device)
        self.head = self.net.model[-1]
        if not isinstance(self.head, Detect):
            raise TypeError(f"expected a Detect head, got {type(self.head).__name__}")
        self.head.export = True  # return the dense (B, 4+nc, A) tensor instead of (dense, raw dict)
        self.imgsz = imgsz
        self.device = device
        self.nc = int(self.head.nc)
        self.stride = [float(s) for s in self.net.stride]

    def dense(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(x.to(self.device))


class OnnxRaw:
    """Raw 8-output ONNX export; decodes on the host exactly like hailo.py::_decode_raw."""

    def __init__(self, path: str, imgsz: int | None):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        static = [d if isinstance(d, int) else 0 for d in inp.shape]
        if not imgsz:
            if not static[2] or not static[3]:
                raise ValueError("ONNX has a dynamic input shape; pass --imgsz-onnx explicitly")
            imgsz = int(static[2])
        self.imgsz = int(imgsz)
        self.out_names = [o.name for o in self.sess.get_outputs()]
        if len(self.out_names) % 2:
            raise ValueError(f"expected an even number of raw outputs, got {len(self.out_names)}")
        self._anchors = None

    def dense(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sess.run(self.out_names, {self.input_name: x.numpy()})
        n = len(y) // 2
        reg = [torch.from_numpy(a) for a in y[0 : 2 * n : 2]]  # reg_p2, reg_p3, ...
        cls = [torch.from_numpy(a) for a in y[1 : 2 * n : 2]]  # cls_p2, cls_p3, ...
        feats = []
        for r in reg:
            hw = int(round(r.shape[-1] ** 0.5))
            if hw * hw != r.shape[-1]:
                raise ValueError(f"non-square reg map: {tuple(r.shape)}")
            feats.append(r.view(r.shape[0], r.shape[1], hw, hw).float())
        strides = [self.imgsz / f.shape[2] for f in feats]
        if self._anchors is None:
            self._anchors = make_anchors(feats, strides)
        anchors, stride_tensor = self._anchors
        boxes = torch.cat([f.flatten(2) for f in feats], 2).transpose(1, 2)
        boxes = dist2bbox(boxes, anchors, xywh=True) * stride_tensor
        scores = torch.cat([c.float().flatten(2) for c in cls], 2).transpose(1, 2).sigmoid()
        return torch.cat((boxes, scores), 2).transpose(1, 2)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def detections(dense: torch.Tensor, conf: float, iou: float, max_det: int, nc: int) -> torch.Tensor:
    """Run the repo NMS and return an (n, 6) [x1, y1, x2, y2, conf, cls] tensor."""
    out = nms.non_max_suppression(
        dense, conf, iou, classes=None, agnostic=False, max_det=max_det, nc=nc, end2end=False
    )
    return out[0]


def match(a: torch.Tensor, b: torch.Tensor, thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one IoU matching. Returns (pairs, a_only, b_only)."""
    if not len(a) or not len(b):
        return 0, len(a), len(b)
    from ultralytics.utils.metrics import box_iou

    iou = box_iou(a[:, :4], b[:, :4])
    used_b = np.zeros(len(b), dtype=bool)
    pairs = 0
    for i in torch.argsort(-a[:, 4]):
        if used_b.all():
            break
        cand = iou[int(i)].masked_fill(torch.from_numpy(used_b), 0.0)
        j = int(cand.argmax())
        if float(cand[j]) > thr:
            used_b[j] = True
            pairs += 1
    return pairs, len(a) - pairs, len(b) - pairs


def draw(frame: np.ndarray, det: torch.Tensor, color: tuple[int, int, int], line: int, show_conf: bool) -> np.ndarray:
    """Draw detections with a readable box even for 4-8 px targets (min 12 px side for the outline)."""
    for *xyxy, conf, _cls in det.tolist():
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line, lineType=cv2.LINE_AA)
        if show_conf:
            label = f"{conf:.2f}"
            cv2.putText(frame, label, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame


def banner(frame: np.ndarray, text: str, h: int = 26) -> np.ndarray:
    """Translucent top banner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def stats(dets: torch.Tensor, frames: int) -> dict:
    """Aggregate per-path detections."""
    n = [len(d) for d in dets]
    confs = [d[:, 4] for d in dets if len(d)]
    allc = torch.cat(confs) if confs else torch.zeros(0)
    return {
        "frames": frames,
        "dets": int(sum(n)),
        "mean_per_frame": float(np.mean(n)) if n else 0.0,
        "frames_with_det": int(sum(i > 0 for i in n)),
        "det_frame_ratio": float(sum(i > 0 for i in n) / frames) if frames else 0.0,
        "mean_conf": float(allc.mean()) if allc.numel() else 0.0,
        "max_conf": float(allc.max()) if allc.numel() else 0.0,
        "max_per_frame": int(max(n)) if n else 0,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="pt vs raw-8-output onnx comparison on one video (no GT)")
    p.add_argument("--video", required=True)
    p.add_argument("--pt", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True, help="output directory (created, never overwritten silently)")
    p.add_argument("--conf", default="0.05", help="comma separated working points, e.g. 0.05,0.15,0.30")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz-pt", type=int, default=0, help="0 = read the training imgsz from the checkpoint")
    p.add_argument("--imgsz-onnx", type=int, default=0, help="0 = read the static input size from the ONNX graph")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--nms-conf", type=float, default=0.001, help="low internal conf, then filtered per --conf")
    p.add_argument("--max-frames", type=int, default=0, help="0 = full video")
    p.add_argument("--device", default="cpu", help="torch device for pt (onnx always runs on CPU here)")
    p.add_argument("--compare-scale", type=float, default=0.5, help="per-side scale of the side-by-side video")
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--verify-frames", type=int, default=8, help="frames used to check pt/onnx numeric agreement")
    p.add_argument("--no-videos", action="store_true")
    args = p.parse_args()

    confs = [float(c) for c in args.conf.split(",") if c.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {NO_GT_NOTE}\n")

    pt = PTModel(args.pt, 640, args.device)
    if not args.imgsz_pt:
        ck = getattr(pt.yolo, "ckpt", None) or {}
        pt.imgsz = int((ck.get("train_args") or {}).get("imgsz", 640) or 640)
    onnx = OnnxRaw(args.onnx, args.imgsz_onnx or None)
    if pt.nc != onnx.dense(torch.zeros(1, 3, onnx.imgsz, onnx.imgsz)).shape[1] - 4:
        print("[warn] pt nc != onnx nc, check that the ONNX matches this checkpoint")

    print(f"[pt]   {args.pt}")
    print(f"       imgsz={pt.imgsz} nc={pt.nc} stride={pt.stride} params={sum(x.numel() for x in pt.net.parameters())}")
    print(f"[onnx] {args.onnx}")
    print(f"       imgsz={onnx.imgsz} outputs={onnx.out_names}")
    if pt.imgsz != onnx.imgsz:
        print(f"[warn] input sizes differ (pt {pt.imgsz} vs onnx {onnx.imgsz}); each path runs at its OWN size")
    else:
        print(f"[ok]   both paths run at {pt.imgsz}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video} {w}x{h} {fps_v:.3f} fps {total} frames")

    pre_pt, pre_on = Preproc(pt.imgsz), Preproc(onnx.imgsz)

    # numeric agreement on a few frames (pt dense vs onnx dense, before NMS)
    if args.verify_frames > 0:
        diffs = []
        for _ in range(args.verify_frames):
            ok, f = cap.read()
            if not ok:
                break
            if pt.imgsz == onnx.imgsz:
                x, _, _ = pre_pt(f)
                a, b = pt.dense(x), onnx.dense(x)
            else:
                a = pt.dense(pre_pt(f)[0])
                b = onnx.dense(pre_on(f)[0])
            if a.shape == b.shape:
                diffs.append(float((a - b).abs().max()))
            else:
                diffs.append(float("nan"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        d = max(diffs) if diffs else float("nan")
        print(f"[verify] dense max|pt-onnx| over {len(diffs)} frames = {d:.6f} (0 means identical decode/weights)")
        if d > 1e-2:
            print("[warn] large pt/onnx deviation: the ONNX may come from a different checkpoint or export")

    writers = {}
    if not args.no_videos:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cw, ch = int(w * args.compare_scale), int(h * args.compare_scale)
        writers = {
            "pt": cv2.VideoWriter(str(out / f"{Path(args.video).stem}_pt.mp4"), fourcc, fps_v, (w, h)),
            "onnx": cv2.VideoWriter(str(out / f"{Path(args.video).stem}_onnx.mp4"), fourcc, fps_v, (w, h)),
            "cmp": cv2.VideoWriter(str(out / f"{Path(args.video).stem}_compare.mp4"), fourcc, fps_v, (cw * 2, ch)),
        }

    primary = confs[0]
    csv_path = out / f"frames_conf{primary:.2f}.csv"
    f_csv = csv_path.open("w", newline="", encoding="utf-8")
    wr = csv.writer(f_csv)
    wr.writerow(
        ["frame", "pt_n", "onnx_n", "pt_maxconf", "onnx_maxconf", "pt_meanconf", "onnx_meanconf",
         "matched_pairs", "pt_only", "onnx_only"]
    )

    all_pt, all_on = {c: [] for c in confs}, {c: [] for c in confs}
    pairs = {c: 0 for c in confs}
    only_p, only_o = {c: 0 for c in confs}, {c: 0 for c in confs}
    t_pt = t_on = 0.0
    n_frames = 0
    skipped = 0
    t0 = time.perf_counter()

    while True:
        if args.max_frames and n_frames >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1

        # ---- pt ----
        s = time.perf_counter()
        x, g, pad = pre_pt(frame)
        d_pt = detections(pt.dense(x), args.nms_conf, args.iou, args.max_det, pt.nc)
        d_pt[:, :4] = pre_pt.restore(d_pt[:, :4], g, pad, frame.shape[:2])
        t_pt += time.perf_counter() - s

        # ---- onnx ----
        s = time.perf_counter()
        x, g, pad = pre_on(frame)
        d_on = detections(onnx.dense(x), args.nms_conf, args.iou, args.max_det, pt.nc)
        d_on[:, :4] = pre_on.restore(d_on[:, :4], g, pad, frame.shape[:2])
        t_on += time.perf_counter() - s

        for c in confs:
            a, b = d_pt[d_pt[:, 4] >= c], d_on[d_on[:, 4] >= c]
            all_pt[c].append(a)
            all_on[c].append(b)
            mp, po, oo = match(a, b)
            pairs[c] += mp
            only_p[c] += po
            only_o[c] += oo
            if c == primary:
                wr.writerow([
                    n_frames, len(a), len(b),
                    f"{float(a[:, 4].max()) if len(a) else 0:.4f}",
                    f"{float(b[:, 4].max()) if len(b) else 0:.4f}",
                    f"{float(a[:, 4].mean()) if len(a) else 0:.4f}",
                    f"{float(b[:, 4].mean()) if len(b) else 0:.4f}",
                    mp, po, oo,
                ])

        if writers:
            a_vis, b_vis = all_pt[primary][-1], all_on[primary][-1]
            left = draw(frame.copy(), a_vis, (0, 220, 0), args.line_width, True)
            right = draw(frame.copy(), b_vis, (0, 165, 255), args.line_width, True)
            banner(left, f"pt conf={primary:.2f} | frame {n_frames} | dets {len(a_vis)}")
            banner(right, f"onnx conf={primary:.2f} | frame {n_frames} | dets {len(b_vis)}")
            writers["pt"].write(left)
            writers["onnx"].write(right)
            sl = cv2.resize(left, None, fx=args.compare_scale, fy=args.compare_scale, interpolation=cv2.INTER_AREA)
            sr = cv2.resize(right, None, fx=args.compare_scale, fy=args.compare_scale, interpolation=cv2.INTER_AREA)
            writers["cmp"].write(np.hstack((sl, sr)))

        if n_frames % 100 == 0:
            print(f"[run] {n_frames} frames pt={len(all_pt[primary][-1])} onnx={len(all_on[primary][-1])}")

    f_csv.close()
    cap.release()
    for v in writers.values():
        v.release()
    wall = time.perf_counter() - t0

    print(f"\n[summary] {n_frames} frames processed (max_frames={args.max_frames or 'all'}), skipped=0")
    summary = {"frames": n_frames, "conf": confs, "iou": args.iou, "pt": str(args.pt), "onnx": str(args.onnx),
               "imgsz_pt": pt.imgsz, "imgsz_onnx": onnx.imgsz, "note": NO_GT_NOTE, "per_conf": {}}
    for c in confs:
        sp, so = stats(all_pt[c], n_frames), stats(all_on[c], n_frames)
        summary["per_conf"][f"{c:.2f}"] = {
            "pt": sp, "onnx": so,
            "matched_pairs": pairs[c], "pt_only": only_p[c], "onnx_only": only_o[c],
        }
        print(f"\n--- conf={c:.2f} ---")
        print(f"{'':10}{'pt':>10}{'onnx':>10}")
        for k in ("dets", "mean_per_frame", "frames_with_det", "det_frame_ratio", "mean_conf", "max_conf", "max_per_frame"):
            print(f"{k:<10}{sp[k]:>10.4f}{so[k]:>10.4f}" if isinstance(sp[k], float) else
                  f"{k:<10}{sp[k]:>10}{so[k]:>10}")
        print(f"matched(IoU>{0.5})={pairs[c]}  pt_only={only_p[c]}  onnx_only={only_o[c]}")
    print(f"\n[fps] pt {n_frames / max(t_pt, 1e-9):.2f} fps ({t_pt / max(n_frames, 1) * 1000:.1f} ms/frame) | "
          f"onnx {n_frames / max(t_on, 1e-9):.2f} fps ({t_on / max(n_frames, 1) * 1000:.1f} ms/frame) | "
          f"wall {n_frames / max(wall, 1e-9):.2f} fps")
    print(f"[note] {NO_GT_NOTE}")

    summary["fps"] = {"pt": n_frames / max(t_pt, 1e-9), "onnx": n_frames / max(t_on, 1e-9), "wall": n_frames / max(wall, 1e-9)}
    summary["skipped_frames"] = skipped
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
