"""Indicative only: raw vs preprocessed inference with an existing checkpoint.

The model was trained on RAW images.  Running it on preprocessed images can only
show how much the input distribution moved, NOT whether preprocessing helps.
"""

import os
import sys

sys.path.insert(0, r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics")
sys.path.insert(0, r"G:\0-xiandong\rk3576\ultralytice_self\ultralytics\tools\mrf_preprocess")

import cv2
import numpy as np

import data as D
import filters as F

W = r"G:\0-rk3588\train\ARD-MAV\runs\dwr_nwd_p2_e200\weights\best.pt"
N = 20
CONFS = (0.25, 0.01, 0.001)


def frames(seq, n, max_dim=12):
    lo, hi = 0, min(seq.frame_count - 1, seq.xml_max - 1)
    idxs = sorted(set(int(round(i)) for i in np.linspace(lo, hi, num=n)))
    cap = cv2.VideoCapture(seq.video)
    cur = -1
    for idx in idxs:
        while cur < idx - 1:
            if not cap.grab():
                break
            cur += 1
        ok, frame = cap.read()
        cur += 1
        if not ok:
            break
        boxes = D.read_boxes(seq.ann_dir, seq.name, idx + 1)
        if not boxes:
            continue
        if max(max(b[2] - b[0], b[3] - b[1]) + 1 for b in boxes) > max_dim:
            continue
        yield frame, boxes
    cap.release()


def hits(res, boxes, pad=2):
    """Prediction centre inside a GT box -> hit; return (n_hit, max_conf_of_hits)."""
    if res.boxes is None or len(res.boxes) == 0:
        return 0, float("nan")
    xywh = res.boxes.xyxy.cpu().numpy()
    cf = res.boxes.conf.cpu().numpy()
    n, best = 0, -1.0
    for (x1, y1, x2, y2), c in zip(xywh, cf):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for b in boxes:
            if b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad:
                n += 1
                best = max(best, float(c))
                break
    return n, (best if n else float("nan"))


def main():
    from ultralytics import YOLO

    model = YOLO(W)
    methods = {"raw": lambda bgr: bgr, "mrf_gate_ms": None}
    gate_fn, gate_kw = F.build_methods()["mrf_gate_ms"]

    for cls, names in D.SEQ_CLASS.items():
        for nm in names[:1]:
            seq = D.make_sequence(D.DEFAULT_ROOT, nm, cls)
            acc = {m: dict(frames=0, det25=0, det01=0, det001=0, hit=0, conf=[])
                   for m in methods}
            for bgr, boxes in frames(seq, N):
                g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                o = gate_fn(g, **gate_kw)
                prep = np.repeat(np.clip(o, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
                for m, fn in methods.items():
                    img = bgr if m == "raw" else prep
                    r = model.predict(img, imgsz=640, conf=0.001, iou=0.7,
                                      max_det=300, verbose=False, device=0)[0]
                    a = acc[m]
                    a["frames"] += 1
                    if r.boxes is not None and len(r.boxes):
                        c = r.boxes.conf.cpu().numpy()
                        a["det25"] += int((c >= 0.25).sum())
                        a["det01"] += int((c >= 0.01).sum())
                        a["det001"] += int((c >= 0.001).sum())
                        n, best = hits(r, boxes)
                        a["hit"] += int(n > 0)
                        if n:
                            a["conf"].append(best if not np.isnan(best) else float(c.max()))
            print("[%s/%s] frames=%d" % (cls, nm, acc["raw"]["frames"]), flush=True)
            for m, a in acc.items():
                nf = max(a["frames"], 1)
                print("  %-12s det@.25=%.1f det@.01=%.1f det@.001=%.1f  GT-hit-frame=%.3f  medConf(hits)=%s" % (
                    m, a["det25"] / nf, a["det01"] / nf, a["det001"] / nf, a["hit"] / nf,
                    "%.3f" % float(np.median(a["conf"])) if a["conf"] else "n/a"), flush=True)


if __name__ == "__main__":
    main()
