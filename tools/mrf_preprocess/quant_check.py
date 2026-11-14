"""Quantisation-friendliness check for the preprocessing outputs.

The network input is 8-bit uint8, and an INT8 export quantises the first layer
from that distribution -- so what matters is:

  * how much of the dynamic range the uint8 mapping keeps (clip fractions),
  * how the 3..8 px target amplitude survives uint8 rounding,
  * what a symmetric per-tensor INT8 scale would do to the output.

This is an *estimate* on sampled frames, not a measured quantised-model run.

Usage
  python quant_check.py --frames 40 --methods raw,mrf_res_r2,dog,tophat,mrf_gate
"""

import argparse

import cv2
import numpy as np

import data as D
import evaluate as E
import filters as F


def map_u8(out, mode):
    if mode == "auto":
        mode = "signed" if out.min() < -1.0 else "direct"
    x = out + 128.0 if mode == "signed" else out
    return np.clip(x, 0, 255).astype(np.uint8), mode, float(out.min()), float(out.max())


def int8_sim(out, q=99.9):
    """Symmetric per-tensor INT8 quantisation of the float output."""
    scale = max(float(np.percentile(np.abs(out), q)), 1e-6) / 127.0
    deq = np.round(out / scale).astype(np.float32) * scale
    err = deq - out
    return scale, float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(out) > 0.5 * scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="phantom43,phantom63,phantom22")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--methods", default="raw,mrf_res,mrf_res_r2,mrf_bgsub_ms,mrf_gate_ms,dog,tophat,lcn")
    ap.add_argument("--map", default="auto")
    args = ap.parse_args()

    methods = F.build_methods()
    names = [m for m in args.methods.split(",") if m in methods]
    rows = {m: dict(clip_lo=[], clip_hi=[], dead=[], p1=[], p99=[], std=[],
                    snr=[], iscale=[], ierr=[], ilost=[]) for m in names}

    for nm in args.seqs.split(","):
        seq = D.make_sequence(D.DEFAULT_ROOT, nm, "x")
        for smp in D.iter_samples(seq, n_frames=args.frames, max_dim=12):
            for m in names:
                fn, kw = methods[m]
                out = fn(smp.gray, **kw)
                u8, mode, lo, hi = map_u8(out, args.map)
                r = rows[m]
                r["clip_lo"].append(float(np.mean(out < (0.0 if mode == "direct" else -128.0))))
                r["clip_hi"].append(float(np.mean(out > (255.0 if mode == "direct" else 127.0))))
                r["dead"].append(float(np.mean(u8 == 0)))
                r["p1"].append(float(np.percentile(u8, 1)))
                r["p99"].append(float(np.percentile(u8, 99)))
                r["std"].append(float(u8.std()))
                # target amplitude after uint8 rounding vs float
                s = E.snr_stats(out.astype(np.float32), smp.boxes[0], smp.boxes, *smp.gray.shape)
                s8 = E.snr_stats(u8.astype(np.float32), smp.boxes[0], smp.boxes, *smp.gray.shape)
                if s and s8:
                    r["snr"].append(s8 / s)
                sc, rmse, lost = int8_sim(out)
                r["iscale"].append(sc)
                r["ierr"].append(rmse)
                r["ilost"].append(lost)

    def med(k):
        return float(np.median(rows[m][k])) if rows[m][k] else float("nan")

    print("| method | clip<lo | clip>hi | zero px | p1 | p99 | std | SNR kept by uint8 | int8 scale | int8 RMSE | px < 0.5q |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in names:
        print("| %s | %.4f | %.4f | %.3f | %.0f | %.0f | %.1f | %.3f | %.3f | %.2f | %.3f |" % (
            m, med("clip_lo"), med("clip_hi"), med("dead"), med("p1"), med("p99"), med("std"),
            med("snr"), med("iscale"), med("ierr"), med("ilost")))


if __name__ == "__main__":
    main()
