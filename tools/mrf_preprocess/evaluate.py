"""Real-data comparison of MRF preprocessing vs classical baselines.

Every preprocessed image goes through ONE common, model-free reader (local
contrast), so results are comparable without training a detector:

    D(x) = |x - box_mean_11(x)|
    Z(x) = (D - med(D_bg)) / (1.4826 * MAD(D_bg))        # bg = non-GT pixels

Metrics
  * local SNR   : max(D) in the GT box / robust sigma of the surrounding ring
                  (per target -> local contrast gain, median / p25 / frac<0.8)
  * AUC / d'    : GT peaks (positives) vs background pixels (negatives)
  * rec@FA=1/f  : detection rate at exactly 1 false peak per frame
  * FA@rec=0.9  : false peaks per frame at the threshold that recovers 90 % of
                  the targets -- the metric that matches the real complaint
                  ("lower the threshold and everything is a false alarm")
  * time        : per-frame preprocessing cost, P50 / P95

Usage
  python evaluate.py --frames 200 --max-dim 12 --out eval.json --md eval.md
"""

import argparse
import itertools
import json
import math
import time
from collections import defaultdict

import cv2
import numpy as np

import data as D
import filters as F

SIGMA_FLOOR = 0.5      # 8-bit sensor noise floor, gray levels
SCORE_FLOOR = 0.5      # floor of the local high-pass std (gray levels)
READER_K = 11          # common reader kernel (bigger than the 3..8 px target)
CFAR_W = 31            # window of the local std used by the detection score
NMS_K = 5              # 5x5 local-max suppression (>=2 px separation)

# log-spaced z bins: handles the heavy-tailed background of natural images
ZMIN, ZMAX, NB = 0.5, 400.0, 96
LOGR = math.log(ZMAX / ZMIN)


def zbin(v):
    t = np.log(np.maximum(np.asarray(v, np.float32), ZMIN) / ZMIN) / LOGR * NB
    return np.clip(t.astype(np.int32), 0, NB - 1)


def zval(b):
    return float(ZMIN * math.exp((b + 0.5) / NB * LOGR))


def reader(x, k=READER_K):
    """Uniform, method-agnostic local-contrast statistic (high-pass)."""
    return np.abs(x - cv2.blur(x, (k, k)))


def detect_map(x, k=READER_K, w=CFAR_W):
    """CFAR-like detection score, identical for every method.

    L(x) = |x - box_mean_k(x)| / (local std of that high-pass over w x w + floor)

    Locally normalised, so a flat scene does not turn 8-bit banding into huge
    scores, and a textured scene is correctly harder.
    """
    D = np.abs(x - cv2.blur(x, (k, k)))
    mu = cv2.blur(D, (w, w))
    mu2 = cv2.blur(D * D, (w, w))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    return D / (sd + SCORE_FLOOR)


def rad(v):
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def hist_add(hist, vals):
    if vals.size == 0:
        return
    hist += np.bincount(zbin(vals), minlength=NB)


def snr_stats(img, box, boxes, H, W):
    """max deviation in the GT box / robust sigma of the surrounding ring.

    Applied directly to the *output image* (no extra filtering), so for a
    residual image this is the residual amplitude and for a normalised image it
    is the local contrast.  Identical definition for every method.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if x2 - x1 + 1 < 2 or y2 - y1 + 1 < 2:
        return None
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(W - 1, x2), min(H - 1, y2)
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None
    g = 4                                   # guard band: skip target spill
    m = g + max(6, int(round(1.2 * max(x2 - x1 + 1, y2 - y1 + 1))))
    ox1, oy1 = max(0, ix1 - m), max(0, iy1 - m)
    ox2, oy2 = min(W - 1, ix2 + m), min(H - 1, iy2 + m)
    patch_in = img[iy1:iy2 + 1, ix1:ix2 + 1]
    patch_out = img[oy1:oy2 + 1, ox1:ox2 + 1]
    mask = np.ones(patch_out.shape, bool)
    mx1, my1 = max(0, ix1 - g), max(0, iy1 - g)
    mx2, my2 = min(W - 1, ix2 + g), min(H - 1, iy2 + g)
    mask[my1 - oy1:my2 - oy1 + 1, mx1 - ox1:mx2 - ox1 + 1] = False
    for ob in boxes:
        if ob is box:
            continue
        bx1, by1, bx2, by2 = [int(round(v)) for v in ob]
        bx1 = max(ox1, bx1 - 2) - ox1
        by1 = max(oy1, by1 - 2) - oy1
        bx2 = min(ox2, bx2 + 2) - ox1
        by2 = min(oy2, by2 + 2) - oy1
        if bx2 > bx1 and by2 > by1:
            mask[max(0, by1):by2 + 1, max(0, bx1):bx2 + 1] = False
    ring = patch_out[mask]
    if ring.size < 25:
        return None
    med = float(np.median(ring))
    amp = float(np.abs(patch_in - med).max())
    return amp / max(rad(ring), SIGMA_FLOOR)


def gt_mask(shape, boxes, pad):
    m = np.zeros(shape, np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(m, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 1, -1)
    return m


def new_acc():
    return dict(n_frames=0, n_targets=0,
                maxima_hist=np.zeros(NB, np.int64),
                pos_z=[], snr_ratio=[], times=[])


def add_acc(a, b):
    for k in ("n_frames", "n_targets"):
        a[k] += b[k]
    a["maxima_hist"] += b["maxima_hist"]
    for k in ("pos_z", "snr_ratio", "times"):
        a[k].extend(b[k])


def counts_above(hist, thr):
    return int(hist[int(zbin(thr)):].sum())


def summarize(acc):
    n = max(acc["n_frames"], 1)
    times = np.asarray(acc["times"]) * 1000.0
    sr = np.asarray(acc["snr_ratio"], np.float32)
    pz = np.asarray(acc["pos_z"], np.float32) if acc["pos_z"] else np.zeros(0, np.float32)

    # --- operating point with 1 false peak per frame
    cum_top = np.cumsum(acc["maxima_hist"][::-1])
    k = int(np.searchsorted(cum_top, 1.0 * n, side="left"))
    b = max(0, NB - 1 - min(k, NB - 1))
    thr_fa1 = zval(b)
    rec_fa1 = float(np.mean(pz >= thr_fa1)) if pz.size else float("nan")
    fa1 = int(acc["maxima_hist"][b:].sum()) / n

    # --- false alarms at fixed recall
    def fa_at(rec):
        if pz.size == 0:
            return float("nan")
        thr = float(np.quantile(pz, 1.0 - rec))
        return counts_above(acc["maxima_hist"], thr) / n

    # --- AUC / d' against the background *peaks* (the same population that
    #     produces false alarms), not against all background pixels: both sides
    #     are extremes, so the AUC is not inflated by the max-over-box bias.
    cnt = acc["maxima_hist"].astype(np.float64)
    total = cnt.sum()
    if pz.size and total > 0:
        cum = np.cumsum(cnt)
        idx = zbin(pz)
        below = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
        auc = float(np.mean((below + 0.5 * cnt[idx]) / total))
    else:
        auc = float("nan")
    zc = ZMIN * np.exp((np.arange(NB) + 0.5) / NB * LOGR)
    mneg = float((cnt * zc).sum() / cnt.sum()) if cnt.sum() else 0.0
    vneg = float((cnt * (zc - mneg) ** 2).sum() / cnt.sum()) if cnt.sum() else 1.0
    dprime = float((pz.mean() - mneg) / math.sqrt(0.5 * (pz.var() + max(vneg, 1e-9))) ) if pz.size else float("nan")

    return dict(
        n_frames=int(acc["n_frames"]), n_targets=int(acc["n_targets"]),
        snr_gain_median=float(np.median(sr)) if sr.size else float("nan"),
        snr_gain_p25=float(np.percentile(sr, 25)) if sr.size else float("nan"),
        frac_gain_lt_08=float(np.mean(sr < 0.8)) if sr.size else float("nan"),
        auc=auc, dprime=dprime,
        score_med=float(np.median(pz)) if pz.size else float("nan"),
        thr_fa1=thr_fa1,
        rec_at_fa1=rec_fa1, fa_per_frame=fa1,
        fa_at_rec90=fa_at(0.90), fa_at_rec50=fa_at(0.50),
        p50_ms=float(np.percentile(times, 50)) if times.size else float("nan"),
        p95_ms=float(np.percentile(times, 95)) if times.size else float("nan"),
    )


def run(args):
    seqs = []
    for cls, names in D.SEQ_CLASS.items():
        for nm in names:
            seqs.append((cls, D.make_sequence(args.root, nm, cls)))
    if args.seqs:
        want = set(args.seqs.split(","))
        seqs = [(c, s) for c, s in seqs if s.name in want]

    methods = F.build_methods()
    for line in (args.set or []):
        name, kv = line.split(":")
        fn, kw = methods[name]
        for item in kv.split(","):
            k, v = item.split("=")
            kw[k] = float(v) if v.replace(".", "", 1).isdigit() else v
        methods[name] = (fn, kw)
    # generic parameter grid: --grid "mrf_gate:tau=4,8,16,32" (repeatable, same
    # base expands as a Cartesian product, variant names get a _keyvalue suffix)
    grid = {}
    for g in (args.grid or []):
        base, kv = g.split(":")
        k, vals = kv.split("=")
        grid.setdefault(base, {})[k] = [float(v) if v.replace(".", "", 1).isdigit() else v
                                        for v in vals.split(",")]
    for base, spec in grid.items():
        fn, kw = methods[base]
        keys = list(spec)
        for combo in itertools.product(*[spec[k] for k in keys]):
            name = base + "".join("_%s%s" % (k[:4], "%g" % v if isinstance(v, float) else v)
                                  for k, v in zip(keys, combo))
            methods[name] = (fn, {**kw, **dict(zip(keys, combo))})
    if args.methods:
        keep = set(args.methods.split(","))
        methods = {k: v for k, v in methods.items() if k in keep}

    per_seq = defaultdict(lambda: defaultdict(new_acc))
    per_cls = defaultdict(lambda: defaultdict(new_acc))
    nsamples = defaultdict(int)
    bk = defaultdict(lambda: defaultdict(list))   # size bucket -> method -> SNR gains

    for cls, seq in seqs:
        print("[seq] %-10s class=%-6s frames=%d" % (seq.name, cls, seq.frame_count), flush=True)
        for smp in D.iter_samples(seq, n_frames=args.frames, max_dim=args.max_dim):
            nsamples[cls] += 1
            H, W = smp.gray.shape
            boxes = smp.boxes
            gm8 = gt_mask((H, W), boxes, 8)
            snr_raw = [snr_stats(smp.gray, b, boxes, H, W) for b in boxes]

            for name, (fn, kw) in methods.items():
                t0 = time.perf_counter()
                out = fn(smp.gray, **kw)
                dt = time.perf_counter() - t0
                Z = detect_map(out)

                acc = per_seq[seq.name][name]
                mx = cv2.dilate(Z, np.ones((NMS_K, NMS_K), np.uint8))
                pm = (Z >= mx) & (Z >= ZMIN) & (gm8 == 0)
                hist_add(acc["maxima_hist"], Z[pm])

                zs, ratios = [], []
                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = [int(round(t)) for t in b]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W - 1, x2), min(H - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    zs.append(float(Z[y1:y2 + 1, x1:x2 + 1].max()))
                    rs = snr_stats(out, b, boxes, H, W)
                    if rs is not None and snr_raw[i] is not None:
                        ratios.append(rs / snr_raw[i])
                        side = max(x2 - x1, y2 - y1) + 1
                        bucket = "s<=6" if side <= 6 else ("s7-9" if side <= 9 else "s10-12")
                        bk[bucket][name].append(rs / snr_raw[i])
                acc["pos_z"].extend(zs)
                acc["snr_ratio"].extend(ratios)
                acc["times"].append(dt)
                acc["n_frames"] += 1
                acc["n_targets"] += len(zs)

        for name in methods:
            add_acc(per_cls[cls][name], per_seq[seq.name][name])

    results = {c: {m: summarize(a) for m, a in mm.items()} for c, mm in per_cls.items()}
    per_s = {s: {m: summarize(a) for m, a in mm.items()} for s, mm in per_seq.items()}
    payload = dict(config={k: v for k, v in vars(args).items()},
                   samples=dict(nsamples), results=results, per_sequence=per_s)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
        print("saved", args.out)

    lines = []
    for cls in results:
        r = results[cls]
        lines.append("\n### %s (frames=%d, targets=%d)\n" % (cls, r["raw"]["n_frames"], r["raw"]["n_targets"]))
        lines.append("| method | SNR med | SNR p25 | gain<0.8 | AUC | d' | score med | thr@FA1 | rec@FA=1 | FA/f | FA@rec.9 | FA@rec.5 | P50 ms | P95 ms |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in methods:
            s = r[m]
            g = "-" if m == "raw" else "%.2fx" % s["snr_gain_median"]
            p = "-" if m == "raw" else "%.2fx" % s["snr_gain_p25"]
            lines.append("| %s | %s | %s | %.0f%% | %.3f | %.2f | %.1f | %.1f | %.3f | %.0f | %.0f | %.0f | %.1f | %.1f |" % (
                m, g, p, 100 * s["frac_gain_lt_08"], s["auc"], s["dprime"], s["score_med"], s["thr_fa1"],
                s["rec_at_fa1"], s["fa_per_frame"], s["fa_at_rec90"], s["fa_at_rec50"],
                s["p50_ms"], s["p95_ms"]))
    # ---- size-bucket table (all classes pooled)
    lines.append("\n### SNR gain by target size (all classes)\n")
    lines.append("| method | " + " | ".join(
        "%s (n=%d)" % (b, len(bk[b][next(iter(methods))])) for b in ("s<=6", "s7-9", "s10-12") if b in bk) + " |")
    lines.append("|---" * (1 + len([b for b in ("s<=6", "s7-9", "s10-12") if b in bk])) + "|")
    for m in methods:
        cells = []
        for b in ("s<=6", "s7-9", "s10-12"):
            if b not in bk:
                continue
            v = np.asarray(bk[b].get(m, []), np.float32)
            cells.append("-" if (m == "raw" or not v.size) else "%.2fx" % np.median(v))
        lines.append("| %s | %s |" % (m, " | ".join(cells)))

    txt = "\n".join(lines)
    print(txt)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(txt)
        print("saved", args.md)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=D.DEFAULT_ROOT)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--max-dim", type=float, default=12.0)
    ap.add_argument("--methods", default="")
    ap.add_argument("--set", action="append", default=[], help="mrf_gate:tau=16,iters=4")
    ap.add_argument("--grid", action="append", default=[], help='e.g. "mrf_gate:tau=4,8,16"')
    ap.add_argument("--out", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    run(args)
