"""MRF-style background-suppression filters and classical baselines.

All functions operate on a single-channel float32 image (gray levels 0..255) and
return a float32 image of the same size.  Pure CPU (numpy + OpenCV), no model.

Neighbourhood: ``nbhd`` selects 4- or 8-connectivity, ``radius`` its size.  With
radius=1 the cloud is the 3x3 ring of the patent, radius=2 the 5x5 cloud, etc.
The radius should be matched to the target size (a 3 px target needs radius 1,
an 8 px target needs radius 2-3), see docs/mrf_preprocess_report.md.
"""

import functools

import cv2
import numpy as np


@functools.lru_cache(maxsize=32)
def kernel(nbhd=8, radius=1):
    """Normalised neighbour kernel with the centre pixel excluded."""
    radius, nbhd = int(radius), int(nbhd)
    n = 2 * radius + 1
    k = np.ones((n, n), np.float32)
    k[radius, radius] = 0.0
    if nbhd == 4:
        cross = np.zeros((n, n), np.float32)
        cross[radius, :] = 1.0
        cross[:, radius] = 1.0
        cross[radius, radius] = 0.0
        k = cross
    return k / k.sum()


def neigh_mean(x, nbhd=8, radius=1):
    return cv2.filter2D(x, -1, kernel(nbhd, radius), borderType=cv2.BORDER_REFLECT)


def roughness(x, nbhd=8, radius=1):
    """Patent eq. (10): d_c x = mean(neighbourhood) - centre.  Signed."""
    return neigh_mean(x, nbhd, radius) - x


# --------------------------------------------------------------------------- #
# Potentials rho
# --------------------------------------------------------------------------- #
def rho_patent_grad(t, tau):
    """rho'(t) of the potential that satisfies patent properties (12)/(13).

        rho'(t) = tau * (u - atan(u)),  u = t / tau   (M normalised to 1)
    =>  rho'(t)/t = 1 - atan(u)/u  -> 0 as t->0,  -> 1 as t->inf
        rho''(t) = 1/(1+u^2) >= 0        (convex)
        rho'(t) >= 0 and increasing      (monotone)
        C^1 (C^inf), rho(0)=0.

    The classic robust potentials (Huber, Charbonnier, Tukey, Geman-McClure) do
    NOT satisfy (12)/(13): their diffusivity rho'(t)/t is maximal at t=0 and
    vanishes for large t, i.e. the opposite behaviour.  They are therefore only
    usable for the forward (background-estimating) diffusion below, never as the
    patent potential itself.
    """
    u = np.asarray(t, np.float32) / np.float32(tau)
    return np.float32(tau) * (u - np.arctan(u))


def rho_patent_ratio(t, tau):
    """rho'(t)/t normalised to [0, 1).  Zero for zero roughness."""
    u = np.asarray(t, np.float32) / np.float32(tau)
    out = np.ones_like(u)
    nz = u > 1e-4
    out[nz] = 1.0 - np.arctan(u[nz]) / u[nz]
    out[~nz] = (u[~nz] ** 2) / 3.0
    return out


def gate_exp(d, tau):
    """Soft gate w(d) = 1 - exp(-(d/tau)^2) in [0,1)."""
    return 1.0 - np.exp(-(np.asarray(d, np.float32) / np.float32(tau)) ** 2)


def gate_rho(d, tau):
    """Gate derived from the patent potential: w = rho'(d)/d (normalised)."""
    return rho_patent_ratio(np.abs(d), tau)


# --------------------------------------------------------------------------- #
# Robust weights used for *forward* (background-estimating) diffusion
# --------------------------------------------------------------------------- #
def _w_charbonnier(d, tau):
    return 1.0 / np.sqrt(1.0 + (d / np.float32(tau)) ** 2)


def _w_geman(d, tau):
    u = (d / np.float32(tau)) ** 2
    return 1.0 / (1.0 + u) ** 2


def _w_tukey(d, tau):
    u = (d / np.float32(tau)) ** 2
    return np.where(u < 1.0, (1.0 - u) ** 2, 0.0).astype(np.float32)


WEIGHTS = {"charbonnier": _w_charbonnier, "geman": _w_geman, "tukey": _w_tukey}


def bg_smooth(x, tau=8.0, iters=6, nbhd=8, radius=1, weight="tukey", scale=1):
    """Edge-preserving background estimate.

        b <- b + c(d) * (mean_neighbours(b) - b)

    with c(d) the robust weight (small roughness -> c ~ 1, large -> c ~ 0).
    This is the conjugate ("forward diffusion") counterpart of the patent
    potential: it keeps the background and stops at the target.

    scale > 1 estimates the background on a down-sampled image (small targets
    vanish at 1/scale resolution) and upsamples: the multi-scale mode.
    """
    wf = WEIGHTS[weight]
    scale = int(scale)
    if scale > 1:
        h, w = x.shape
        small = cv2.resize(x, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    else:
        small = x
    b = small.astype(np.float32, copy=True)
    for _ in range(int(iters)):
        d = neigh_mean(b, nbhd, radius) - b
        b = b + wf(d, tau) * d
    if scale > 1:
        b = cv2.resize(b, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
    return b


# --------------------------------------------------------------------------- #
# MRF variants
# --------------------------------------------------------------------------- #
def mrf_residual(x, nbhd=8, radius=1, polarity="signed", tau=8.0):
    """(a) Roughness residual map, i.e. patent eq. (10) used as the output."""
    d = roughness(x, nbhd, radius)
    if polarity == "signed":
        out = d
    elif polarity == "abs":
        out = np.abs(d)
    elif polarity == "gate":
        out = d * rho_patent_ratio(np.abs(d), tau)
    else:
        raise ValueError(polarity)
    return out.astype(np.float32)


def mrf_bgsub(x, tau=8.0, iters=6, nbhd=8, radius=1, weight="tukey", scale=1,
              polarity="signed", gain=1.0):
    """(b) edge-preserving background estimate, output = x - background."""
    bg = bg_smooth(x, tau, iters, nbhd, radius, weight, scale)
    r = x - bg
    if polarity == "abs":
        r = np.abs(r)
    return (gain * r).astype(np.float32)


def mrf_gate(x, tau=8.0, iters=6, nbhd=8, radius=1, weight="tukey", scale=1,
             gate="exp", polarity="signed", gain=1.0):
    """(c) out = bg + w(d)*(x - bg):  background flattened, target kept."""
    bg = bg_smooth(x, tau, iters, nbhd, radius, weight, scale)
    r = x - bg
    d = np.abs(r)
    w = gate_exp(d, tau) if gate == "exp" else gate_rho(d, tau)
    out = bg + w * r
    if polarity == "abs":
        out = np.abs(out - bg)
    return (gain * out).astype(np.float32)


def mrf_flux(x, tau=8.0, lam=1.0, nbhd=8, radius=1, iters=1, polarity="signed"):
    """(d) Patent-faithful fixed point of J(u)=1/2||u-x||^2 + lam*sum_c rho(d_c u).

        dJ/du = (u-x) + lam*(I-A)*rho'(d) = 0  =>  u = x + lam*(I-A)*rho'(d)

    (I-A) is the neighbourhood Laplacian (A = neighbour mean, symmetric), so this
    is literally "x convolved with an anisotropic weight matrix" (patent, last
    paragraph of the MRF section).
    """
    u = x.astype(np.float32, copy=True)
    for _ in range(int(iters)):
        d = roughness(u, nbhd, radius)
        u = u + np.float32(lam) * (d - neigh_mean(d, nbhd, radius))
    if polarity == "abs":
        u = np.abs(u - x)
    return u.astype(np.float32)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def dog(x, s1=1.0, s2=3.0):
    """Difference of Gaussians (band-pass around the target scale)."""
    return (cv2.GaussianBlur(x, (0, 0), s1) - cv2.GaussianBlur(x, (0, 0), s2)).astype(np.float32)


def tophat(x, ksize=15):
    """Signed top-hat / black-hat: bright targets positive, dark ones negative."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ksize), int(ksize)))
    th = cv2.morphologyEx(x, cv2.MORPH_TOPHAT, k)
    bh = cv2.morphologyEx(x, cv2.MORPH_BLACKHAT, k)
    return (th - bh).astype(np.float32)


def lcn(x, win=15, eps=1e-3):
    """Local contrast normalisation (x - mean) / (std + eps)."""
    m = cv2.blur(x, (int(win), int(win)))
    m2 = cv2.blur(x * x, (int(win), int(win)))
    var = np.maximum(m2 - m * m, 0.0)
    return ((x - m) / (np.sqrt(var) + np.float32(eps))).astype(np.float32)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_methods():
    """Ordered dict name -> (callable, kwargs).  Edit here to sweep."""
    return {
        "raw": (lambda x, **kw: x.astype(np.float32), {}),
        # ---- MRF variants ------------------------------------------------
        "mrf_res": (mrf_residual, {"nbhd": 8, "radius": 1}),
        "mrf_res_r2": (mrf_residual, {"nbhd": 8, "radius": 2}),
        "mrf_res_r3": (mrf_residual, {"nbhd": 8, "radius": 3}),
        "mrf_bgsub": (mrf_bgsub, {"tau": 8.0, "iters": 6, "weight": "tukey", "scale": 1}),
        "mrf_bgsub_ms": (mrf_bgsub, {"tau": 8.0, "iters": 6, "weight": "tukey", "scale": 4}),
        "mrf_gate": (mrf_gate, {"tau": 8.0, "iters": 6, "weight": "tukey", "scale": 1}),
        "mrf_gate_ms": (mrf_gate, {"tau": 8.0, "iters": 6, "weight": "tukey", "scale": 4}),
        "mrf_gate_rho": (mrf_gate, {"tau": 8.0, "iters": 6, "weight": "tukey", "scale": 1, "gate": "rho"}),
        "mrf_flux": (mrf_flux, {"tau": 8.0, "lam": 1.0}),
        # ---- classical baselines -----------------------------------------
        "dog": (dog, {"s1": 1.0, "s2": 3.0}),
        "tophat": (tophat, {"ksize": 15}),
        "lcn": (lcn, {"win": 15}),
    }
