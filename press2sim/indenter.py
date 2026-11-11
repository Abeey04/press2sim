"""The bottle's path from its two dots: cap height, tilt, and the press events."""
from __future__ import annotations

import numpy as np

from .config import Config


def cap_path(t: np.ndarray, cap_mm: np.ndarray, tape_mm: np.ndarray, cfg: Config, rest_sel=None) -> dict:
    """Bottom-centre of the cap (x, z in mm) and the bottle's tilt from the two front-face dots.

    The cap dot sits cap_dot_mm above the cap's flat face and the tape dot tape_dot_mm above it, both
    on the bottle's axis. The bottle's front face is not in the foam face's plane, so its own scale is
    taken from the dots' measured separation at rest. When one dot is missing, the other one plus the
    cap-to-tape offset interpolated in time stands in for it (the offset changes slowly: tilt and
    perspective only). Tilt is positive when the top leans right; NaN unless both dots are seen.
    """
    ind = cfg.indenter
    want = ind.tape_dot_mm - ind.cap_dot_mm
    d = tape_mm - cap_mm
    L = np.linalg.norm(d, axis=-1)
    both = np.isfinite(L)
    L_rest = L[rest_sel] if rest_sel is not None else L
    L0 = float(np.nanmedian(L_rest)) if np.isfinite(L_rest).any() else float(np.nanmedian(L))
    k = L0 / want if np.isfinite(L0) else 1.0
    off = np.where(both[:, None], cap_mm - tape_mm, np.nan)          # tape dot -> cap dot
    for c in range(2):
        good = np.isfinite(off[:, c])
        if good.sum() >= 2:
            off[:, c] = np.interp(t, t[good], off[good, c])
        elif good.any():
            off[:, c] = off[good, c][0]
    have_cap = np.isfinite(cap_mm[:, 0])
    cap_est = np.where(have_cap[:, None], cap_mm, tape_mm + off)
    axis = -off / np.linalg.norm(off, axis=-1, keepdims=True)         # unit vector cap -> tape
    bottom = cap_est - ind.cap_dot_mm * k * axis
    tilt = np.where(both, np.degrees(np.arctan2(axis[:, 0], axis[:, 1])), np.nan)
    source = np.where(both, 2, np.where(have_cap, 1, np.where(np.isfinite(tape_mm[:, 0]), 3, 0)))
    # A cap position seen in a single frame with nothing either side of it is not trusted: once the bottle
    # is lost, a stray dark blob can pass for one of its dots for a frame and would set the deepest point.
    seen = np.isfinite(bottom[:, 1])
    lone = seen & ~np.r_[False, seen[:-1]] & ~np.r_[seen[1:], False]
    bottom[lone] = np.nan
    tilt[lone] = np.nan
    source[lone] = 0
    return {"x_mm": bottom[:, 0], "z_mm": bottom[:, 1], "tilt_deg": tilt, "dot_separation_mm": L,
            "source": source, "bottle_scale": k, "separation_rest_mm": L0}


def _interp_cross(t, y, level, i):
    """Time where y crosses `level` between samples i-1 and i (linear)."""
    if i <= 0:
        return float(t[i])
    y0, y1 = y[i - 1], y[i]
    if not (np.isfinite(y0) and np.isfinite(y1)) or y1 == y0:
        return float(t[i])
    return float(t[i - 1] + (level - y0) / (y1 - y0) * (t[i] - t[i - 1]))


def events(t: np.ndarray, cap_z: np.ndarray, top_z: float, cfg: Config, foam_top_dot_z: np.ndarray | None = None):
    """Contact, deepest point, lift and release times, plus depth and speed summaries.

    depth = top_z - cap_z. Contact: depth first exceeds contact_threshold_mm. Lift: after the deepest
    point the cap starts rising fast. Release: the gap between the cap and the foam under it grows
    release_gap_mm beyond its value at the deepest point (needs the top-row dot under the cap);
    without it, release falls back to the time the cap is back above the original top surface.
    """
    tun = cfg.tuning
    depth = top_z - cap_z
    ok = np.isfinite(depth)
    out = {"max_depth_mm": float("nan")}
    if ok.sum() < 5 or np.nanmax(depth) < tun.contact_threshold_mm:
        out["note"] = "no press found"
        return out
    i_c = int(np.argmax(np.where(ok, depth, -np.inf) >= tun.contact_threshold_mm))
    i_m = int(np.nanargmax(np.where(ok, depth, np.nan)))
    out.update(t_contact=_interp_cross(t, depth, tun.contact_threshold_mm, i_c), t_max=float(t[i_m]),
               max_depth_mm=float(depth[i_m]))
    # press speed: 10% -> 90% of max depth on the way down
    lo, hi = 0.1 * depth[i_m], 0.9 * depth[i_m]
    seg = np.arange(i_c, i_m + 1)
    a = seg[np.argmax(depth[seg] >= lo)]; b = seg[np.argmax(depth[seg] >= hi)]
    if t[b] > t[a]:
        out["press_speed_mm_s"] = float((depth[b] - depth[a]) / (t[b] - t[a]))
    # hold: time within 0.5 mm of the deepest point
    near = ok & (depth >= depth[i_m] - 0.5)
    idx = np.where(near)[0]
    out["hold_s"] = float(t[idx.max()] - t[idx.min()]) if len(idx) else 0.0
    # lift: first frame after the hold where the cap rises faster than 20 mm/s
    after = np.arange(idx.max() if len(idx) else i_m, len(t))
    vz = np.gradient(cap_z, t)
    rising = after[np.isfinite(vz[after]) & (vz[after] > 20.0)]
    if len(rising):
        out["t_lift"] = float(t[rising[0]])
        up = vz[after][np.isfinite(vz[after])]
        out["release_speed_mm_s"] = float(np.nanmax(up)) if len(up) else float("nan")
    rel = None
    if foam_top_dot_z is not None and np.isfinite(foam_top_dot_z[i_m]):
        gap = cap_z - foam_top_dot_z
        g0 = gap[i_m]
        cand = after[np.isfinite(gap[after]) & (gap[after] > g0 + tun.release_gap_mm)]
        if len(cand):
            rel = float(t[cand[0]])
            out["release_from"] = "gap to top-row dot"
    if rel is None:
        cand = after[ok[after] & (depth[after] < 0)]
        if len(cand):
            rel = float(t[cand[0]])
            out["release_from"] = "cap above original surface"
    if rel is not None:
        out["t_release"] = rel
    return out
