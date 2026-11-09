"""Finding the black dots: candidates, grid labelling, sub-pixel centres, bottle dots, foam outline."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .config import Config


def roi_from_mm(calib, x_mm, z_mm, shape, pad_px=0):
    """Pixel bounding box (x0, y0, x1, y1) of a rectangle given in card millimetres."""
    c = np.array([[x_mm[0], z_mm[0]], [x_mm[1], z_mm[0]], [x_mm[1], z_mm[1]], [x_mm[0], z_mm[1]]])
    p = calib.mm_to_px(c)
    x0 = int(max(0, np.floor(p[:, 0].min()) - pad_px)); x1 = int(min(shape[1], np.ceil(p[:, 0].max()) + pad_px))
    y0 = int(max(0, np.floor(p[:, 1].min()) - pad_px)); y1 = int(min(shape[0], np.ceil(p[:, 1].max()) + pad_px))
    return x0, y0, x1, y1


def contrast_image(gray: np.ndarray, roi, diam_px: float) -> np.ndarray:
    """How much darker each pixel is than its surroundings (dots positive, foam and sparkles ~0).

    A morphological closing with a kernel wider than a dot removes the dots and gives the local
    background; unlike a max filter it does not spread the foam's bright sparkles.
    """
    x0, y0, x1, y1 = roi
    sub = gray[y0:y1, x0:x1]
    k = int(2 * round(1.1 * diam_px) + 1)
    bg = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return (bg.astype(np.float32) - sub.astype(np.float32)).clip(0, None)


def blobs(contrast: np.ndarray, thr: float, diam_px: float, offset=(0, 0), min_area_ratio=0.15,
          max_area_ratio=4.0):
    """Connected dark blobs above thr: contrast-weighted centroids (px), areas, peak contrast."""
    lab_n, lab, st, _ = cv2.connectedComponentsWithStats((contrast > thr).astype(np.uint8), connectivity=8)
    if lab_n <= 1:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    w = np.where(lab > 0, contrast - thr, 0).ravel()
    yy, xx = np.indices(contrast.shape)
    L = lab.ravel()
    sw = np.bincount(L, w, lab_n)
    cx = np.bincount(L, w * xx.ravel(), lab_n) / np.maximum(sw, 1e-9)
    cy = np.bincount(L, w * yy.ravel(), lab_n) / np.maximum(sw, 1e-9)
    peak = ndimage.maximum(contrast, lab, np.arange(lab_n))
    area = st[:, cv2.CC_STAT_AREA].astype(float)
    bw, bh = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    a0 = np.pi * (diam_px / 2) ** 2
    ok = np.ones(lab_n, bool)
    ok[0] = False
    ok &= (area >= min_area_ratio * a0) & (area <= max_area_ratio * a0)
    ok &= area / np.maximum(bw * bh, 1) > 0.3
    ok &= np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1) < 5
    c = np.c_[cx, cy][ok] + np.asarray(offset, float)
    return c, area[ok], np.asarray(peak)[ok]


def candidates(gray, roi, diam_px, frac=0.4):
    """Dot candidates in a region, threshold set from the strongest dark features in it."""
    C = contrast_image(gray, roi, diam_px)
    hi = np.percentile(C, 99.7)
    thr = max(12.0, frac * hi)
    pts, area, peak = blobs(C, thr, diam_px, offset=roi[:2])
    return pts, area, peak, thr


def _best_offset(vals, pitch, n, lo, hi, tol):
    """1-D lattice search: offset in [lo, hi] maximising points within tol of offset + pitch*k."""
    best = (-1, np.inf, lo)
    for off in np.arange(lo, hi, 0.1 * tol):
        k = np.round((vals - off) / pitch)
        inside = (k >= 0) & (k < n)
        d = np.abs(vals - (off + pitch * k))
        hit = inside & (d < tol)
        score = int(np.unique(k[hit]).size * 1000 + hit.sum())
        err = d[hit].mean() if hit.any() else np.inf
        if score > best[0] or (score == best[0] and err < best[1]):
            best = (score, err, off)
    return best[2]


def label_grid(pts: np.ndarray, rows: int, cols: int, pitch: float, z0_range=None, x0_range=None,
               tol: float | None = None):
    """Assign candidate points (in a frame where the grid pitch is `pitch` and z points up) to grid nodes.

    Returns (rows, cols, 2) positions with NaN for missing nodes, plus a report. Works in card
    millimetres (z up) or in pixels flipped so that z points up.
    """
    tol = 0.3 * pitch if tol is None else tol
    x, z = pts[:, 0], pts[:, 1]
    zr = z0_range or (z.min() - pitch, z.max())
    z0 = _best_offset(z, pitch, rows, *zr, tol)
    near_rows = np.abs(z - (z0 + pitch * np.clip(np.round((z - z0) / pitch), 0, rows - 1))) < tol
    xr = x0_range or (x.min() - pitch, x.max())
    x0 = _best_offset(x[near_rows], pitch, cols, *xr, tol)
    row_z = z0 + pitch * np.arange(rows)
    col_x = x0 + pitch * np.arange(cols)
    grid = np.full((rows, cols, 2), np.nan)
    for _ in range(3):                      # assign, then let each row and column settle to its dots
        grid[:] = np.nan
        nodes = np.stack(np.meshgrid(col_x, row_z), -1)           # (rows, cols, 2)
        d = np.linalg.norm(pts[None, None] - nodes[:, :, None], axis=-1)   # (rows, cols, M)
        order = np.argsort(d, axis=None)
        used_p, used_n = set(), set()
        lim = 0.45 * pitch
        for flat in order:
            r, c, m = np.unravel_index(flat, d.shape)
            if d[r, c, m] > lim:
                break
            if (r, c) in used_n or m in used_p:
                continue
            grid[r, c] = pts[m]
            used_n.add((r, c)); used_p.add(m)
        rz = np.nanmean(grid[..., 1], axis=1)
        cx = np.nanmean(grid[..., 0], axis=0)
        row_z = np.where(np.isfinite(rz), rz, row_z)
        col_x = np.where(np.isfinite(cx), cx, col_x)
    found = np.isfinite(grid[..., 0])
    fit = np.stack(np.meshgrid(col_x, row_z), -1)
    dev = np.linalg.norm(grid - fit, axis=-1)[found]
    rep = {"found": int(found.sum()), "expected": rows * cols,
           "row_z": row_z.tolist(), "col_x": col_x.tolist(),
           "pitch_x_mean": float(np.mean(np.diff(col_x))), "pitch_z_mean": float(np.mean(np.diff(row_z))),
           "marking_dev_rms": float(np.sqrt(np.mean(dev ** 2))) if dev.size else float("nan"),
           "unused_candidates": int(len(pts) - found.sum())}
    return grid, rep


def find_grid(gray, calib, cfg: Config, frac=0.4):
    """Detect and label the 15 x 7 dot grid in one frame. Returns pixel grid (rows, cols, 2) and report."""
    g = cfg.grid
    ppm = calib.report["px_per_mm"]
    diam = g.dot_diameter_mm * ppm
    roi = roi_from_mm(calib, (-60, 340), (g.z0_mm - 6, g.z0_mm + g.pitch_mm * (g.rows - 1) + 6.5), gray.shape,
                      pad_px=int(diam))
    pts, area, peak, thr = candidates(gray, roi, diam, frac)
    if len(pts) < 10:
        empty = np.full((g.rows, g.cols, 2), np.nan)
        return empty, empty.copy(), {"found": 0, "expected": g.rows * g.cols, "candidates": int(len(pts))}
    mm = calib.px_to_mm(pts)
    grid_mm, rep = label_grid(mm, g.rows, g.cols, g.pitch_mm,
                              z0_range=(g.z0_mm - 4, g.z0_mm + 4),
                              x0_range=(mm[:, 0].min() - 5, mm[:, 0].max() + g.pitch_mm))
    px = calib.mm_to_px(grid_mm)
    rep.update(threshold=float(thr), candidates=int(len(pts)), diam_px=float(diam))
    # Rest-state appearance of each dot, for validity checks while tracking.
    C = contrast_image(gray, roi, diam)
    _, ar, pk = blobs(C, thr, diam, offset=roi[:2])
    rep["dot_area_px_median"] = float(np.median(area)) if len(area) else float("nan")
    rep["dot_peak_median"] = float(np.median(peak)) if len(peak) else float("nan")
    return px, grid_mm, rep


def refine(contrast_roi: np.ndarray, roi, p: np.ndarray, half: np.ndarray, thr: float):
    """Contrast-weighted centroid of the blob nearest p inside a local window. Returns (pt, area, peak) or None."""
    x0, y0 = roi[0], roi[1]
    cx, cy = p[0] - x0, p[1] - y0
    hx, hy = half
    a0 = int(max(0, np.floor(cx - hx))); a1 = int(min(contrast_roi.shape[1], np.ceil(cx + hx) + 1))
    b0 = int(max(0, np.floor(cy - hy))); b1 = int(min(contrast_roi.shape[0], np.ceil(cy + hy) + 1))
    if a1 - a0 < 3 or b1 - b0 < 3:
        return None
    w = contrast_roi[b0:b1, a0:a1]
    m = w > thr
    if not m.any():
        return None
    lab, n = ndimage.label(m)
    if n > 1:
        yy, xx = np.indices(w.shape)
        best, bd = 0, np.inf
        for k in range(1, n + 1):
            sel = lab == k
            d = np.hypot(xx[sel] - (cx - a0), yy[sel] - (cy - b0)).min()
            if d < bd:
                best, bd = k, d
        m = lab == best
    ww = np.where(m, w - thr, 0.0)
    s = ww.sum()
    if s <= 0:
        return None
    yy, xx = np.indices(w.shape)
    q = np.array([(ww * xx).sum() / s + a0 + x0, (ww * yy).sum() / s + b0 + y0])
    return q, float(m.sum()), float(w[m].max())


def _yellow_ring(bgr, pt, diam):
    """Is the dot at pt sitting on something yellow (the cap)? Median hue/saturation around it."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    r1, r2 = int(0.9 * diam), int(1.9 * diam)
    H, W = bgr.shape[:2]
    a0, a1, b0, b1 = max(0, x - r2), min(W, x + r2 + 1), max(0, y - r2), min(H, y + r2 + 1)
    patch = cv2.cvtColor(bgr[b0:b1, a0:a1], cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[b0:b1, a0:a1]
    rr = np.hypot(xx - x, yy - y)
    ring = (rr >= r1) & (rr <= r2)
    hue, sat = np.median(patch[..., 0][ring]), np.median(patch[..., 1][ring])
    return bool(12 <= hue <= 40 and sat >= 90)


def find_bottle_dots(gray, calib, cfg: Config, face_top_mm: float, near_x_mm: float | None = None, bgr=None):
    """The cap dot and the tape dot on the bottle's front face, or None if the bottle is not in view.

    The pair must be one above the other at about the right spacing; of such pairs the lowest wins
    (the cap dot is the bottle's lowest mark). With the colour frame, the lower dot must also sit on
    the yellow cap and the upper one must not.
    """
    ind = cfg.indenter
    ppm = calib.report["px_per_mm"]
    diam = ind.dot_diameter_mm * ppm
    H, W = gray.shape
    top_mm = calib.px_to_mm(np.array([[W / 2, 0]]))[0, 1]
    xr = (0, 280) if near_x_mm is None else (near_x_mm - 60, near_x_mm + 60)
    # at rest the bottle hangs above the foam, so search only above the top surface
    roi = roi_from_mm(calib, xr, (face_top_mm + 2, min(top_mm, face_top_mm + 160)), gray.shape, pad_px=0)
    pts, area, peak, thr = candidates(gray, roi, diam, frac=0.4)
    if len(pts) < 2:
        return None
    mm = calib.px_to_mm(pts)
    want = ind.tape_dot_mm - ind.cap_dot_mm
    best, bscore = None, np.inf
    for i in range(len(mm)):
        if mm[i, 1] < face_top_mm + 2:
            continue
        for j in range(len(mm)):
            if i == j or mm[j, 1] <= mm[i, 1]:
                continue
            dz = mm[j, 1] - mm[i, 1]
            dx = mm[j, 0] - mm[i, 0]
            if abs(dz - want) > 5 or abs(dx) > 8:
                continue
            if bgr is not None and (not _yellow_ring(bgr, pts[i], diam) or _yellow_ring(bgr, pts[j], diam)):
                continue
            score = abs(dz - want) + abs(dx) + 0.05 * (mm[i, 1] - face_top_mm) - 0.01 * (peak[i] + peak[j])
            if score < bscore:
                best, bscore = (i, j), score
    if best is None:
        return None
    i, j = best
    return {"cap_px": pts[i], "tape_px": pts[j], "cap_mm": mm[i], "tape_mm": mm[j],
            "separation_mm": float(np.linalg.norm(mm[j] - mm[i]))}


def _blueness(bgr):
    f = bgr.astype(np.float32)
    return f[..., 0] - 0.5 * (f[..., 1] + f[..., 2])


def _otsu_1d(v):
    v = v[np.isfinite(v)]
    h, e = np.histogram(v, 64)
    c = 0.5 * (e[1:] + e[:-1])
    w = np.cumsum(h); m = np.cumsum(h * c)
    mt = m[-1]; wt = w[-1]
    between = (mt * w - m * wt) ** 2 / np.maximum(w * (wt - w), 1e-9)
    return c[np.argmax(between[:-1])]


def _crossing(x, v, thr, i):
    """Position where profile v crosses thr between samples i-1 and i (linear interpolation)."""
    if i <= 0 or i >= len(v):
        return float(x[min(max(i, 0), len(x) - 1)])
    a, b = v[i - 1], v[i]
    if a == b:
        return float(x[i])
    f = (thr - a) / (b - a)
    return float(x[i - 1] + np.clip(f, 0, 1) * (x[i] - x[i - 1]))


def _sample(img, calib, xs, zs):
    p = calib.mm_to_px(np.stack([xs, zs], -1)).astype(np.float32)
    return cv2.remap(img, p[..., 0:1].reshape(1, -1), p[..., 1:2].reshape(1, -1), cv2.INTER_LINEAR).ravel()


def foam_outline(bgr, calib, cfg: Config, grid_mm):
    """Measure the block's left and right ends and its top edge (card mm) from the blue of the foam."""
    b = _blueness(bgr)
    ends = []
    x = np.arange(-20, 300, 0.25)
    g = cfg.grid
    for z in g.z0_mm + g.pitch_mm * (np.arange(g.rows - 1) + 0.5):      # midway between dot rows
        v = _sample(b, calib, x, np.full_like(x, z))
        thr = _otsu_1d(v)
        on = ndimage.binary_closing(v > thr, structure=np.ones(int(8 / 0.25)))   # bridge dots and specks
        if not on.any():
            continue
        # longest run of foam pixels along the line; edges by interpolating the threshold crossing
        d = np.diff(np.r_[0, on.astype(int), 0])
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        k = np.argmax(e - s)
        ends.append((_crossing(x, v, thr, s[k]), _crossing(x, v, thr, e[k])))
    out = {}
    col0 = np.nanmean(grid_mm[:, 0, 0])
    if ends:
        L = np.median([a for a, _ in ends]); R = np.median([b_ for _, b_ in ends])
        out.update(x_left_mm=float(L), x_right_mm=float(R), length_mm=float(R - L), ends_from="foam colour")
        if abs((R - L) - cfg.block.length_mm) > 6:
            out["warning"] = f"measured length {R - L:.1f} mm vs {cfg.block.length_mm} mm; using the dot grid instead"
            L = col0 - cfg.grid.x0_mm
            out.update(x_left_mm=float(L), ends_from="dot grid")
    else:
        out.update(x_left_mm=float(col0 - cfg.grid.x0_mm), ends_from="dot grid")
    tops = []
    z = np.arange(30, 110, 0.25)
    for xx in np.arange(out["x_left_mm"] + 6, out["x_left_mm"] + cfg.block.length_mm - 5, 4):
        v = _sample(b, calib, np.full_like(z, xx), z)
        thr = _otsu_1d(v)
        on = v > thr
        if on[0] and not on.all():
            tops.append((float(xx), _crossing(z, v, thr, int(np.argmin(on)))))     # first non-foam sample going up
    if tops:
        tz = np.array([z_ for _, z_ in tops])
        out.update(z_top_mm=float(np.median(tz)), z_top_spread_mm=float(np.std(tz)),
                   top_profile=[[round(x_, 2), round(z_, 3)] for x_, z_ in tops])
    bases = []
    z = np.arange(30, -15, -0.25)
    for xx in np.arange(out["x_left_mm"] + 10, out["x_left_mm"] + cfg.block.length_mm - 10, 10):
        v = _sample(b, calib, np.full_like(z, xx), z)
        thr = _otsu_1d(v)
        on = v > thr
        if on[0] and not on.all():
            bases.append(_crossing(z, v, thr, int(np.argmin(on))))   # first non-foam sample going down
    if bases:
        out.update(z_base_mm=float(np.median(bases)), z_base_spread_mm=float(np.std(bases)))
    return out
