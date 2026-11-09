"""Tracking the dots through a clip with sub-pixel, contrast-weighted centroids.

At 240 fps a dot moves at most a few pixels between frames, so each frame we find every dark blob
once, then hand each tracked dot the blob nearest its predicted position. Blobs that look wrong
(merged, half-hidden, blurred) are marked invalid rather than guessed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import dots
from .config import Config
from .video import Clip


def find_rest(clip: Clip, calib, cfg: Config, dt_median: float, kind: str = "press", need_bottle: bool = False):
    """The still, undeformed stretch to measure from.

    Clips start with the slate (dots hidden) and, for presses, the bottle being brought in, so the
    first still moment is not the right one. Every ~0.1 s the full dot grid is detected; its shape
    (after removing any whole-image shift) is compared with the first complete grid. The press,
    weight or fingertip starts where the shape first changes by more than 0.4 mm; the rest state is
    the last still, undeformed stretch before that (with the bottle dots visible, for presses).
    Static clips use their longest still stretch.
    """
    tun = cfg.tuning
    ppm = calib.report["px_per_mm"]
    step = max(1, int(round(1.0 / (tun.rest_sample_hz * dt_median))))
    min_found = cfg.grid.rows * cfg.grid.cols - 2
    S = []                                   # (frame index, t, px grid or None)
    for i, t, g in clip.frames(step=step):
        px, mm, rep = dots.find_grid(g, calib, cfg)
        S.append((i, t, px if rep["found"] >= min_found else None, rep))
    comp = [k for k, s in enumerate(S) if s[2] is not None]
    if not comp:
        raise RuntimeError(f"{clip.path.name}: the full dot grid was never visible")
    R0 = S[comp[0]][2]

    def shape(a, b):
        d = (a - b).reshape(-1, 2)
        ok = np.isfinite(d[:, 0])
        d = d[ok] - np.median(d[ok], axis=0)
        return np.linalg.norm(d, axis=1)

    dev = {k: float(np.percentile(shape(S[k][2], R0), 90)) / ppm for k in comp}
    deformed = {k: dev[k] > 0.4 for k in comp}
    # A dead weight mostly sinks the whole face a fraction of a millimetre, which this shape check does
    # not see, so a weight clip is one long still stretch and its rest is that stretch's start: in this
    # session 2.6-4.6 s, well before each weight went on (7.3-9.2 s). Counting a uniform sink of more than
    # 0.12 mm as deformed was tried and dropped: slow creep under the pad then leaves no rest at all.
    onset = None
    for a, b in zip(comp[:-1], comp[1:]):
        if deformed[a] and deformed[b]:
            onset = a
            break
    # still runs of complete, undeformed samples
    runs, cur = [], []
    for k in range(len(S)):
        ok = S[k][2] is not None and not deformed.get(k, True)
        if ok and cur and (k - cur[-1] > 1 or np.median(shape(S[k][2], S[cur[-1]][2])) > tun.rest_motion_px):
            runs.append(cur); cur = []
        if ok:
            cur.append(k)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    need = max(2, int(np.ceil(tun.rest_min_s * tun.rest_sample_hz)))
    runs = [r for r in runs if len(r) >= need]
    if onset is not None:
        runs = [r for r in runs if r[-1] < onset]
    if not runs:
        raise RuntimeError(f"{clip.path.name}: no still stretch of {tun.rest_min_s} s with all dots visible "
                           f"{'before the press' if onset is not None else ''}")
    if kind == "static" or onset is None:
        run = max(runs, key=len)
        k_start = run[0]
    else:
        run = runs[-1]
        if need_bottle:
            for r in runs[::-1]:
                _, g_last = clip.frame(S[r[-1]][0])
                if dots.find_bottle_dots(g_last, calib, cfg, face_top_mm=cfg.block.height_mm) is not None:
                    run = r
                    break
        # measure from the last part of the stretch, closest to the press
        t_end = S[run[-1]][1]
        k_start = next(k for k in run if S[k][1] >= t_end - tun.rest_average_s)
    i0, t0 = S[k_start][0], S[k_start][1]
    return {"index": int(i0), "t": float(t0), "end_index": int(S[run[-1]][0]), "end_t": float(S[run[-1]][1]),
            "onset_t": None if onset is None else float(S[onset][1]),
            "grid_px": S[k_start][2], "grid_mm": calib.px_to_mm(S[k_start][2]), "report": S[k_start][3]}


def local_dot(Cimg: np.ndarray, roi, q: np.ndarray, r: float, ref_peak: float, masks=None, mask_r: float = 0.0):
    """The dot nearest q within r px, thresholded at half its own expected contrast.

    For the bottle's dots: their contrast differs from the foam dots' (the cap dot is brown on yellow)
    and changes as the bottle moves, so a threshold set from the foam dots loses them. Discs of radius
    mask_r around `masks` (the foam dots) are ignored. Returns (pt, area_px, peak) or None.
    """
    x0, y0 = roi[0], roi[1]
    a0 = int(max(0, np.floor(q[0] - r - x0))); a1 = int(min(Cimg.shape[1], np.ceil(q[0] + r - x0) + 1))
    b0 = int(max(0, np.floor(q[1] - r - y0))); b1 = int(min(Cimg.shape[0], np.ceil(q[1] + r - y0) + 1))
    if a1 - a0 < 3 or b1 - b0 < 3 or not np.all(np.isfinite(q)):
        return None
    w = Cimg[b0:b1, a0:a1]
    Y, X = np.mgrid[b0:b1, a0:a1]
    X = X + x0; Y = Y + y0
    keep = (X - q[0]) ** 2 + (Y - q[1]) ** 2 <= r * r
    if masks is not None and len(masks):
        m = masks[np.isfinite(masks[:, 0])]
        m = m[np.hypot(m[:, 0] - q[0], m[:, 1] - q[1]) < r + mask_r]
        for c in m:
            keep &= (X - c[0]) ** 2 + (Y - c[1]) ** 2 > mask_r ** 2
    wk = np.where(keep, w, 0.0)
    thr = 0.5 * min(float(wk.max()), ref_peak)
    if thr <= 0:
        return None
    lab, n = ndimage.label(wk > thr)
    best, bd = None, np.inf
    for k in range(1, n + 1):
        sel = lab == k
        pk = float(wk[sel].max())
        if pk < 0.4 * ref_peak:
            continue
        ww = np.where(sel, wk - thr, 0.0)
        s = ww.sum()
        pt = np.array([(ww * X).sum() / s, (ww * Y).sum() / s])
        d = np.hypot(*(pt - q))
        if d < bd:
            best, bd = (pt, float(sel.sum()), pk), d
    return best


def track(clip: Clip, calib, cfg: Config, start: int, init_px: np.ndarray, rest_area: np.ndarray,
          rest_peak: np.ndarray, grid_shape=(7, 15), extra_radius=None, progress=None, on_frame=None):
    """Track N points from frame `start` to the end.

    init_px: (N, 2) starting positions; the first rows*cols are grid dots in row-major order, any
    extra points (the bottle's dots) follow. Grid dots take the nearest dark blob; the bottle's dots
    are found locally with their own contrast, away from the foam dots, and must stay the bottle's
    length apart. Positions are predicted from velocity in px/s, so dropped frames do not throw
    the search off. Returns times (T,), positions (T, N, 2) with NaN where not valid, valid (T, N),
    area (T, N).
    """
    tun = cfg.tuning
    ppm = calib.report["px_per_mm"]
    diam = cfg.grid.dot_diameter_mm * ppm
    N = len(init_px)
    R, C = grid_shape
    G = R * C
    n_extra = N - G
    thr = tun.contrast_fraction * float(np.nanmedian(rest_peak[:G]))

    pts_all = init_px[np.isfinite(init_px[:, 0])]
    times, P, V, A = [], [], [], []
    p = init_px.copy()
    rest_px = init_px.copy()
    vel = np.zeros_like(p)                      # px per second
    lost = np.zeros(N, bool)
    miss = np.zeros(N, int)
    roi = None
    t_prev = None
    sep0 = float(np.linalg.norm(rest_px[G + 1] - rest_px[G])) if n_extra == 2 else None
    sep_last = sep0
    axis_last = (rest_px[G + 1] - rest_px[G]) / sep0 if n_extra == 2 else None
    half_area = np.full(N, np.nan)              # bottle dots: area at half their own contrast, at rest
    for idx, t, g in clip.frames(start=0):
        if roi is None:
            H, W = g.shape
            pad = int(4 * diam)
            x0 = max(0, int(pts_all[:, 0].min()) - pad); x1 = min(W, int(pts_all[:, 0].max()) + pad)
            y0 = 0 if n_extra else max(0, int(pts_all[:, 1].min()) - pad)
            y1 = min(H, int(pts_all[:, 1].max()) + pad)
            roi = (x0, y0, x1, y1)
        times.append(t)
        if idx < start:
            P.append(np.full((N, 2), np.nan)); V.append(np.zeros(N, bool)); A.append(np.full(N, np.nan))
            continue
        dt = 0.0 if t_prev is None else t - t_prev
        t_prev = t
        Cimg = dots.contrast_image(g, roi, diam)
        comp, area, peak = dots.blobs(Cimg, thr, diam, offset=roi[:2], min_area_ratio=0.05, max_area_ratio=6.0)
        q = p + vel * dt
        if lost[:G].any():
            # a lost grid dot is expected where its neighbours say it is: its rest position moved like them
            disp = (p[:G] - rest_px[:G]).reshape(R, C, 2)
            okg = (~lost[:G]).reshape(R, C)
            for i in np.where(lost[:G])[0]:
                r, c = divmod(i, C)
                r0, r1, c0, c1 = max(0, r - 2), min(R, r + 3), max(0, c - 2), min(C, c + 3)
                nb = disp[r0:r1, c0:c1][okg[r0:r1, c0:c1]]
                if len(nb):
                    q[i] = rest_px[i] + np.median(nb, axis=0)
        newp = q.copy()
        ok = np.zeros(N, bool)
        ar = np.full(N, np.nan)
        claimed = set()
        tree = cKDTree(comp) if len(comp) else None

        # the bottle's dots first, so a foam dot can never take their blob
        if n_extra:
            for e in range(G, N):
                if not np.isfinite(rest_peak[e]):
                    continue
                r_e = min(60.0, 8.0 + 0.5 * np.hypot(*vel[e]) * dt + 6.0 * miss[e])
                if extra_radius:
                    r_e = min(r_e, float(extra_radius))
                # never take the other bottle dot: mask it where it is (found) or is expected to be
                other = [newp[f] if ok[f] else q[f] for f in range(G, N) if f != e and np.isfinite(q[f, 0])]
                masks = np.vstack([q[:G]] + [np.atleast_2d(o) for o in other])
                f = local_dot(Cimg, roi, q[e], r_e, float(rest_peak[e]), masks, 0.75 * diam)
                if f is None:
                    continue
                pt, a_, pk = f
                if idx == start:
                    half_area[e] = a_
                ratio = a_ / half_area[e] if np.isfinite(half_area[e]) else 1.0
                # the cap dot stays below the tape dot (image y grows downwards)
                if n_extra == 2 and sep0:
                    if e == G and miss[G + 1] < 40 and np.isfinite(q[G + 1, 1]) and pt[1] < q[G + 1, 1] + 0.5 * sep0:
                        continue
                    if e == G + 1 and ok[G] and pt[1] > newp[G, 1] - 0.5 * sep0:
                        continue
                if 0.25 <= ratio <= 4.0:
                    newp[e] = pt; ok[e] = True; ar[e] = a_
            if n_extra == 2:
                # rigid bottle: the two dots stay (almost) the same distance apart and the axis turns slowly
                if ok[G] and ok[G + 1]:
                    d = newp[G + 1] - newp[G]
                    sep = float(np.linalg.norm(d))
                    turn = np.degrees(np.arccos(np.clip(np.dot(d / sep, axis_last), -1, 1)))
                    if abs(sep - sep_last) > 3.0 + 0.03 * sep0 or turn > 4.0:
                        # keep the dot that landed nearer its prediction
                        e_bad = G if np.linalg.norm(newp[G] - q[G]) > np.linalg.norm(newp[G + 1] - q[G + 1]) else G + 1
                        ok[e_bad] = False; newp[e_bad] = q[e_bad]
                    else:
                        sep_last, axis_last = sep, d / sep
                off = axis_last * sep_last
                if ok[G] and not ok[G + 1]:
                    newp[G + 1] = newp[G] + off
                elif ok[G + 1] and not ok[G]:
                    newp[G] = newp[G + 1] - off
            if tree is not None:
                for e in range(G, N):
                    if ok[e]:
                        claimed.update(tree.query_ball_point(newp[e], 0.75 * diam))

        # rest of the grid: nearest blob within just under half the distance to the nearest neighbour
        rad = np.empty(G)
        gp = q[:G].reshape(R, C, 2)
        dn = np.full((R, C), np.inf)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.full_like(gp, np.nan)
            rs = slice(max(dr, 0), R + min(dr, 0)); rd = slice(max(-dr, 0), R + min(-dr, 0))
            cs = slice(max(dc, 0), C + min(dc, 0)); cd = slice(max(-dc, 0), C + min(-dc, 0))
            sh[rd, cd] = gp[rs, cs]
            dd = np.linalg.norm(sh - gp, axis=-1)
            dn = np.fmin(dn, np.where(np.isfinite(dd), dd, np.inf))
        rad[:] = np.clip(0.45 * dn.ravel(), 2.0, 1.2 * diam)
        if tree is not None:
            fin = np.isfinite(q[:G, 0])
            d = np.full(G, np.inf); j = np.zeros(G, int)
            if fin.any():
                d[fin], j[fin] = tree.query(q[:G][fin], k=1, distance_upper_bound=float(rad[fin].max()))
            claim = {}
            for i in range(G):
                if not np.isfinite(d[i]) or d[i] > rad[i]:
                    continue
                jj = int(j[i])
                if jj in claimed or (jj in claim and d[claim[jj]] <= d[i]):
                    continue
                claim[jj] = i
            for jj, i in claim.items():
                ratio = area[jj] / rest_area[i] if np.isfinite(rest_area[i]) and rest_area[i] > 0 else 1.0
                bright_ok = peak[jj] >= 0.4 * rest_peak[i] if np.isfinite(rest_peak[i]) else True
                if tun.min_area_ratio <= ratio <= tun.max_area_ratio and bright_ok:
                    newp[i] = comp[jj]; ok[i] = True; ar[i] = area[jj]; claimed.add(jj)
        # local fallback for grid dots that lost their blob (e.g. merged with a neighbour)
        for i in np.where(~ok[:G] & np.isfinite(q[:G, 0]))[0]:
            r = dots.refine(Cimg, roi, q[i], np.array([rad[i], rad[i]]), thr)
            if r is not None:
                pt, a_, pk = r
                ratio = a_ / rest_area[i] if rest_area[i] > 0 else 1.0
                near_bottle = n_extra and any(ok[e] and np.linalg.norm(pt - newp[e]) < 0.75 * diam for e in range(G, N))
                if tun.min_area_ratio <= ratio <= tun.max_area_ratio and pk >= 0.4 * rest_peak[i] and not near_bottle:
                    newp[i] = pt; ok[i] = True; ar[i] = a_
        if dt > 0:
            new_vel = np.where(ok[:, None], 0.6 * (newp - p) / dt + 0.4 * vel, 0.0)
            if n_extra:
                # a bottle dot lost for a moment keeps moving as it was (fast presses, hands in the way)
                new_vel[G:] = np.where(ok[G:, None], new_vel[G:], 0.9 * vel[G:])
            vel = new_vel
        lost = ~ok
        miss = np.where(ok, 0, miss + 1)
        p = newp
        P.append(np.where(ok[:, None], p, np.nan)); V.append(ok); A.append(ar)
        if on_frame is not None:
            on_frame(idx, t, g)
        if progress is not None:
            progress(idx)
    return np.asarray(times), np.stack(P), np.stack(V), np.stack(A)
