"""One clip in, one tracks file out: rest state, dots and bottle tracked, millimetres, events, checks."""
from __future__ import annotations

import json
import warnings
import time
from pathlib import Path

import numpy as np

from . import dots
from .calib import Calibration
from .config import Config
from .indenter import cap_path, events
from .stability import BackgroundCheck
from .tracker import find_rest, track
from .video import Clip, frame_times

KINDS = ("calib", "static", "press", "fingertip", "weight")


def _appearance(g, calib, cfg, pts_px, roi):
    """Rest-frame area and peak contrast of the blob under each point (for validity checks)."""
    ppm = calib.report["px_per_mm"]
    diam = cfg.grid.dot_diameter_mm * ppm
    C = dots.contrast_image(g, roi, diam)
    hi = np.percentile(C, 99.7)
    comp, area, peak = dots.blobs(C, max(12.0, 0.4 * hi), diam, offset=roi[:2])
    G = cfg.grid.rows * cfg.grid.cols
    area0 = np.full(len(pts_px), np.nan); peak0 = np.full(len(pts_px), np.nan)
    for i, p in enumerate(pts_px):
        if not np.isfinite(p[0]) or not len(comp):
            continue
        d = np.linalg.norm(comp - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < 0.6 * diam:
            area0[i], peak0[i] = area[j], peak[j]
    thr = cfg.tuning.contrast_fraction * float(np.nanmedian(peak0[:G]))
    comp, area, peak = dots.blobs(C, thr, diam, offset=roi[:2])
    for i, p in enumerate(pts_px):
        if not np.isfinite(p[0]) or not len(comp):
            continue
        d = np.linalg.norm(comp - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < 0.6 * diam:
            area0[i] = area[j]
    return area0, peak0


def _fill_missing(grid_px: np.ndarray, g0, calib, cfg):
    """Dots not found in the rest frame: placed from their neighbours, then looked for there."""
    g = grid_px.copy()
    miss = ~np.isfinite(g[..., 0])
    if not miss.any():
        return g
    R, C = miss.shape
    ppm = calib.report["px_per_mm"]
    diam = cfg.grid.dot_diameter_mm * ppm
    roi = (0, 0, g0.shape[1], g0.shape[0])
    for r, c in zip(*np.where(miss)):
        guess = []
        for (a, b), (e, f) in ((((r, c - 1), (r, c + 1))), (((r - 1, c), (r + 1, c)))):
            if 0 <= a < R and 0 <= b < C and 0 <= e < R and 0 <= f < C and np.isfinite(g[a, b, 0]) and np.isfinite(g[e, f, 0]):
                guess.append(0.5 * (g[a, b] + g[e, f]))
        for (a, b), (e, f) in ((((r, c - 1), (r, c - 2))), (((r, c + 1), (r, c + 2))), (((r - 1, c), (r - 2, c))),
                               (((r + 1, c), (r + 2, c)))):
            if 0 <= a < R and 0 <= b < C and 0 <= e < R and 0 <= f < C and np.isfinite(g[a, b, 0]) and np.isfinite(g[e, f, 0]):
                guess.append(2 * g[a, b] - g[e, f])
        if not guess:
            continue
        q = np.mean(guess, axis=0)
        x0, y0 = int(max(0, q[0] - 3 * diam)), int(max(0, q[1] - 3 * diam))
        sub_roi = (x0, y0, int(min(g0.shape[1], q[0] + 3 * diam)), int(min(g0.shape[0], q[1] + 3 * diam)))
        Cimg = dots.contrast_image(g0, sub_roi, diam)
        hit = dots.refine(Cimg, sub_roi, q, np.array([0.35, 0.35]) * cfg.grid.pitch_mm * ppm, 0.5 * Cimg.max())
        g[r, c] = hit[0] if hit is not None else q
    return g


def _camera_summary(res, t_last, tail_s=0.6):
    """Camera motion relative to the rest frame; the last moments (tap to stop recording) reported apart."""
    if "t" not in res:
        return res
    tt, mag = np.array(res["t"]), np.hypot(res["dx"], res["dy"])
    body = tt <= t_last - tail_s
    out = {"features": res["features"],
           "max_shift_px": float(mag[body].max()) if body.any() else float("nan"),
           "median_shift_px": float(np.median(mag[body])) if body.any() else float("nan"),
           "max_shift_px_last_0.6s": float(mag[~body].max()) if (~body).any() else 0.0}
    return out


def _foam_top_under(outline: dict, x_card: float, cfg: Config):
    """Height of the undisturbed top surface under the cap (mm), from the top edge on either side of it.

    The block's top is not perfectly level (it varies by a couple of millimetres along its length), and
    at rest the bottle may already sit on it, so the edge 32-60 mm either side of the cap's centre is
    fitted with a line and read off under the cap. Also returns how far the foam under the cap sits
    below that line at rest (the bottle's own weight), or None.
    """
    prof = np.array(outline.get("top_profile", []), dtype=float).reshape(-1, 2)
    fallback = outline.get("z_top_mm")
    if fallback is None or abs(fallback - cfg.block.height_mm) > 4:
        fallback = cfg.block.height_mm
    if not np.isfinite(x_card) or len(prof) < 4:
        return float(fallback), "median top edge", None
    dx = np.abs(prof[:, 0] - x_card)
    flank = (dx >= 32) & (dx <= 60)
    left, right = flank & (prof[:, 0] < x_card), flank & (prof[:, 0] > x_card)
    if left.sum() >= 2 and right.sum() >= 2:
        a, b = np.polyfit(prof[flank, 0], prof[flank, 1], 1)
        z = float(a * x_card + b)
        note = "top edge either side of the cap"
    elif flank.sum() >= 2:
        z = float(np.median(prof[flank, 1]))
        note = "top edge beside the cap (one side)"
    else:
        return float(fallback), "median top edge", None
    under = dx <= 15
    preload = float(np.median(prof[under, 1]) - z) if under.any() else None
    if abs(z - cfg.block.height_mm) > 4:
        return float(fallback), "median top edge (local edge implausible)", preload
    return z, note, preload


def process_clip(path: str | Path, code: str, kind: str, calib: Calibration, cfg: Config, out_dir: Path,
                 rotation_ccw: int | None = None, make_plots: bool = True, log=print) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices before the rest frame are expected
        return _process_clip(path, code, kind, calib, cfg, out_dir, rotation_ccw, make_plots, log)


def _process_clip(path, code, kind, calib, cfg, out_dir, rotation_ccw, make_plots, log):
    path = Path(path)
    t_start = time.time()
    out_dir = Path(out_dir)
    (out_dir / "tracks").mkdir(parents=True, exist_ok=True)
    info = frame_times(path)
    rot = calib.report.get("rotation_ccw") if rotation_ccw is None else rotation_ccw
    clip = Clip(path, rot)
    R, C = cfg.grid.rows, cfg.grid.cols
    G = R * C

    rest = find_rest(clip, calib, cfg, info.dt_median, kind=kind, need_bottle=(kind == "press"))
    k0 = rest["index"]
    _, g0 = clip.frame(k0)
    _, bgr0 = clip.frame(k0, color=True)
    outline = dots.foam_outline(bgr0, calib, cfg, rest["grid_mm"])
    x_left = outline["x_left_mm"]
    grid_px = _fill_missing(rest["grid_px"], g0, calib, cfg).reshape(-1, 2)

    pts = grid_px
    bottle = None
    if kind == "press":
        press_x = x_left + cfg.block.length_mm / 2
        bottle = dots.find_bottle_dots(g0, calib, cfg, face_top_mm=cfg.block.height_mm, bgr=bgr0)
        if bottle is None:
            log(f"  {code}: bottle dots not found at rest; tracking the foam only")
        else:
            pts = np.vstack([grid_px, bottle["cap_px"], bottle["tape_px"]])
    face_mm = np.array([[x_left, 0], [x_left + cfg.block.length_mm, cfg.block.height_mm]])
    fpx = calib.mm_to_px(np.array([[face_mm[0, 0], face_mm[0, 1]], [face_mm[1, 0], face_mm[1, 1]]]))
    face_box = (int(fpx[:, 0].min()), int(fpx[:, 1].min()), int(fpx[:, 0].max()), int(fpx[:, 1].max()))
    pad = int(3 * cfg.grid.dot_diameter_mm * calib.report["px_per_mm"])
    H, W = g0.shape
    roi = (max(0, face_box[0] - pad), 0 if bottle else max(0, face_box[1] - pad),
           min(W, face_box[2] + pad), min(H, face_box[3] + pad))
    area0, peak0 = _appearance(g0, calib, cfg, pts, roi)

    bg = BackgroundCheck(face_box, cfg.tuning.bg_features)
    t, P, V, A = track(clip, calib, cfg, k0, pts, area0, peak0, grid_shape=(R, C), on_frame=bg)

    mm = calib.px_to_mm(P)
    mm[..., 0] -= x_left                                   # x from the block's left end
    t0 = t[k0]
    sel = (t >= t0) & (t <= min(rest["end_t"], t0 + cfg.tuning.rest_average_s))
    rest_mm = np.nanmean(mm[sel, :G], axis=0)
    rest_std = np.nanstd(mm[sel, :G], axis=0)
    disp = mm[:, :G] - rest_mm

    meta = {
        "code": code, "kind": kind, "file": path.name, "calibration": calib.report.get("clip"),
        "time": info.summary(),
        "rest": {"index": k0, "t": float(t0), "still_until_t": rest["end_t"], "onset_t": rest.get("onset_t"),
                 "frames_averaged": int(sel.sum()),
                 "noise_rms_mm": float(np.sqrt(np.nanmean(rest_std ** 2))) if sel.sum() > 2 else None},
        "grid": {k: v for k, v in rest["report"].items() if k not in ("row_z", "col_x")},
        "outline": outline,
        "valid_fraction_after_rest": float(V[k0:, :G].mean()),
        "never_valid_dots": int((~V[k0:, :G].any(axis=0)).sum()),
        "camera_still": _camera_summary(bg.result(), t[-1]),
        "processing_s": None,
    }
    arrays = {"t": t, "pts_px": P.astype(np.float32), "valid": V, "pts_mm": mm.astype(np.float32),
              "rest_mm": rest_mm, "disp_mm": disp.astype(np.float32), "rows": R, "cols": C, "k0": k0}
    bres = bg.result()
    if "t" in bres:
        arrays.update(bg_t=np.array(bres["t"]), bg_dx=np.array(bres["dx"]), bg_dy=np.array(bres["dy"]))

    if bottle is not None:
        cp = cap_path(t, mm[:, G], mm[:, G + 1], cfg, rest_sel=sel)
        cx_rest = np.nanmedian(cp["x_mm"][sel]) if sel.any() else np.nan
        col = int(np.nanargmin(np.abs(rest_mm[(R - 1) * C:(R) * C, 0] - cx_rest))) if np.isfinite(cx_rest) else C // 2
        top_dot_z = mm[:, (R - 1) * C + col, 1]
        z_top, top_note, preload = _foam_top_under(outline, cx_rest + x_left, cfg)
        ev = events(t, cp["z_mm"], z_top, cfg, foam_top_dot_z=top_dot_z)
        ev["foam_top_used_mm"] = float(z_top)
        ev["foam_top_from"] = top_note
        sep = cp["dot_separation_mm"]
        src = cp["source"][k0:]
        meta["indenter"] = {"rest_x_mm": float(cx_rest), "rest_height_above_foam_mm": float(np.nanmedian(cp["z_mm"][sel]) - z_top),
                            "foam_under_cap_at_rest_mm": preload,
                            "dot_separation_rest_mm": cp["separation_rest_mm"],
                            "dot_separation_expected_mm": cfg.indenter.tape_dot_mm - cfg.indenter.cap_dot_mm,
                            "dot_separation_range_mm": [float(np.nanmin(sep)), float(np.nanmax(sep))],
                            "bottle_scale": float(cp["bottle_scale"]),
                            "tilt_deg_max": float(np.nanmax(np.abs(cp["tilt_deg"]))),
                            "valid_fraction": float(np.mean(np.isfinite(cp["z_mm"][k0:]))),
                            "both_dots_fraction": float(np.mean(src == 2)),
                            "cap_dot_fraction": float(np.mean((src == 1) | (src == 2))),
                            "column_under_cap": col}
        meta["events"] = ev
        arrays.update(cap_x_mm=cp["x_mm"], cap_z_mm=cp["z_mm"], tilt_deg=cp["tilt_deg"], cap_source=cp["source"],
                      foam_top_mm=np.float64(z_top))
    meta["processing_s"] = round(time.time() - t_start, 1)

    np.savez_compressed(out_dir / "tracks" / f"{code}.npz", **arrays)
    (out_dir / "tracks" / f"{code}.json").write_text(json.dumps(meta, indent=1, default=float))
    if make_plots:
        from . import qa
        qa.clip_plots(code, clip, calib, cfg, arrays, meta, out_dir / "qa")
    return meta
