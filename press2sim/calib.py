"""Pixel-to-millimetre calibration from a checkerboard clip (C1, C2).

The card stands on its bottom edge, flat against the dotted face, so its printed plane is the face
plane. A homography maps image pixels to card millimetres: x along the face, z = height above the
table. Corners are averaged over every frame in which the board sits still.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .video import Clip


def board_visible(gray: np.ndarray, cfg: Config) -> bool:
    """Fast check on a half-size image, so the slow accurate search only runs where a board is."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ok, _ = cv2.findChessboardCorners(small, (cfg.board.inner_cols, cfg.board.inner_rows),
                                      flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH |
                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
    return bool(ok)


def detect_corners(gray: np.ndarray, cfg: Config):
    """Inner corners as (rows, cols, 2) pixels, row 0 lowest in the image, col 0 leftmost; or None."""
    pattern = (cfg.board.inner_cols, cfg.board.inner_rows)
    flags = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if not ok:
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags | cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return None
    c = corners.reshape(cfg.board.inner_rows, cfg.board.inner_cols, 2).astype(np.float64)
    if c[0, -1, 0] < c[0, 0, 0]:      # columns must run left to right
        c = c[:, ::-1]
    if c[-1, 0, 1] > c[0, 0, 1]:      # rows must run upward (image y decreasing)
        c = c[::-1]
    return c


def _fit(px: np.ndarray, mm: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(px.reshape(-1, 1, 2), mm.reshape(-1, 1, 2), 0)
    return H


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (..., 2) points through a homography; NaNs pass through."""
    p = np.asarray(pts, dtype=np.float64)
    shp = p.shape
    p = p.reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    out = q[:, :2] / q[:, 2:3]
    return out.reshape(shp)


@dataclass
class Calibration:
    H: np.ndarray                 # pixels -> card mm (x right, z up from the table)
    H_inv: np.ndarray             # card mm -> pixels
    corners_px: np.ndarray        # (rows, cols, 2) averaged corners
    corners_mm: np.ndarray        # (rows, cols, 2) known positions
    report: dict

    def px_to_mm(self, pts):
        m = apply_h(self.H, pts)
        f = self.report.get("face")
        if f:
            m = m.copy()
            m[..., 1] = f["a_z"] * (m[..., 1] - f["z0"])
        return m

    def mm_to_px(self, pts):
        f = self.report.get("face")
        if f:
            pts = np.array(pts, dtype=np.float64, copy=True)
            pts[..., 1] = pts[..., 1] / f["a_z"] + f["z0"]
        return apply_h(self.H_inv, pts)

    def card_mm(self, pts):
        """Card millimetres, without the face-plane correction (for the card's own corners)."""
        return apply_h(self.H, pts)

    def raw(self) -> "Calibration":
        """The same calibration without the face-plane correction."""
        rep = {k: v for k, v in self.report.items() if k != "face"}
        return Calibration(self.H, self.H_inv, self.corners_px, self.corners_mm, rep)

    def save(self, path: Path):
        d = dict(self.report)
        d.update(H=self.H.tolist(), H_inv=self.H_inv.tolist(), corners_px=self.corners_px.tolist(),
                 corners_mm=self.corners_mm.tolist())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=1))

    @staticmethod
    def load(path: Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return Calibration(np.array(d.pop("H")), np.array(d.pop("H_inv")), np.array(d.pop("corners_px")),
                           np.array(d.pop("corners_mm")), d)


def calibrate_clip(path: str | Path, cfg: Config, rotation_ccw: int | None = None,
                   sample_hz: float = 12.0) -> Calibration:
    """Find the board in a calibration clip, average the still frames, fit and check the homography."""
    clip = Clip(path, rotation_ccw)
    from .video import frame_times
    info = frame_times(path)
    step = max(1, int(round(1.0 / (sample_hz * info.dt_median))))
    found = []   # (index, t, corners)
    for i, t, g in clip.frames(step=step):
        if not board_visible(g, cfg):
            continue
        c = detect_corners(g, cfg)
        if c is not None:
            found.append((i, t, c))
    if len(found) < 3:
        raise RuntimeError(f"{Path(path).name}: checkerboard found in only {len(found)} sampled frames. "
                           "Check the whole board is in view, flat, and not cut off.")
    # Longest run of consecutive detections where the board did not move (median corner shift < 0.3 px).
    runs, cur = [], [found[0]]
    for a, b in zip(found[:-1], found[1:]):
        consecutive = b[0] - a[0] <= step
        still = np.median(np.linalg.norm(b[2] - a[2], axis=-1)) < 0.3
        if consecutive and still:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    run = max(runs, key=len)
    stack = np.stack([r[2] for r in run])               # (n, rows, cols, 2)
    corners = stack.mean(axis=0)
    jitter = float(np.median(stack.std(axis=0))) if len(run) > 1 else float("nan")

    mm = cfg.board.corner_xz_mm()
    H = _fit(corners, mm)
    resid = np.linalg.norm(apply_h(H, corners) - mm, axis=-1)
    # Hold-out check: fit on a checkerboard half of the corners, test on the other half.
    rr, cc = np.indices(mm.shape[:2])
    a = ((rr + cc) % 2 == 0)
    Ha = _fit(corners[a], mm[a])
    held = np.linalg.norm(apply_h(Ha, corners[~a]) - mm[~a], axis=-1)
    m = apply_h(H, corners)
    sq_x = np.linalg.norm(np.diff(m, axis=1), axis=-1).mean()
    sq_z = np.linalg.norm(np.diff(m, axis=0), axis=-1).mean()
    px_per_mm = float(np.linalg.norm(np.diff(corners, axis=1), axis=-1).mean() / cfg.board.square_mm)
    report = {
        "clip": Path(path).name, "rotation_ccw": clip.rotation_ccw,
        "image_size": [int(info.width), int(info.height)],
        "frames_with_board": len(found), "frames_averaged": len(run),
        "t_used_s": [float(run[0][1]), float(run[-1][1])],
        "corner_jitter_px": jitter,
        "fit_rms_mm": float(np.sqrt(np.mean(resid ** 2))), "fit_max_mm": float(resid.max()),
        "holdout_rms_mm": float(np.sqrt(np.mean(held ** 2))), "holdout_max_mm": float(held.max()),
        "square_mm_measured_x": float(sq_x), "square_mm_measured_z": float(sq_z),
        "px_per_mm": px_per_mm,
    }
    return Calibration(H, np.linalg.inv(H), corners, mm, report)


def _similarity(src: np.ndarray, dst: np.ndarray):
    """Least-squares 2D similarity dst ~ s R src + t (Umeyama)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    t = md - s * R @ ms
    return s, R, t


def compare(c1: Calibration, c2: Calibration) -> dict:
    """Did the camera or the foam move between two calibration clips?

    The card is placed by hand each time, so a sideways shift along x is expected and harmless.
    Both cards stand on the table, so a vertical shift, a tilt, or a scale change means the camera
    or the face moved; residuals after the best similarity fit mean the viewing angle changed.
    """
    p = c1.card_mm(c2.corners_px).reshape(-1, 2)     # where C2's corners sit in C1's card millimetres
    q = c2.corners_mm.reshape(-1, 2)
    s, R, t = _similarity(q, p)
    res = np.linalg.norm((s * (R @ q.T)).T + t - p, axis=1)
    d_px = np.linalg.norm(c2.corners_px - c1.corners_px, axis=-1)
    return {"a": c1.report["clip"], "b": c2.report["clip"],
            "card_shift_x_mm": float(t[0]), "card_shift_z_mm": float(t[1]),
            "tilt_change_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            "scale_change_pct": float(100 * (s - 1)),
            "residual_rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "corner_shift_px_median": float(np.median(d_px)),
            "verdict": ("unchanged" if abs(t[1]) < 0.3 and abs(100 * (s - 1)) < 0.3 and
                        np.sqrt(np.mean(res ** 2)) < 0.15 else "check: camera or foam may have moved")}


def face_correction(calib: Calibration, cfg: Config, clip_path, dt_median: float,
                    row_span_mm: float | None = None) -> dict:
    """Heights on the dotted face, corrected for the card not lying exactly in the face's plane.

    Measured on a still stretch of the bare face (the calibration clip has one before the card goes
    in). Two rulers on the foam itself: the block's height (base to top edge, found from the foam's
    colour) and the distance between the bottom and top rows of dots measured with a ruler. Heights
    become a_z * (z_card - z0), so the foam's base sits at z = 0. Widths are only checked, against
    the block's length.
    """
    from . import dots
    from .tracker import find_rest
    from .video import Clip
    raw = calib.raw()
    clip = Clip(clip_path, raw.report.get("rotation_ccw"))
    rest = find_rest(clip, raw, cfg, dt_median, kind="static")
    _, bgr = clip.frame(rest["index"], color=True)
    out = dots.foam_outline(bgr, raw, cfg, rest["grid_mm"])
    g = rest["grid_mm"]
    rows_meas = float(np.nanmean(g[-1, :, 1]) - np.nanmean(g[0, :, 1]))
    refs = {}
    if "z_top_mm" in out and "z_base_mm" in out:
        refs["block_height"] = cfg.block.height_mm / (out["z_top_mm"] - out["z_base_mm"])
    if row_span_mm:
        refs["dot_rows"] = row_span_mm / rows_meas
    if not refs or "z_base_mm" not in out:
        raise RuntimeError(f"{Path(clip_path).name}: could not measure the foam's base and top edges")
    vals = np.array(list(refs.values()))
    a = float(vals.mean())
    return {"a_z": a, "z0": float(out["z_base_mm"]), "references": {k: float(v) for k, v in refs.items()},
            "spread_pct": float(100 * (vals.max() - vals.min()) / a) if len(vals) > 1 else None,
            "from_clip": Path(clip_path).name, "rest_t_s": [rest["t"], rest["end_t"]],
            "measured_card_mm": {"z_top": out.get("z_top_mm"), "z_top_spread": out.get("z_top_spread_mm"),
                                 "z_base": out.get("z_base_mm"), "z_base_spread": out.get("z_base_spread_mm"),
                                 "length": out.get("length_mm"), "dot_rows_span": rows_meas,
                                 "dot_pitch_x": float(np.nanmean(np.diff(np.nanmean(g[..., 0], axis=0)))),
                                 "dot_pitch_z": float(np.nanmean(np.diff(np.nanmean(g[..., 1], axis=1))))},
            "length_check_pct": (100 * (out["length_mm"] / cfg.block.length_mm - 1)) if "length_mm" in out else None}
