"""Plots and cross-clip checks: what to look at before trusting any fit."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def calib_plot(calib, clip_path, out_png: Path, rotation_ccw=None):
    from .video import Clip
    clip = Clip(clip_path, calib.report.get("rotation_ccw") if rotation_ccw is None else rotation_ccw)
    i = None
    t0, t1 = calib.report["t_used_s"]
    for idx, t, img in clip.frames(color=True):
        if t >= 0.5 * (t0 + t1):
            i = img
            break
    if i is None:
        return
    res = calib.card_mm(calib.corners_px) - calib.corners_mm
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.imshow(i[..., ::-1])
    c = calib.corners_px.reshape(-1, 2)
    r = res.reshape(-1, 2)
    ax.plot(c[:, 0], c[:, 1], "+", color="lime", ms=6)
    ax.quiver(c[:, 0], c[:, 1], r[:, 0], -r[:, 1], color="red", angles="xy", scale_units="xy", scale=0.01)
    rep = calib.report
    ax.set_title(f"{rep['clip']}: fit RMS {rep['fit_rms_mm']:.3f} mm, hold-out RMS {rep['holdout_rms_mm']:.3f} mm, "
                 f"{rep['px_per_mm']:.2f} px/mm (red arrows = residuals x100)")
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def clip_plots(code, clip, calib, cfg, A, meta, qa_dir: Path):
    qa_dir.mkdir(parents=True, exist_ok=True)
    t, V, mm, disp = A["t"], A["valid"], A["pts_mm"], A["disp_mm"]
    R, C = int(A["rows"]), int(A["cols"])
    G = R * C
    k0 = int(A["k0"])
    has_cap = "cap_z_mm" in A
    ev = meta.get("events", {})

    # 1. time series
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    top = disp[:, (R - 1) * C:R * C, 1]
    cmap = plt.get_cmap("viridis")
    for c in range(C):
        axs[0].plot(t, top[:, c], color=cmap(c / (C - 1)), lw=0.8)
    axs[0].set_ylabel("top-row dots, vertical\nmovement (mm)")
    if has_cap:
        z_top = float(A["foam_top_mm"]) if "foam_top_mm" in A else cfg.block.height_mm
        depth = z_top - A["cap_z_mm"]
        src = A["cap_source"] if "cap_source" in A else np.full(len(t), 2)
        axs[1].plot(t, depth, color="tab:orange", lw=1.2, label="cap depth below the undisturbed top")
        guess = np.where(src == 3, depth, np.nan)
        if np.isfinite(guess).any():
            axs[1].plot(t, guess, ".", ms=2, color="tab:red", label="cap dot hidden: from the tape dot")
        axs[1].set_ylabel("cap depth (mm)")
        if np.isfinite(depth).any():
            axs[1].set_ylim(-5, 1.12 * max(5.0, float(np.nanmax(depth))))
        y0, y1 = axs[1].get_ylim()
        for n, (k, col) in enumerate((("t_contact", "g"), ("t_max", "k"), ("t_lift", "m"), ("t_release", "r"))):
            if k in ev:
                for ax in axs:
                    ax.axvline(ev[k], color=col, ls="--", lw=0.8)
                axs[1].text(ev[k], y1 - (0.06 + 0.11 * n) * (y1 - y0), " " + k[2:], color=col, fontsize=8, va="top")
        axs[1].legend(loc="upper right")
    else:
        axs[1].plot(t, np.nanmean(disp[:, :, 1], axis=1), color="tab:blue")
        axs[1].set_ylabel("mean vertical movement\nof all dots (mm)")
    dt = np.diff(t) * 1e3
    axs[2].plot(t[1:], dt, ".", ms=2, color="gray", label="gap between frames (ms)")
    ax2 = axs[2].twinx()
    ax2.plot(t, V[:, :G].mean(axis=1), color="tab:green", lw=1, label="fraction of dots tracked")
    ax2.set_ylim(0, 1.05)
    axs[2].set_ylabel("frame gap (ms)")
    ax2.set_ylabel("dots tracked", color="tab:green")
    axs[2].set_xlabel("time in clip (s)")
    axs[0].axvline(t[k0], color="gray", lw=0.8)
    fig.suptitle(f"{code} ({meta['kind']}): rest from {t[k0]:.2f} s, "
                 f"max depth {ev.get('max_depth_mm', float('nan')):.1f} mm")
    fig.tight_layout()
    fig.savefig(qa_dir / f"{code}_timeseries.png", dpi=100)
    plt.close(fig)

    # 2. displacement field at the moment of largest motion
    mag = np.nanmax(np.linalg.norm(disp, axis=-1), axis=1)
    if has_cap and "t_max" in ev:
        km = int(np.argmin(np.abs(t - ev["t_max"])))
    else:
        km = int(np.nanargmax(np.where(np.isfinite(mag), mag, -1)))
    rest = A["rest_mm"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.add_patch(plt.Rectangle((0, 0), cfg.block.length_mm, cfg.block.height_mm, fill=False, ec="0.5"))
    d = disp[km]
    ok = np.isfinite(d[:, 0])
    ax.quiver(rest[ok, 0], rest[ok, 1], d[ok, 0], d[ok, 1], angles="xy", scale_units="xy", scale=1, width=0.003,
              color="tab:blue")
    ax.plot(rest[:, 0], rest[:, 1], ".", color="0.6", ms=3)
    if has_cap and np.isfinite(A["cap_z_mm"][km]):
        cx, cz = A["cap_x_mm"][km], A["cap_z_mm"][km]
        w, h = cfg.indenter.cap_width_mm, cfg.indenter.cap_height_mm
        ax.add_patch(plt.Rectangle((cx - w / 2, cz), w, h, color="gold", alpha=0.6))
    ax.set_aspect("equal")
    ax.set_xlim(-5, cfg.block.length_mm + 5)
    ax.set_ylim(-5, cfg.block.height_mm + 30)
    ax.set_xlabel("x from left end (mm)"); ax.set_ylabel("height (mm)")
    ax.set_title(f"{code}: dot movement at t = {t[km]:.3f} s, arrows true to scale (mm)")
    fig.tight_layout()
    fig.savefig(qa_dir / f"{code}_field.png", dpi=100)
    plt.close(fig)

    # 3. overlays on four frames
    want = [k0, km]
    if "t_release" in ev:
        want.append(int(np.argmin(np.abs(t - (ev["t_release"] + 0.05)))))
    want.append(len(t) - 1)
    imgs = {}
    for idx, tt, img in clip.frames(color=True):
        if idx in want:
            imgs[idx] = img
        if idx >= max(want):
            break
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.5))
    P = A["pts_px"]
    for ax in axs.ravel()[len(want):]:
        ax.axis("off")
    for ax, k in zip(axs.ravel(), want):
        if k not in imgs:
            continue
        img = imgs[k]
        ok = np.isfinite(P[k, :, 0])
        x0, x1 = np.nanmin(P[k0, :, 0]) - 260, np.nanmax(P[k0, :, 0]) + 260
        y0, y1 = np.nanmin(P[k0, :, 1]) - 220, np.nanmax(P[k0, :, 1]) + 80
        ax.imshow(img[..., ::-1])
        ax.plot(P[k, :G][ok[:G], 0], P[k, :G][ok[:G], 1], "o", mfc="none", mec="lime", ms=7, mew=1)
        if P.shape[1] > G:
            ax.plot(P[k, G:][ok[G:], 0], P[k, G:][ok[G:], 1], "o", mfc="none", mec="red", ms=10, mew=1.5)
        lost = ~ok[:G] & np.isfinite(P[k0, :G, 0])
        ax.plot(P[k0, :G][lost, 0], P[k0, :G][lost, 1], "x", color="red", ms=6)
        ax.set_xlim(max(0, x0), min(img.shape[1], x1)); ax.set_ylim(min(img.shape[0], y1), max(0, y0))
        ax.set_title(f"frame {k}, t = {t[k]:.3f} s: green = tracked, red x = not tracked this frame", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(qa_dir / f"{code}_frames.png", dpi=90)
    plt.close(fig)


def noise_floor(tracks_dir: Path, codes):
    """Per-dot standard deviation (mm) over each whole static clip."""
    out = {}
    for code in codes:
        f = tracks_dir / f"{code}.npz"
        if not f.exists():
            continue
        A = np.load(f)
        k0 = int(A["k0"]); G = int(A["rows"]) * int(A["cols"])
        s = np.nanstd(A["pts_mm"][k0:, :G], axis=0)       # (G, 2)
        out[code] = {"x_std_mm_median": float(np.nanmedian(s[:, 0])), "z_std_mm_median": float(np.nanmedian(s[:, 1])),
                     "std_mm_p95": float(np.nanpercentile(np.linalg.norm(s, axis=1), 95)),
                     "frames": int(len(A["t"]) - k0)}
    return out


def repeatability(tracks_dir: Path, pairs, window=(-0.1, 1.0)):
    """RMS difference (mm) between two clips of the same press, aligned on the contact time."""
    out = {}
    for a, b in pairs:
        fa, fb = tracks_dir / f"{a}.json", tracks_dir / f"{b}.json"
        if not (fa.exists() and fb.exists()):
            continue
        ma, mb = json.loads(fa.read_text()), json.loads(fb.read_text())
        ea, eb = ma.get("events", {}), mb.get("events", {})
        if "t_contact" not in ea or "t_contact" not in eb:
            continue
        A, B = np.load(tracks_dir / f"{a}.npz"), np.load(tracks_dir / f"{b}.npz")
        ta = A["t"] - ea["t_contact"]; tb = B["t"] - eb["t_contact"]
        end = max(ea.get("t_release", ea["t_max"]), eb.get("t_release", eb["t_max"])) - min(ea["t_contact"], eb["t_contact"])
        grid = np.arange(window[0], end + window[1], 0.004)
        da, db = A["disp_mm"], B["disp_mm"]
        diffs = []
        for d in range(da.shape[1]):
            for k in range(2):
                ya = np.interp(grid, ta, da[:, d, k], left=np.nan, right=np.nan)
                yb = np.interp(grid, tb, db[:, d, k], left=np.nan, right=np.nan)
                diffs.append(ya - yb)
        diffs = np.array(diffs)
        out[f"{a}~{b}"] = {"rms_mm": float(np.sqrt(np.nanmean(diffs ** 2))),
                           "max_depth_mm": [ea.get("max_depth_mm"), eb.get("max_depth_mm")],
                           "seconds_compared": float(grid[-1] - grid[0])}
    return out


def summary_csv(tracks_dir: Path, out_csv: Path):
    rows = []
    for f in sorted(tracks_dir.glob("*.json")):
        m = json.loads(f.read_text())
        ev = m.get("events", {})
        rows.append({
            "code": m["code"], "kind": m["kind"], "file": m["file"],
            "frames": m["time"]["n_frames"], "duration_s": round(m["time"]["duration_s"], 2),
            "fps_avg": round(m["time"]["fps_average"], 1), "dropped_gaps": m["time"]["n_long_gaps"],
            "rest_t_s": round(m["rest"]["t"], 2), "rest_noise_mm": m["rest"]["noise_rms_mm"],
            "dots_tracked_frac": round(m["valid_fraction_after_rest"], 3),
            "max_depth_mm": ev.get("max_depth_mm"), "press_speed_mm_s": ev.get("press_speed_mm_s"),
            "hold_s": ev.get("hold_s"), "t_contact": ev.get("t_contact"), "t_release": ev.get("t_release"),
            "tilt_max_deg": m.get("indenter", {}).get("tilt_deg_max"),
            "camera_max_shift_px": m["camera_still"].get("max_shift_px"),
            "foam_length_mm": m["outline"].get("length_mm"), "foam_top_mm": m["outline"].get("z_top_mm"),
        })
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows
