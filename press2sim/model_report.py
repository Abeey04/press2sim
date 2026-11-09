"""Fit the model, predict every press, and write the comparison plots, numbers and a video."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import fit as F
from . import model as M

ROW_NAMES = {6: "top row", 3: "middle row", 0: "bottom row"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_set(trials, tracks: Path, log):
    clips = {}
    for r in trials:
        code, kind = r["code"], r["kind"]
        if kind not in ("press", "weight", "fingertip") or not (tracks / f"{code}.npz").exists():
            continue
        load = float(r["load_kg"]) if r.get("load_kg") else None
        cd = M.load_clip(tracks, code, kind, load_kg=load)
        cd.held_out = str(r.get("held_out", "")).strip().lower() == "yes"
        clips[code] = cd
    return clips


def metrics(cd: M.ClipData, U: np.ndarray, start=None):
    sel = M.window(cd, start if start is not None else cd.events.get("t_contact", cd.t[cd.k0]) - 0.1)
    err = U[sel] - cd.disp[sel]
    ev = cd.events
    out = {"rms_error_mm": M.rms(err), "rms_movement_mm": M.rms(cd.disp[sel]),
           "max_movement_mm": float(np.nanmax(np.linalg.norm(cd.disp[sel], axis=-1)))}
    out["relative_error"] = out["rms_error_mm"] / out["rms_movement_mm"]
    if "t_max" in ev:
        k = int(np.searchsorted(cd.t, ev["t_max"]))
        out["error_at_deepest_mm"] = M.rms(U[k] - cd.disp[k])
        out["movement_at_deepest_mm"] = M.rms(cd.disp[k])
    rec = F.recovery_frames(cd)
    if rec.any():
        out["error_recovery_mm"] = M.rms(U[rec] - cd.disp[rec])
        out["movement_recovery_mm"] = M.rms(cd.disp[rec])
    rows = {}
    for r, name in ROW_NAMES.items():
        rows[name] = M.rms(err[:, r * 15:(r + 1) * 15])
    out["rms_error_by_row_mm"] = rows
    return out


def finger_track(cd: M.ClipData, tab: M.Table, x_c: float):
    """Fingertip depth from the one top-row dot under it: the model maps its sink to the indentation."""
    top = np.arange(6 * 15, 7 * 15)
    j = top[np.nanargmin(np.abs(cd.rest[top, 0] - x_c))]
    Uk = tab.at(cd.rest[[j]], x_c, cd.z_top)[:, 0, 1]          # (K,) sink of that dot per depth
    dz = -np.nan_to_num(cd.disp[:, j, 1])
    depth = np.interp(dz, -Uk, tab.d)
    depth[:cd.k0] = np.nan
    return depth, int(j)


# ------------------------------------------------------------------------------------------ plots

def plot_clip(cd, U, q, Fz, contact, p, out: Path, title, baseline=None):
    plt = _plt()
    t = cd.t
    sel = M.window(cd, cd.events.get("t_contact", t[cd.k0]) - 0.5)
    t0 = t[sel][0]
    xc = cd.x_c if cd.x_c is not None else 117.0
    under = int(np.nanargmin(np.abs(cd.rest[6 * 15:7 * 15, 0] - xc)))
    picks = [(6, under, "top row, under the cap"), (6, max(0, under - 4), "top row, 40 mm to the side"),
             (3, under, "middle row, under the cap"), (0, under, "bottom row, under the cap")]
    fig, axs = plt.subplots(len(picks) + 1, 1, figsize=(10, 11), sharex=True)
    for ax, (r, c, name) in zip(axs, picks):
        j = r * 15 + c
        ax.plot(t[sel] - t0, cd.disp[sel, j, 1], color="k", lw=1.6, label="measured (vertical)")
        ax.plot(t[sel] - t0, U[sel, j, 1], color="tab:orange", lw=1.4, ls="--", label="model (vertical)")
        ax.plot(t[sel] - t0, cd.disp[sel, j, 0], color="0.55", lw=1.0, label="measured (sideways)")
        ax.plot(t[sel] - t0, U[sel, j, 0], color="tab:blue", lw=1.0, ls="--", label="model (sideways)")
        if baseline is not None:
            ax.plot(t[sel] - t0, baseline[sel, j, 1], color="tab:red", lw=0.8, ls=":", label="no damping")
        ax.set_ylabel("mm"); ax.set_title(name, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=3, loc="lower left")
    ax = axs[-1]
    if cd.depth is not None:
        ax.plot(t[sel] - t0, cd.depth[sel], color="k", lw=1.2, label="cap depth (measured input)")
    ax.plot(t[sel] - t0, q[sel], color="tab:orange", lw=1.2, ls="--", label="foam surface under the cap (model)")
    ax.set_ylabel("mm"); ax.set_xlabel("time from just before contact (s)")
    ax.set_ylim(-3, 1.1 * np.nanmax(q[sel]) + 3)
    ax2 = ax.twinx()
    ax2.plot(t[sel] - t0, Fz[sel] * p.E, color="tab:green", lw=1.0, label="force (model)")
    ax2.set_ylabel("force on the cap (N)", color="tab:green")
    cs = np.where(sel & contact)[0]
    if len(cs):
        ax.axvspan(t[cs[0]] - t0, t[cs[-1]] - t0, color="tab:orange", alpha=0.08, label="cap touching (model)")
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_fields(cd, U, out: Path, title):
    plt = _plt()
    ev = cd.events
    times = [("deepest", ev.get("t_max"))]
    if ev.get("t_release") is not None:
        times.append(("0.1 s after the foam lets go of the cap", ev["t_release"] + 0.1))
    fig, axs = plt.subplots(len(times), 1, figsize=(9, 3.3 * len(times)))
    axs = np.atleast_1d(axs)
    for ax, (name, tt) in zip(axs, times):
        if tt is None:
            continue
        k = int(np.searchsorted(cd.t, tt))
        P = cd.rest
        ax.add_patch(_plt().Rectangle((0, 0), M.BLOCK[0], cd.z_top, fill=False, ec="0.6"))
        ax.quiver(P[:, 0], P[:, 1], cd.disp[k, :, 0], cd.disp[k, :, 1], angles="xy", scale_units="xy", scale=1,
                  color="k", width=0.003, label="measured")
        ax.quiver(P[:, 0], P[:, 1], U[k, :, 0], U[k, :, 1], angles="xy", scale_units="xy", scale=1,
                  color="tab:orange", width=0.003, alpha=0.85, label="model")
        low = float(np.nanmin(np.r_[P[:, 1] + cd.disp[k, :, 1], P[:, 1] + U[k, :, 1], 0.0]))
        ax.set_aspect("equal"); ax.set_xlim(-5, M.BLOCK[0] + 5); ax.set_ylim(low - 5, cd.z_top + 8)
        ax.set_title(f"{name} (t = {cd.t[k]:.2f} s): RMS error {M.rms(U[k] - cd.disp[k]):.2f} mm", fontsize=9)
        ax.set_xlabel("x along the face (mm)"); ax.set_ylabel("height (mm)")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def render_3d(p: M.Params, x_c: float, depth: float, out: Path, title: str):
    """The FE block at one indentation, outer faces coloured by how far they moved."""
    from . import fem
    plt = _plt()
    fb = fem.FoamBlock(p.nu, p.eps_y, p.h, ("cap", x_c, p.y0, M.CAP[0], M.CAP[1]), block=M.BLOCK, support=p.support,
                       eps_d=p.eps_d, fine=M.FINE, skin=None if p.skin_t <= 0 or p.skin_k >= 1 else (p.skin_t, p.skin_k))
    for d in np.linspace(depth / 4, depth, 4):
        fb.solve_depth(float(d))
    b = fb.b
    X = b.nodes_xyz()
    Ud = fb.u.reshape(*b.shape, 3)
    D = X + Ud
    mag = np.linalg.norm(Ud, axis=-1)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(0, depth)
    cmap = plt.get_cmap("viridis")
    faces = [(D[:, 0, :, :], mag[:, 0, :]), (D[:, :, -1, :], mag[:, :, -1]), (D[0, :, :, :], mag[0, :, :]),
             (D[-1, :, :, :], mag[-1, :, :])]
    for P, m in faces:
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=cmap(norm(m)), rstride=1, cstride=1,
                        linewidth=0.2, edgecolor=(0, 0, 0, 0.15), shade=False)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("depth into block (mm)"); ax.set_zlabel("height (mm)")
    ax.set_box_aspect((M.BLOCK[0], M.BLOCK[1], M.BLOCK[2] * 1.3))
    ax.view_init(elev=22, azim=-62)
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(m, ax=ax, shrink=0.6, label="movement (mm)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_video(cd: M.ClipData, U, raw: Path, calib, rotation, meta, out: Path, q=None, every=2):
    """The clip with measured dots (green) and the model's prediction (magenta) drawn on it."""
    import cv2
    from .video import Clip
    x_left = meta["outline"]["x_left_mm"]
    pred_mm = cd.rest[None] + U
    pred_mm = np.concatenate([pred_mm[..., :1] + x_left, pred_mm[..., 1:]], -1)
    clip = Clip(raw, rotation)
    sel = M.window(cd, cd.events.get("t_contact", cd.t[cd.k0]) - 0.6)
    k_first, k_last = np.where(sel)[0][[0, -1]]
    writer = None
    ffmpeg = shutil.which("ffmpeg")
    for i, t, bgr in clip.frames(color=True, start=k_first, stop=k_last + 1, step=every):
        img = bgr.copy()
        meas = cd.disp[i]
        mpx = calib.mm_to_px(np.c_[cd.rest[:, 0] + meas[:, 0] + x_left, cd.rest[:, 1] + meas[:, 1]])
        ppx = calib.mm_to_px(pred_mm[i])
        for (a, b), ok in zip(mpx, np.isfinite(meas).all(-1)):
            if ok:
                cv2.circle(img, (int(round(a)), int(round(b))), 9, (60, 200, 60), 2, cv2.LINE_AA)
        for a, b in ppx:
            if np.isfinite(a):
                c = (int(round(a)), int(round(b)))
                cv2.line(img, (c[0] - 6, c[1]), (c[0] + 6, c[1]), (230, 40, 230), 2, cv2.LINE_AA)
                cv2.line(img, (c[0], c[1] - 6), (c[0], c[1] + 6), (230, 40, 230), 2, cv2.LINE_AA)
        err = M.rms(U[i] - meas)
        txt = f"{cd.code}  t={t:5.2f}s  green: tracked dots   magenta: model prediction   RMS error {err:4.2f} mm"
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (255, 255, 255), -1)
        cv2.putText(img, txt, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        if cd.depth is not None and np.isfinite(cd.depth[i]):
            cv2.putText(img, f"cap depth {cd.depth[i]:5.1f} mm" + (f"   model foam surface {q[i]:5.1f} mm" if q is not None else ""),
                        (14, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        small = cv2.resize(img, (img.shape[1] // 2 * 2 // 2, img.shape[0] // 2 * 2 // 2))
        if writer is None:
            h, w = small.shape[:2]
            if ffmpeg:
                writer = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                           "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                           "-crf", "23", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
            else:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        if ffmpeg:
            writer.stdin.write(small.tobytes())
        else:
            writer.write(small)
    if ffmpeg:
        writer.stdin.close(); writer.wait()
    else:
        writer.release()


# ------------------------------------------------------------------------------------------ driver

def run(trials, out: Path, raw: Path, calib=None, rotation=None, log=print, video=True, render=True, start=None,
        shape_passes=2, shape_steps=None, refit_shape=True):
    t_start = time.time()
    tracks = out / "tracks"
    mdir = out / "model"; mdir.mkdir(parents=True, exist_ok=True)
    cache = mdir / "cache"
    clips = load_set(trials, tracks, log)
    train = [c for c in clips.values() if c.kind == "press" and not c.held_out and c.depth is not None]
    weights = [c for c in clips.values() if c.kind == "weight" and c.load_kg and c.load_kg >= 0.9]
    held = [c for c in clips.values() if c.held_out]
    log(f"training presses: {', '.join(c.code for c in train)}")
    log(f"dead weights: {', '.join(c.code for c in weights)}; held out: {', '.join(c.code for c in held)}")
    p = M.Params() if start is None else start
    report = {"train": [c.code for c in train], "weights": [c.code for c in weights], "held_out": [c.code for c in held],
              "start": asdict(p)}

    if refit_shape:
        log("1/4 what the block stands on (dead weights)")
        p, sup_loss, sup_rigid = F.fit_support(weights, cache, p, log=log)
        log("2/4 shape of the deformation (presses, while the cap pushes)")
        p, shape_rms, hist = F.fit_shape(train, cache, p, log=log, passes=shape_passes, steps=shape_steps)
        report["shape"] = {"rms_mm": shape_rms, "history": hist}
        p, sup_loss, sup_rigid = F.fit_support(weights, cache, p, log=log)    # again, with the fitted shape
        report["support"] = {"value_per_mm": p.support, "pattern_misfit": sup_loss, "pattern_misfit_if_rigid": sup_rigid}
    else:
        log("1-2/4 shape and support kept from the earlier fit")
        sup_loss, sup_rigid = F.support_loss(p, weights, cache)[0], F.support_loss(replace(p, support=1e3), weights, cache)[0]
        report["support"] = {"value_per_mm": p.support, "pattern_misfit": sup_loss, "pattern_misfit_if_rigid": sup_rigid}
        report["shape"] = {"rms_mm": F.shape_loss(p, train, cache)}
        log(f"  shape RMS {report['shape']['rms_mm']:.3f} mm on the pressing frames; support pattern misfit "
            f"{sup_loss:.3f} (rigid {sup_rigid:.3f})")

    log("3/4 damping (whole presses: letting go of the cap and creeping back)")
    tab = M.build_table(p, ("cap", 117.0), cache=cache)
    p, time_rms, time_none = F.fit_time(train, tab, p, log=log)
    report["damping"] = {"recovery_rms_mm": time_rms, "recovery_rms_no_damping_mm": time_none}

    log("4/4 stiffness (dead weights, force known)")
    ptab = M.build_table(p, ("plate",), cache=cache)
    E, e_rms, per_w, wdata = F.fit_E(weights, ptab, p, log=log)
    p = replace(p, E=E)
    report["stiffness"] = {"E_Pa": E, "rms_mm": e_rms, "per_clip": per_w}

    # derived numbers people ask for: effective stiffness and damping of the block under the cap
    R = lambda d: float(np.interp(d, tab.d, tab.R))
    eff = {}
    for d in (5.0, 10.0, 20.0, 30.0):
        k_inst = E * (R(d + 0.5) - R(d - 0.5))                    # N/mm, instantaneous tangent
        eff[f"{d:g}mm"] = {"force_N_long_term": E * p.g_inf * R(d), "force_N_fast": E * R(d),
                           "stiffness_N_per_mm_fast": k_inst, "stiffness_N_per_mm_long_term": k_inst * p.g_inf}
    k10 = E * (R(10.5) - R(9.5))
    damping = {"relaxation": [{"fraction": g, "tau_s": tau, "viscosity_Pa_s": g * E * tau,
                               "c_N_s_per_mm_at_10mm": g * k10 * tau} for g, tau in zip(p.g, p.tau)],
               "dashpot": {"tau_v_s": p.tau_v, "viscosity_Pa_s": E * p.tau_v, "c_N_s_per_mm_at_10mm": k10 * p.tau_v}}
    report["params"] = asdict(p)
    report["effective"] = {"under_cap": eff, "damping": damping,
                           "plateau_stress_Pa": E * p.eps_y, "plate_stiffness_N_per_mm": E * float(ptab.R[1] / ptab.d[1])}
    (mdir / "params.json").write_text(json.dumps({k: v for k, v in report.items()}, indent=1, default=float))

    # predictions: every press with the cap, plus the fingertip
    p_nodamp = replace(p, g=tuple(1e-9 for _ in p.g), tau_v=1e-9)
    rows = []
    preds = {}
    for cd in [c for c in clips.values() if c.kind == "press"]:
        if abs(cd.x_c - 117.0) > 12:                              # off-centre press: its own FE run
            ctab = M.build_table(p, ("cap", round(cd.x_c, 1)), cache=cache)
        else:
            ctab = tab
        U, q, Fz, c = M.predict_clip(cd, ctab, p)
        U0, q0, _, _ = M.predict_clip(cd, ctab, p_nodamp)
        mt = metrics(cd, U); m0 = metrics(cd, U0)
        rows.append({"clip": cd.code, "set": "held out" if cd.held_out else "training", **{k: v for k, v in mt.items() if not isinstance(v, dict)},
                     **{f"{k} (row)": v for k, v in mt["rms_error_by_row_mm"].items()},
                     "rms_error_no_damping_mm": m0["rms_error_mm"],
                     "recovery_error_no_damping_mm": m0.get("error_recovery_mm"),
                     "max_depth_mm": cd.events.get("max_depth_mm"), "peak_force_model_N": float(np.nanmax(Fz) * E)})
        preds[cd.code] = (U, q, Fz, c, U0)
        np.savez_compressed(mdir / f"pred_{cd.code}.npz", t=cd.t, U=U, q=q, F_N=Fz * E, contact=c, U_no_damping=U0)
    for cd in [c for c in clips.values() if c.kind == "fingertip"]:
        sel = M.window(cd)
        top = np.arange(6 * 15, 7 * 15)
        k_deep = int(np.nanargmin(np.where(sel, np.nanmin(cd.disp[:, top, 1], axis=1), np.inf)))
        x_c = float(cd.rest[top[np.nanargmin(cd.disp[k_deep, top, 1])], 0])
        ftab = M.build_table(p, ("disc", round(x_c, 1), 9.0, 8.0), cache=cache)
        depth, j = finger_track(cd, ftab, x_c)
        cd.depth, cd.x_c = depth, x_c
        # the press itself: from where the fingertip first goes 1 mm in (hands pass in front earlier on)
        k_on = int(np.where((np.arange(len(depth)) < k_deep) & ~(np.nan_to_num(depth) > 1.0))[0][-1]) + 1
        t_on = float(cd.t[k_on])
        cd.events = dict(cd.events, t_max=float(cd.t[k_deep]), t_contact=t_on,
                         t_lift=float(cd.t[k_deep]), max_depth_mm=float(np.nanmax(depth)))
        U, q, Fz, c = M.predict_clip(cd, ftab, p)
        U0, q0, _, _ = M.predict_clip(cd, ftab, p_nodamp)
        others = np.ones(U.shape[1], bool); others[j] = False
        mt = metrics(cd, np.where(others[None, :, None], U, np.nan), start=t_on - 0.5)
        m0 = metrics(cd, np.where(others[None, :, None], U0, np.nan), start=t_on - 0.5)
        rows.append({"clip": cd.code, "set": "held out (fingertip; driven by the one dot under it)",
                     **{k: v for k, v in mt.items() if not isinstance(v, dict)},
                     **{f"{k} (row)": v for k, v in mt["rms_error_by_row_mm"].items()},
                     "rms_error_no_damping_mm": m0["rms_error_mm"], "recovery_error_no_damping_mm": None,
                     "max_depth_mm": float(np.nanmax(depth)), "peak_force_model_N": float(np.nanmax(Fz) * E)})
        preds[cd.code] = (U, q, Fz, c, U0)
        # also what the scoring needs: where the fingertip was, the dot that drove it (not scored), when it went in
        np.savez_compressed(mdir / f"pred_{cd.code}.npz", t=cd.t, U=U, q=q, F_N=Fz * E, contact=c, U_no_damping=U0,
                            x_c=x_c, driving_dot=j, t_on=t_on, score_from=t_on - 0.5)

    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(mdir / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    report["predictions"] = rows
    (mdir / "params.json").write_text(json.dumps(report, indent=1, default=float))
    log("predictions (RMS error / RMS movement of all dots, from contact to the end):")
    for r in rows:
        log(f"  {r['clip']:4s} {r['set'][:8]:8s} error {r['rms_error_mm']:.2f} mm of {r['rms_movement_mm']:.2f} mm "
            f"({100 * r['relative_error']:.0f}%), no damping {r['rms_error_no_damping_mm']:.2f} mm")

    # figures for the held-out clips and one training clip
    show = [c for c in clips.values() if c.held_out] + [clips[k] for k in ("T05", "T09") if k in clips]
    for cd in show:
        U, q, Fz, c, U0 = preds[cd.code]
        tag = "held out" if cd.held_out else "training"
        plot_clip(cd, U, q, Fz, c, p, mdir / f"{cd.code}_timeseries.png",
                  f"{cd.code} ({tag}): model driven by the cap's depth only", baseline=U0)
        plot_fields(cd, U, mdir / f"{cd.code}_field.png", f"{cd.code} ({tag}): measured (black) vs model (orange)")
    if render:
        for cd in [c for c in clips.values() if c.held_out and c.kind == "press"][:1]:
            render_3d(p, cd.x_c, float(cd.events.get("max_depth_mm", 20.0)), mdir / f"{cd.code}_3d.png",
                      f"{cd.code}: the model block at its deepest ({cd.events.get('max_depth_mm', 0):.0f} mm)")
    log(f"model done in {time.time() - t_start:.0f} s -> {mdir}")
    return p, report, preds, clips
