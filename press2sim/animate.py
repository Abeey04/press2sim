"""The simulated foam block in 3D, side by side with the real clip, driven by the same cap motion.

Left: the original video with the tracked dots (green) and the model's predicted dots (magenta).
Right: the whole simulated block, coloured by how far each point has moved, with the cap where the
measured cap was and the predicted dots on its front face. Bottom: the cap's measured depth (the only
input), the simulated foam surface under the cap and the simulated force on the cap.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import model as M


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _quads(P):
    """(a, b, 3) grid of points -> (n, 4, 3) quads."""
    return np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2).reshape(-1, 4, 3)


def _qmean(V):
    return 0.25 * (V[:-1, :-1] + V[1:, :-1] + V[1:, 1:] + V[:-1, 1:]).ravel()


def _box(x0, x1, y0, y1, z0, z1):
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [c[i] for i in f]


def block_polys(full: M.FullTable, q: float, cmap, vmax: float):
    """Outer faces of the deformed block as quads and their colours."""
    X = np.stack(np.meshgrid(full.xn, full.yn, full.zn, indexing="ij"), -1)
    U = full.at_depth(q)
    D = X + U
    mag = np.linalg.norm(U, axis=-1)
    faces = [(D[:, 0], mag[:, 0]), (D[:, :, -1], mag[:, :, -1]), (D[0], mag[0]), (D[-1], mag[-1]),
             (D[:, -1], mag[:, -1])]
    polys, vals = [], []
    for P, V in faces:
        polys.append(_quads(P)); vals.append(_qmean(V))
    polys = np.concatenate(polys); vals = np.concatenate(vals)
    return polys, cmap(np.clip(vals / vmax, 0, 1))


def dots_3d(full: M.FullTable, q: float, rest_xz: np.ndarray):
    """Predicted 3D positions of the face dots (dot rest positions already in the FE's frame)."""
    U = full.at_depth(q)[:, 0]                                   # front face (nx, nz, 3)
    x, z = rest_xz[:, 0], rest_xz[:, 1]
    xi = np.clip(np.searchsorted(full.xn, x) - 1, 0, len(full.xn) - 2)
    zi = np.clip(np.searchsorted(full.zn, z) - 1, 0, len(full.zn) - 2)
    fx = np.clip((x - full.xn[xi]) / (full.xn[xi + 1] - full.xn[xi]), 0, 1)[:, None]
    fz = np.clip((z - full.zn[zi]) / (full.zn[zi + 1] - full.zn[zi]), 0, 1)[:, None]
    u = (1 - fx) * (1 - fz) * U[xi, zi] + fx * (1 - fz) * U[xi + 1, zi] + (1 - fx) * fz * U[xi, zi + 1] + fx * fz * U[xi + 1, zi + 1]
    return np.c_[x + u[:, 0], u[:, 1] - 0.6, z + u[:, 2]]


def simulate_video(code: str, out: Path, raw_video: Path, calib, rotation, dst: Path, p: M.Params, cache: Path,
                   fps_out=30, step=4, log=print):
    """Side-by-side video: the real press and the simulated block."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .video import Clip
    plt = _plt()
    cd = M.load_clip(out / "tracks", code, "press")
    meta = json.loads((out / "tracks" / f"{code}.json").read_text())
    pred = np.load(out / "model" / f"pred_{code}.npz")
    U, q, F = pred["U"], pred["q"], pred["F_N"]
    x_fe = 117.0 if abs(cd.x_c - 117.0) <= 12 else round(cd.x_c, 1)
    full = M.build_full(p, ("cap", x_fe), cache=cache)
    rest_fe = np.c_[cd.rest[:, 0] + (x_fe - cd.x_c), cd.rest[:, 1] * M.BLOCK[2] / cd.z_top]
    x_left = meta["outline"]["x_left_mm"]
    ev = cd.events
    t0 = ev["t_contact"] - 0.4
    t1 = min(cd.t_end, (ev.get("t_release") or ev["t_lift"]) + 1.4)
    k_idx = np.where((cd.t >= t0) & (cd.t <= t1))[0][::step]
    vmax = max(5.0, float(np.nanmax(q)))
    cmap = plt.get_cmap("viridis")
    H = M.BLOCK[2]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], left=0.04, right=0.94, top=0.9, bottom=0.07, hspace=0.16,
                          wspace=0.04)
    ax_v = fig.add_subplot(gs[0, 0]); ax_v.set_axis_off()
    ax_3 = fig.add_subplot(gs[0, 1], projection="3d")
    ax_t = fig.add_subplot(gs[1, :])
    crop = (380, 130, 1480, 880)                                  # x0, y0, x1, y1 in the full frame
    im = ax_v.imshow(np.zeros((crop[3] - crop[1], crop[2] - crop[0], 3), np.uint8))
    meas_sc = ax_v.scatter([], [], s=70, facecolors="none", edgecolors=(0.2, 0.85, 0.2), linewidths=1.4)
    pred_sc = ax_v.scatter([], [], s=28, marker="+", c=[(0.95, 0.15, 0.95)], linewidths=1.4)
    ax_v.set_title("the real press: tracked dots (green circles) and predicted dots (magenta crosses)", fontsize=11)
    sl = slice(int(np.searchsorted(cd.t, t0)), int(np.searchsorted(cd.t, t1)))
    tt = cd.t[sl] - t0
    ax_t.plot(tt, cd.depth[sl], color="k", lw=1.6, label="cap depth: the only input (measured from the bottle dots)")
    ax_t.plot(tt, q[sl], color="tab:orange", lw=1.3, ls="--", label="simulated foam surface under the cap")
    ax_t.set_ylabel("depth (mm)"); ax_t.set_xlabel("time (s)")
    ax_t.set_xlim(0, t1 - t0); ax_t.set_ylim(-3, 1.15 * np.nanmax(q[sl]) + 3)
    ax_f = ax_t.twinx()
    ax_f.plot(tt, F[sl], color="tab:green", lw=1.2, label="simulated force on the cap")
    ax_f.set_ylabel("force (N)", color="tab:green")
    h1, l1 = ax_t.get_legend_handles_labels(); h2, l2 = ax_f.get_legend_handles_labels()
    ax_t.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower center")
    marker = ax_t.axvline(0, color="tab:red", lw=1.2)
    slow = (cd.t[k_idx[-1]] - cd.t[k_idx[0]]) / max(1e-9, len(k_idx) / fps_out)
    fig.suptitle(f"{code} (held out): the foam simulated from the cap's motion alone, next to the real clip "
                 f"(playing {1 / slow:.1f}x slower than real time)", fontsize=13, y=0.975)

    ffmpeg = shutil.which("ffmpeg")
    W, Hh = fig.canvas.get_width_height()
    if ffmpeg:
        proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}",
                                 "-r", str(fps_out), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                                 "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    else:
        import cv2
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, Hh))
    clip = Clip(raw_video, rotation)
    want = set(int(k) for k in k_idx)
    coll = None; dsc = None
    n = 0
    for i, t, bgr in clip.frames(color=True, start=int(k_idx[0]), stop=int(k_idx[-1]) + 1):
        if i not in want:
            continue
        rgb = bgr[crop[1]:crop[3], crop[0]:crop[2], ::-1]
        im.set_data(rgb)
        meas = cd.disp[i]
        ok = np.isfinite(meas).all(-1)
        mpx = calib.mm_to_px(np.c_[cd.rest[:, 0] + meas[:, 0] + x_left, cd.rest[:, 1] + meas[:, 1]])[ok]
        ppx = calib.mm_to_px(np.c_[cd.rest[:, 0] + U[i, :, 0] + x_left, cd.rest[:, 1] + U[i, :, 1]])
        meas_sc.set_offsets(mpx - crop[:2]); pred_sc.set_offsets(ppx - crop[:2])
        # the simulated block
        ax_3.cla()
        ax_3.computed_zorder = False                  # draw the dots over the block's face
        qi = float(q[i]) if np.isfinite(q[i]) else 0.0
        polys, cols = block_polys(full, qi, cmap, vmax)
        d_cap = cd.depth[i] if np.isfinite(cd.depth[i]) else -60.0
        zb = H - d_cap
        cap = _box(x_fe - 30, x_fe + 30, max(p.y0, 0.0), p.y0 + 60, zb, zb + 19)
        body = _box(x_fe - 30.5, x_fe + 30.5, max(p.y0, 0.0) - 0.5, p.y0 + 60.5, zb + 19, zb + 95)
        allp = list(polys) + cap + body
        allc = list(cols) + [(1.0, 0.82, 0.1, 1.0)] * 6 + [(0.93, 0.93, 0.9, 0.55)] * 6
        coll = Poly3DCollection(allp, facecolors=allc, edgecolors=(0, 0, 0, 0.12), linewidths=0.3, zorder=1)
        ax_3.add_collection3d(coll)
        P3 = dots_3d(full, qi, rest_fe)
        ax_3.scatter(P3[:, 0], P3[:, 1], P3[:, 2], s=9, c=[(0.95, 0.15, 0.95)], depthshade=False, zorder=5)
        ax_3.set_xlim(0, M.BLOCK[0]); ax_3.set_ylim(0, M.BLOCK[1]); ax_3.set_zlim(0, H + 95)
        ax_3.set_box_aspect((M.BLOCK[0], M.BLOCK[1], H + 95))
        ax_3.view_init(elev=16, azim=-76)
        ax_3.set_axis_off()
        ax_3.set_title(f"the simulated block (3D voxel model); colour: how far it moved, 0 to {vmax:.0f} mm", fontsize=11, pad=-8)
        marker.set_xdata([t - t0, t - t0])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if ffmpeg:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            writer.write(frame[..., ::-1].copy())
        n += 1
        if n % 50 == 0:
            log(f"  {n}/{len(k_idx)} frames")
    if ffmpeg:
        proc.stdin.close(); proc.wait()
    else:
        writer.release()
    plt.close(fig)
    return dst


def grid_polys(rest, disp, cmap, vmax):
    """Outer faces of a deformed vertex grid (nx, ny, nz, 3) as quads with colours."""
    D = rest + disp
    mag = np.linalg.norm(disp, axis=-1)
    faces = [(D[:, 0], mag[:, 0]), (D[:, :, -1], mag[:, :, -1]), (D[0], mag[0]), (D[-1], mag[-1]), (D[:, -1], mag[:, -1])]
    polys = np.concatenate([_quads(P) for P, _ in faces])
    vals = np.concatenate([_qmean(V) for _, V in faces])
    return polys, cmap(np.clip(vals / vmax, 0, 1))


def face_points_3d(rest, disp, pts_xz):
    """3D positions of points on the front face (y = 0) of a deformed vertex grid."""
    xs = rest[:, 0, 0, 0]; zs = rest[0, 0, :, 2]
    U = disp[:, 0]
    x, z = pts_xz[:, 0], pts_xz[:, 1]
    xi = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    zi = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
    fx = np.clip((x - xs[xi]) / (xs[xi + 1] - xs[xi]), 0, 1)[:, None]
    fz = np.clip((z - zs[zi]) / (zs[zi + 1] - zs[zi]), 0, 1)[:, None]
    u = (1 - fx) * (1 - fz) * U[xi, zi] + fx * (1 - fz) * U[xi + 1, zi] + (1 - fx) * fz * U[xi, zi + 1] + fx * fz * U[xi + 1, zi + 1]
    return np.c_[x + u[:, 0], u[:, 1] - 0.6, z + u[:, 2]]


def compare_video(code, out: Path, raw_video: Path, calib, rotation, dst: Path, p: M.Params, mj, U_mj, cache: Path,
                  fps_out=30, step=5, log=print):
    """Real clip | our simulated block | MuJoCo's block, all driven by the same measured cap depth.

    mj: (frames, grid_rest, grid_disp_per_frame) from mujoco_foam.dynamic_run; U_mj: (T, N, 2) MuJoCo's dots.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .video import Clip
    plt = _plt()
    cd = M.load_clip(out / "tracks", code, "press")
    meta = json.loads((out / "tracks" / f"{code}.json").read_text())
    pred = np.load(out / "model" / f"pred_{code}.npz")
    U, q, F = pred["U"], pred["q"], pred["F_N"]
    x_fe = 117.0 if abs(cd.x_c - 117.0) <= 12 else round(cd.x_c, 1)
    full = M.build_full(p, ("cap", x_fe), cache=cache)
    rest_fe = np.c_[cd.rest[:, 0] + (x_fe - cd.x_c), cd.rest[:, 1] * M.BLOCK[2] / cd.z_top]
    mj_frames, mj_rest, mj_disp = mj
    mj_index = {int(k): n for n, k in enumerate(mj_frames)}
    x_left = meta["outline"]["x_left_mm"]
    ev = cd.events
    t0 = max(ev["t_contact"] - 0.3, cd.t[mj_frames[0]])
    t1 = min(cd.t_end, cd.t[mj_frames[-1]], (ev.get("t_release") or ev["t_lift"]) + 1.2)
    k_idx = [k for k in np.where((cd.t >= t0) & (cd.t <= t1))[0][::step] if int(k) in mj_index]
    vmax = max(5.0, float(np.nanmax(q)))
    cmap = plt.get_cmap("viridis")
    H = M.BLOCK[2]
    under = int(np.nanargmin(np.abs(cd.rest[90:105, 0] - cd.x_c)))
    picks = [(6 * 15 + under, "top row, under the cap"), (3 * 15 + under, "middle row, under the cap"),
             (6 * 15 + max(0, under - 4), "top row, 40 mm to the side")]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1], left=0.035, right=0.985, top=0.9, bottom=0.07, hspace=0.18, wspace=0.08)
    ax_v = fig.add_subplot(gs[0, 0]); ax_v.set_axis_off()
    ax_o = fig.add_subplot(gs[0, 1], projection="3d")
    ax_m = fig.add_subplot(gs[0, 2], projection="3d")
    ax_d = fig.add_subplot(gs[1, :2])
    ax_e = fig.add_subplot(gs[1, 2])
    crop = (380, 130, 1480, 880)
    im = ax_v.imshow(np.zeros((crop[3] - crop[1], crop[2] - crop[0], 3), np.uint8))
    meas_sc = ax_v.scatter([], [], s=55, facecolors="none", edgecolors=(0.2, 0.85, 0.2), linewidths=1.3)
    ax_v.set_title("the real press (tracked dots in green)", fontsize=12)
    sl = np.arange(int(np.searchsorted(cd.t, t0)), int(np.searchsorted(cd.t, t1)))
    tt = cd.t[sl] - t0
    for (j, name), c in zip(picks, ("k", "0.35", "0.6")):
        ax_d.plot(tt, cd.disp[sl, j, 1], color=c, lw=2.0, label=f"measured: {name}")
        ax_d.plot(tt, U[sl, j, 1], color="tab:orange", lw=1.3, ls="--")
        ax_d.plot(tt, U_mj[sl, j, 1], color="tab:blue", lw=1.3, ls="-.")
    ax_d.plot([], [], color="tab:orange", ls="--", label="ours"); ax_d.plot([], [], color="tab:blue", ls="-.", label="MuJoCo")
    ax_d.set_ylabel("vertical movement (mm)"); ax_d.set_xlabel("time (s)"); ax_d.legend(fontsize=8, ncol=2, loc="lower left")
    ax_d.set_xlim(0, t1 - t0); ax_d.grid(alpha=0.3)
    e_o = np.array([M.rms(U[k] - cd.disp[k]) for k in sl]); e_m = np.array([M.rms(U_mj[k] - cd.disp[k]) for k in sl])
    ax_e.plot(tt, e_o, color="tab:orange", label=f"ours (whole press {M.rms((U[sl] - cd.disp[sl]).ravel()):.2f} mm)")
    ax_e.plot(tt, e_m, color="tab:blue", label=f"MuJoCo (whole press {M.rms((U_mj[sl] - cd.disp[sl]).ravel()):.2f} mm)")
    ax_e.set_title("error over all 105 dots", fontsize=10); ax_e.set_ylabel("RMS error (mm)"); ax_e.set_xlabel("time (s)")
    ax_e.legend(fontsize=8, loc="upper left"); ax_e.set_xlim(0, t1 - t0); ax_e.grid(alpha=0.3)
    m1 = ax_d.axvline(0, color="tab:red", lw=1.1); m2 = ax_e.axvline(0, color="tab:red", lw=1.1)
    slow = (cd.t[k_idx[-1]] - cd.t[k_idx[0]]) / max(1e-9, len(k_idx) / fps_out)
    fig.suptitle(f"{code} (held out): our foam model and MuJoCo, both given only the cap's measured depth "
                 f"(playing {1 / slow:.1f}x slower than real time)", fontsize=14, y=0.975)
    ffmpeg = shutil.which("ffmpeg")
    W, Hh = fig.canvas.get_width_height()
    if ffmpeg:
        proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}",
                                 "-r", str(fps_out), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                                 "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    else:
        import cv2
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, Hh))
    clip = Clip(raw_video, rotation)
    want = set(int(k) for k in k_idx)
    n = 0

    def draw_block(ax, polys, cols, dots, title, zb):
        ax.cla(); ax.computed_zorder = False
        cap = _box(x_fe - 30, x_fe + 30, max(p.y0, 0.0), p.y0 + 60, zb, zb + 19)
        body = _box(x_fe - 30.5, x_fe + 30.5, max(p.y0, 0.0) - 0.5, p.y0 + 60.5, zb + 19, zb + 95)
        ax.add_collection3d(Poly3DCollection(list(polys) + cap + body, facecolors=list(cols) + [(1.0, 0.82, 0.1, 1.0)] * 6
                                             + [(0.93, 0.93, 0.9, 0.5)] * 6, edgecolors=(0, 0, 0, 0.12), linewidths=0.3, zorder=1))
        ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2], s=7, c=[(0.95, 0.15, 0.95)], depthshade=False, zorder=5)
        ax.set_xlim(0, M.BLOCK[0]); ax.set_ylim(0, M.BLOCK[1]); ax.set_zlim(0, H + 95)
        ax.set_box_aspect((M.BLOCK[0], M.BLOCK[1], H + 95)); ax.view_init(elev=16, azim=-76); ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=-6)

    for i, t, bgr in clip.frames(color=True, start=int(k_idx[0]), stop=int(k_idx[-1]) + 1):
        if i not in want:
            continue
        im.set_data(bgr[crop[1]:crop[3], crop[0]:crop[2], ::-1])
        meas = cd.disp[i]; ok = np.isfinite(meas).all(-1)
        mpx = calib.mm_to_px(np.c_[cd.rest[:, 0] + meas[:, 0] + x_left, cd.rest[:, 1] + meas[:, 1]])[ok]
        meas_sc.set_offsets(mpx - crop[:2])
        d_cap = cd.depth[i] if np.isfinite(cd.depth[i]) else -60.0
        qi = float(q[i]) if np.isfinite(q[i]) else 0.0
        polys, cols = block_polys(full, qi, cmap, vmax)
        draw_block(ax_o, polys, cols, dots_3d(full, qi, rest_fe), "ours: crushable foam, soft top, damping", H - d_cap)
        gd = mj_disp[mj_index[i]]
        polys, cols = grid_polys(mj_rest, gd, cmap, vmax)
        draw_block(ax_m, polys, cols, face_points_3d(mj_rest, gd, rest_fe), "MuJoCo: linear elastic flex block", H - d_cap)
        m1.set_xdata([t - t0, t - t0]); m2.set_xdata([t - t0, t - t0])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if ffmpeg:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            writer.write(frame[..., ::-1].copy())
        n += 1
        if n % 50 == 0:
            log(f"  {n}/{len(k_idx)} frames")
    if ffmpeg:
        proc.stdin.close(); proc.wait()
    else:
        writer.release()
    plt.close(fig)
    return dst
