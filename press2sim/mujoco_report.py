"""Our model against MuJoCo on the same presses, the same dots and the same input (the cap's depth).

Three models, each driven only by the measured cap depth:
- ours: crushable foam, soft cut top, compliant support, quasi-linear viscoelastic (fitted);
- our solver given MuJoCo's physics: linear elastic, rigid base, no soft top. If this agrees with
  MuJoCo, the two solvers agree, and any gap to "ours" is down to the physics MuJoCo lacks;
- MuJoCo: a flex block with the fitted E and damping, Poisson's ratio picked on the training presses.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import fit as F
from . import model as M
from . import model_report as MR
from . import mujoco_foam as MJ


def mj_table_at(cd: M.ClipData, d_tab, rest_grid, disp_grid, x_fe):
    rest_fe = np.c_[cd.rest[:, 0] + (x_fe - cd.x_c), cd.rest[:, 1] * M.BLOCK[2] / cd.z_top]
    return np.stack([MJ.face_at(rest_grid, disp_grid[k], rest_fe) for k in range(len(d_tab))])


def quasi_static(cd: M.ClipData, Uk, d_tab):
    """Purely elastic response: the foam surface is wherever the cap is (no lag, no relaxation)."""
    q = np.where(np.isfinite(cd.depth), np.clip(cd.depth, 0.0, None), 0.0)
    q[:cd.k0] = 0.0
    U = M.field_at(Uk, d_tab, q) - M.field_at(Uk, d_tab, np.array([q[cd.k0]]))
    U[:cd.k0] = np.nan
    return U, q


def run(trials, out: Path, p: M.Params, nus=(0.0, 0.3), log=print):
    mdir = out / "model"
    cache = mdir / "cache"
    clips = MR.load_set(trials, out / "tracks", log)
    presses = [c for c in clips.values() if c.kind == "press"]
    train = [c for c in presses if not c.held_out]
    # MuJoCo: one static table per Poisson ratio (cap at the centre), pick the best on training presses
    best = None
    tried = {}
    for nu in nus:
        setup = MJ.MjSetup(E=p.E, nu=nu, damping=max(p.tau_v, 1e-4), y0=p.y0)
        d_tab, rest_g, disp_g = MJ.static_table(setup, 117.0, log=log, cache=cache)
        errs = []
        for cd in train:
            Uk = mj_table_at(cd, d_tab, rest_g, disp_g, 117.0)
            fr = M.contact_frames(cd)
            d0 = max(0.0, float(np.nan_to_num(cd.depth[cd.k0])))
            U = M.field_at(Uk, d_tab, np.clip(cd.depth[fr], 0, d_tab[-1])) - M.field_at(Uk, d_tab, np.array([d0]))
            errs.append((U - cd.disp[fr]).ravel())
        loss = M.rms(np.concatenate(errs))
        tried[nu] = {"pressing_frames_rms_mm": loss, "max_depth_reached_mm": float(d_tab[-1])}
        log(f"  MuJoCo nu={nu}: RMS {loss:.3f} mm on the training presses' pressing frames (table up to {d_tab[-1]:.0f} mm)")
        if best is None or loss < best[0]:
            best = (loss, nu, setup, d_tab, rest_g, disp_g)
    _, nu_mj, setup_mj, d_mj, rest_mj, disp_mj = best
    # our solver with MuJoCo's physics
    p_lin = replace(p, eps_y=10.0, h=1.0, eps_d=50.0, support=1e3, skin_t=0.0, skin_k=1.0, nu=nu_mj)
    tab_lin = M.build_table(p_lin, ("cap", 117.0), cache=cache)
    tab = M.build_table(p, ("cap", 117.0), cache=cache)
    rows = []
    for cd in presses:
        if abs(cd.x_c - 117.0) > 12:                     # off-centre press: its own tables, cap where it was
            x_fe = round(cd.x_c, 1)
            t_o = M.build_table(p, ("cap", x_fe), cache=cache)
            t_l = M.build_table(p_lin, ("cap", x_fe), cache=cache)
            d_m, r_m, u_m = MJ.static_table(setup_mj, x_fe, log=log, cache=cache)
        else:
            x_fe, t_o, t_l, d_m, r_m, u_m = 117.0, tab, tab_lin, d_mj, rest_mj, disp_mj
        U_ours, *_ = M.predict_clip(cd, t_o, p)
        U_lin, _ = quasi_static(cd, t_l.at(cd.rest, cd.x_c, cd.z_top), t_l.d)
        U_mj, _ = quasi_static(cd, mj_table_at(cd, d_m, r_m, u_m, x_fe), d_m)
        r = {"clip": cd.code, "set": "held out" if cd.held_out else "training", "max_depth_mm": cd.events.get("max_depth_mm")}
        for name, U in (("ours", U_ours), ("our solver, MuJoCo's physics", U_lin), ("MuJoCo", U_mj)):
            m = MR.metrics(cd, U)
            r[f"{name}: RMS error mm"] = m["rms_error_mm"]
            r[f"{name}: error at deepest mm"] = m.get("error_at_deepest_mm")
        r["RMS movement mm"] = MR.metrics(cd, U_ours)["rms_movement_mm"]
        rows.append(r)
        log(f"  {cd.code}: ours {r['ours: RMS error mm']:.2f}, same physics as MuJoCo {r['our solver, MuJoCo' + chr(39) + 's physics: RMS error mm']:.2f}, "
            f"MuJoCo {r['MuJoCo: RMS error mm']:.2f} mm (movement {r['RMS movement mm']:.2f} mm)")
    keys = list(rows[0].keys())
    with open(mdir / "mujoco_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = {"mujoco_nu": nu_mj, "tried": tried, "setup": asdict(setup_mj), "rows": rows}
    (mdir / "mujoco_comparison.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary, (setup_mj, d_mj, rest_mj, disp_mj), tab_lin


def profile_plot(cd: M.ClipData, preds: dict, path: Path, k=None):
    """Vertical movement along the top, middle and bottom rows at the deepest frame: measured and models."""
    plt = MR._plt()
    k = int(np.searchsorted(cd.t, cd.events["t_max"])) if k is None else k
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    styles = {"measured": dict(color="k", lw=2.2, marker="o", ms=4), "ours": dict(color="tab:orange", lw=1.6, ls="--"),
              "our solver, MuJoCo's physics": dict(color="tab:purple", lw=1.2, ls=":"), "MuJoCo": dict(color="tab:blue", lw=1.6, ls="-.")}
    for ax, (r, name) in zip(axs, ((6, "top row (5 mm below the surface)"), (3, "middle row"), (0, "bottom row"))):
        sl = slice(r * 15, (r + 1) * 15)
        x = cd.rest[sl, 0]
        ax.plot(x, cd.disp[k, sl, 1], label="measured", **styles["measured"])
        for nm, U in preds.items():
            ax.plot(x, U[k, sl, 1], label=nm, **styles[nm])
        ax.set_title(name, fontsize=10, loc="left"); ax.set_ylabel("vertical movement (mm)"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="lower left")
    axs[-1].set_xlabel("x along the face (mm)")
    fig.suptitle(f"{cd.code} at its deepest ({cd.events.get('max_depth_mm', 0):.0f} mm): each model given only the cap's depth", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def dynamic_dots(cd: M.ClipData, frames, rest_grid, disp_grid, x_fe):
    """Dot movement (T, N, 2) from MuJoCo's time-domain run (NaN outside the simulated frames)."""
    rest_fe = np.c_[cd.rest[:, 0] + (x_fe - cd.x_c), cd.rest[:, 1] * M.BLOCK[2] / cd.z_top]
    U = np.full(cd.disp.shape, np.nan)
    first = MJ.face_at(rest_grid, disp_grid[0], rest_fe)
    for n, k in enumerate(frames):
        U[k] = MJ.face_at(rest_grid, disp_grid[n], rest_fe) - first
    return U


def inversion_report(rest_grid, disp_grid, d_tab, x_fe=117.0):
    """Where MuJoCo's vertex layers under the cap end up below the layer beneath (elements turned inside out)."""
    ix = int(np.argmin(np.abs(rest_grid[:, 0, 0, 0] - x_fe)))
    out = []
    for k, dd in enumerate(d_tab):
        z = rest_grid[ix, :, :, 2] + disp_grid[k][ix, :, :, 2]          # (ny, nz) column heights across the depth
        bad = int(np.sum(np.diff(z, axis=1) < 0))
        out.append({"depth_mm": float(dd), "inverted_layer_pairs_under_cap": bad,
                    "front_column_heights_mm": [round(float(v), 1) for v in z[0]]})
    return out
