"""Checks behind the README's limitations: what the fitted model rests on, and how firmly.

Run after `python -m press2sim model` (it reads out/model/params.json and the tracks).

    python scripts/model_checks.py --trials trials.csv --out out                   # all but the slow part, ~15 min on 2 cores
    python scripts/model_checks.py --trials trials.csv --out out --part numerics   # finer grids, more solver passes, ~40 min

Parts:
  layer      the soft top layer under the cap: its thickness, and the elements pushed past 100% compression,
             at each depth
  travel     in each press, how much of the cap's travel never reaches the top-row dot under it
  offset     the pressing-frame error with the soft top, without it, and without it but with a constant taken
             off the cap's depth instead
  stiffness  E fitted to each dead weight alone, and to all four under other assumptions about the block
             (no soft top, a rigid table, neither)
  damping    what the dent under the cap after release can and cannot say: no damping against the fit, each
             press, the score over all dots, the dashpot (which neither the dent nor the moment of release
             pins down), and a second relaxing term
  numerics   the same block on finer grids, and with the element-stiffness iteration run to 80 passes
Writes out/model/model_checks.json and, with the stiffness part, out/model/stiffness_per_weight.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from press2sim import fem, fit as F, model as M, model_report as MR  # noqa: E402
from press2sim.cli import read_trials  # noqa: E402

PARTS = ("layer", "travel", "offset", "stiffness", "damping", "numerics")
FAST = PARTS[:-1]


def load_params(out: Path) -> M.Params:
    d = json.loads((out / "model" / "params.json").read_text())["params"]
    return M.Params(**{f.name: (tuple(d[f.name]) if isinstance(d.get(f.name), list) else d[f.name])
                       for f in fields(M.Params) if f.name in d})


def skin_of(p: M.Params):
    return None if p.skin_t <= 0 or p.skin_k >= 1 else (p.skin_t, p.skin_k)


def cap_x(cd) -> float:
    """Where the cap sits in the model for this press: the centre, or its own place when well off it."""
    return 117.0 if abs(cd.x_c - 117.0) <= 12 else round(cd.x_c, 1)


def no_damping(p: M.Params) -> M.Params:
    return replace(p, g=tuple(1e-9 for _ in p.g), tau_v=1e-9)


# ------------------------------------------------------------------------------------------ layer

def part_layer(p, log):
    """Node heights in the column under the cap's centre at the dotted face, depth by depth."""
    fb = fem.FoamBlock(p.nu, p.eps_y, p.h, ("cap", 117.0, p.y0, M.CAP[0], M.CAP[1]), block=M.BLOCK, fine=M.FINE,
                       support=p.support, eps_d=p.eps_d, skin=skin_of(p))
    b = fb.b
    ix = int(np.argmin(np.abs(b.x - 117.0)))
    k = int(np.searchsorted(b.z, 68.0) - 1)                       # the top dot row is 5 mm below the surface
    w = (68.0 - b.z[k]) / (b.z[k + 1] - b.z[k])
    rows = []
    for d in M.DEPTHS[1:]:
        fb.solve_depth(float(d))
        U = fb.u.reshape(*b.shape, 3)
        z = b.z + U[ix, 0, :, 2]
        e_zz = np.einsum("nij,nj->ni", fb.Bc, fb.u[fb.dof])[:, 2]
        sink = -((1 - w) * U[ix, 0, k, 2] + w * U[ix, 0, k + 1, 2])
        rows.append({"depth_mm": float(d), "top_layer_thickness_mm": float(z[-1] - z[-2]),
                     "elements_past_100pct_compression": int(np.sum(e_zz < -1.0)),
                     "top_dot_row_sink_mm": float(sink), "travel_lost_above_the_dots_mm": float(d - sink)})
        log(f"  layer at {d:4.1f} mm: {rows[-1]['top_layer_thickness_mm']:+.2f} mm thick, "
            f"{rows[-1]['elements_past_100pct_compression']} of {fb.n_el} elements past 100%, "
            f"{rows[-1]['travel_lost_above_the_dots_mm']:.1f} mm of travel lost above the dots")
    return {"layer_mm": p.skin_t, "layer_stiffness": p.skin_k, "elements": int(fb.n_el), "by_depth": rows}


# ------------------------------------------------------------------------------------------ travel

def part_travel(presses, log):
    """At each press's deepest frame: the cap's depth minus the sink of the top-row dot under it."""
    rows = []
    for cd in presses:
        k = int(np.searchsorted(cd.t, cd.events["t_max"]))
        j = F.TOP_ROW[np.nanargmin(np.abs(cd.rest[F.TOP_ROW, 0] - cd.x_c))]
        sink = -float(cd.disp[k, j, 1])
        rows.append({"clip": cd.code, "cap_depth_mm": float(cd.depth[k]), "dot_sink_mm": sink,
                     "travel_lost_mm": float(cd.depth[k]) - sink})
        log(f"  {cd.code}: cap {rows[-1]['cap_depth_mm']:.1f} mm deep, the dot under it sank {sink:.1f} mm, "
            f"{rows[-1]['travel_lost_mm']:.1f} mm lost")
    lost = [r["travel_lost_mm"] for r in rows]
    return {"rows": rows, "lost_min_mm": min(lost), "lost_max_mm": max(lost), "lost_mean_mm": float(np.mean(lost))}


# ------------------------------------------------------------------------------------------ offset

def pressing_error(tab, train, offset_mm=0.0):
    """fit.shape_loss with a constant taken off the cap's depth."""
    errs = []
    for cd in train:
        Uk = tab.at(cd.rest, cd.x_c, cd.z_top)
        fr = M.contact_frames(cd)
        d_rest = max(0.0, float(np.nan_to_num(cd.depth[cd.k0])) - offset_mm)
        U = M.field_at(Uk, tab.d, cd.depth[fr] - offset_mm) - M.field_at(Uk, tab.d, np.array([d_rest]))
        errs.append((U - cd.disp[fr]).ravel())
    return M.rms(np.concatenate(errs))


def part_offset(p, train, cache, log):
    tab = M.build_table(p, ("cap", 117.0), cache=cache)
    bare = M.build_table(replace(p, skin_t=0.0, skin_k=1.0), ("cap", 117.0), cache=cache)
    res = {"with_soft_top_mm": pressing_error(tab, train), "without_soft_top_mm": pressing_error(bare, train),
           "without_soft_top_by_offset_mm": {f"{o:g}": pressing_error(bare, train, o) for o in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)}}
    best = min(res["without_soft_top_by_offset_mm"].items(), key=lambda kv: kv[1])
    res["best_offset_mm"], res["without_soft_top_best_offset_mm"] = float(best[0]), best[1]
    log(f"  pressing-frame error: {res['with_soft_top_mm']:.3f} mm with the soft top, {res['without_soft_top_mm']:.3f} mm "
        f"without, {best[1]:.3f} mm without it and {best[0]} mm off the cap's depth")
    return res


# ------------------------------------------------------------------------------------------ stiffness

def part_stiffness(p, weights, cache, out, log):
    ptab = M.build_table(p, ("plate",), cache=cache)
    alone = {}
    for cd in weights:
        E, rms_mm, per, _ = F.fit_E([cd], ptab, p)
        alone[cd.code] = {"E_kPa": E / 1e3, "rms_mm": rms_mm, **{k: per[cd.code][k] for k in
                          ("load_kg", "t_on_s", "t_off_s", "sink_measured_mm", "sink_model_mm")}}
        log(f"  {cd.code} alone ({cd.load_kg:g} kg): E = {E / 1e3:.0f} kPa")
    (out / "model" / "stiffness_per_weight.json").write_text(json.dumps(alone, indent=1))
    variants = {}
    for name, q in (("as fitted", p), ("no soft top", replace(p, skin_t=0.0, skin_k=1.0)),
                    ("rigid table", replace(p, support=1e3)),
                    ("no soft top, rigid table", replace(p, skin_t=0.0, skin_k=1.0, support=1e3))):
        pt = M.build_table(q, ("plate",), cache=cache)
        E, rms_mm, per, _ = F.fit_E(weights, pt, q)
        ct = M.build_table(q, ("cap", 117.0), cache=cache)
        variants[name] = {"E_kPa": E / 1e3, "weights_rms_mm": rms_mm,
                          "cap_force_N": {f"{d:g} mm": E * float(np.interp(d, ct.d, ct.R)) for d in (5.0, 10.0, 20.0, 30.0)},
                          "sink_model_mm": {c: v["sink_model_mm"] for c, v in per.items()},
                          "sink_measured_mm": {c: v["sink_measured_mm"] for c, v in per.items()}}
        log(f"  {name}: E = {E / 1e3:.0f} kPa, weights fitted to {rms_mm:.3f} mm, "
            f"{variants[name]['cap_force_N']['30 mm']:.0f} N to push the cap 30 mm in")
    # where the soft top starts to buckle under the pad
    area = M.BLOCK[0] * M.BLOCK[1] * 1e-6
    kg = p.eps_y * p.skin_k * p.E * area / M.G_ACC if skin_of(p) else None
    return {"each_weight_alone": alone, "all_four_weights": variants, "soft_top_buckles_under_the_pad_at_kg": kg}


# ------------------------------------------------------------------------------------------ damping

def dent_score(p, clips, tabs):
    return M.rms(np.concatenate([F.dent_error(cd, M.predict_clip(cd, tabs[cd.code], p)[0]).ravel() for cd in clips]))


def dent_left(cd, U, after_s=1.0):
    """Sink (mm) of the dots under the cap, `after_s` after the foam lets go: predicted and measured."""
    tr = cd.events["t_release"]
    sel = (cd.t > tr + after_s - 0.05) & (cd.t < tr + after_s + 0.05)
    j = F.dent_dots(cd)
    return -float(np.nanmedian(U[sel][:, j, 1])), -float(np.nanmedian(cd.disp[sel][:, j, 1]))


def release_time(cd, dz):
    """When the foam lets go of the cap, by the pipeline's rule (indenter.events): the gap between the cap
    and the top-row dot under it has grown 2 mm past its value at the deepest frame. dz: vertical movement
    (T, N) of the dots, measured or predicted."""
    j = F.TOP_ROW[np.nanargmin(np.abs(cd.rest[F.TOP_ROW, 0] - cd.x_c))]
    gap = (cd.z_top - cd.depth) - (cd.rest[j, 1] + dz[:, j])
    k_deep = int(np.searchsorted(cd.t, cd.events["t_max"]))
    after = np.arange(int(np.searchsorted(cd.t, cd.events["t_lift"])), len(cd.t))
    grown = after[np.isfinite(gap[after]) & (gap[after] > gap[k_deep] + 2.0)]
    return float(cd.t[grown[0]]) if len(grown) else np.nan


def fit_terms(p, train, tab, g0, tau0, free_tau):
    """Relaxing fractions (and the times marked free) fitted to the training dent; the dashpot stays as it is."""
    n = len(g0)
    free = [i for i in range(n) if free_tau[i]]

    def unpack(v):
        g = 1 / (1 + np.exp(-v[:n]))
        tau = np.array(tau0, float)
        tau[free] = np.exp(v[n:])
        return replace(p, g=tuple(map(float, g)), tau=tuple(map(float, tau)))

    v0 = np.r_[np.log(np.array(g0) / (1 - np.array(g0))), np.log(np.array(tau0)[free])]
    res = minimize(lambda v: F.time_loss(unpack(v), train, tab), v0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 2e-3, "fatol": 2e-5})
    return unpack(res.x), float(res.fun)


def part_damping(p, train, held, cache, log):
    tabs = {cd.code: M.build_table(p, ("cap", cap_x(cd)), cache=cache) for cd in train + held}
    tab = M.build_table(p, ("cap", 117.0), cache=cache)
    none = no_damping(p)
    res = {"fitted": {"g": list(p.g), "tau_s": list(p.tau), "tau_v_ms": p.tau_v * 1e3},
           "dent_rms_mm": {"training": {"no damping": dent_score(none, train, tabs), "fitted": dent_score(p, train, tabs)},
                           "held out": {"no damping": dent_score(none, held, tabs), "fitted": dent_score(p, held, tabs)}},
           "all_dots_around_the_lift_rms_mm": {"no damping": F.recovery_loss(none, train, tab),
                                               "fitted": F.recovery_loss(p, train, tab)}}
    d = res["dent_rms_mm"]
    log(f"  dent under the cap after release: training {d['training']['no damping']:.3f} mm with no damping, "
        f"{d['training']['fitted']:.3f} mm fitted; held out {d['held out']['no damping']:.3f} and {d['held out']['fitted']:.3f} mm")
    a = res["all_dots_around_the_lift_rms_mm"]
    log(f"  all 105 dots around the lift (training): {a['no damping']:.4f} mm with no damping, {a['fitted']:.4f} mm fitted")
    # each press, and the dent left one second after release
    rows = []
    for cd in train + held:
        U = M.predict_clip(cd, tabs[cd.code], p)[0]
        U0 = M.predict_clip(cd, tabs[cd.code], none)[0]
        left = dent_left(cd, U)
        rows.append({"clip": cd.code, "set": "held out" if cd.held_out else "training",
                     "max_depth_mm": cd.events.get("max_depth_mm"), "seconds_after_release": cd.t_end - cd.events["t_release"],
                     "dent_error_no_damping_mm": M.rms(F.dent_error(cd, U0)), "dent_error_mm": M.rms(F.dent_error(cd, U)),
                     "dent_1s_after_release_model_mm": left[0], "dent_1s_after_release_measured_mm": left[1],
                     "dent_1s_after_release_no_damping_mm": dent_left(cd, U0)[0]})
        log(f"  {cd.code}: dent error {rows[-1]['dent_error_no_damping_mm']:.3f} mm with no damping, "
            f"{rows[-1]['dent_error_mm']:.3f} mm fitted; one second on {left[1]:.2f} mm measured, {left[0]:.2f} mm in the model")
    res["rows"] = rows
    # the dashpot, which is not fitted: what the dent and the moment of release say about it
    dashpots = sorted({0.0, p.tau_v, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100})
    res["dent_rms_training_by_dashpot_ms"] = {f"{1e3 * tv:g}": dent_score(replace(p, tau_v=max(tv, 1e-9)), train, tabs)
                                              for tv in dashpots}
    log("  dashpot (ms) against the training dent error: " +
        ", ".join(f"{k}: {v:.4f}" for k, v in res["dent_rms_training_by_dashpot_ms"].items()))
    measured = {cd.code: release_time(cd, cd.disp[..., 1]) for cd in train}
    res["release_model_minus_measured_ms_by_dashpot_ms"] = {}
    for tv in dashpots:
        q = replace(p, tau_v=max(tv, 1e-9))
        late = {cd.code: 1e3 * (release_time(cd, M.predict_clip(cd, tabs[cd.code], q)[0][..., 1]) - measured[cd.code])
                for cd in train}
        res["release_model_minus_measured_ms_by_dashpot_ms"][f"{1e3 * tv:g}"] = {
            "median_ms": float(np.nanmedian(list(late.values()))), "per_clip_ms": late}
    log("  dashpot (ms) against how late the model's foam lets go of the cap (median over the training presses, ms): " +
        ", ".join(f"{k}: {v['median_ms']:+.0f}" for k, v in res["release_model_minus_measured_ms_by_dashpot_ms"].items()))
    # a second, slower relaxing term: all four numbers free, then its time held at a few values
    two, loss = fit_terms(p, train, tab, (p.g[0], 0.05), (p.tau[0], 2.0), (True, True))
    res["second_term"] = {"free": {"g": list(two.g), "tau_s": list(two.tau), "dent_rms_training_mm": loss,
                                   "dent_rms_held_out_mm": dent_score(two, held, tabs)}, "slow_time_held_at_s": {}}
    log(f"  two terms, all free: g {np.round(two.g, 3).tolist()}, tau {np.round(two.tau, 2).tolist()} s, "
        f"training {loss:.4f} mm, held out {res['second_term']['free']['dent_rms_held_out_mm']:.4f} mm")
    for t2 in (2.0, 3.0, 5.0, 10.0):
        q, loss = fit_terms(p, train, tab, (two.g[0], two.g[1]), (two.tau[0], t2), (True, False))
        res["second_term"]["slow_time_held_at_s"][f"{t2:g}"] = {"g": list(q.g), "tau_s": list(q.tau),
                                                               "dent_rms_training_mm": loss,
                                                               "dent_rms_held_out_mm": dent_score(q, held, tabs)}
        log(f"  two terms, the slow one held at {t2:g} s: g {np.round(q.g, 3).tolist()}, training {loss:.4f} mm")
    return res


# ------------------------------------------------------------------------------------------ numerics

def cap_table(p, fine=M.FINE, iters=10, tol=0.03):
    """model.build_table for the cap at the centre, with the grid spacing and the iteration's limits exposed."""
    fb = fem.FoamBlock(p.nu, p.eps_y, p.h, ("cap", 117.0, p.y0, M.CAP[0], M.CAP[1]), block=M.BLOCK, fine=fine,
                       support=p.support, eps_d=p.eps_d, skin=skin_of(p))
    R, face, unsettled = [0.0], [np.zeros((len(fb.b.x), len(fb.b.z), 3))], [0.0]
    for d in M.DEPTHS[1:]:
        R.append(fb.solve_depth(float(d), iters=iters, tol=tol))
        face.append(fb.face_nodes().copy())
        ec = fb.strain_c(fb.u)
        target = fb.alpha * np.where(ec > 1e-9, fem.foam_stress(ec, fb.eps_y, fb.h, fb.eps_d) / np.maximum(ec, 1e-9), 1.0)
        unsettled.append(float(np.max(np.abs(np.log(target / fb.E)))))
    R = np.maximum.accumulate(np.array(R)) + np.arange(len(R)) * 1e-12
    return M.Table(M.DEPTHS.copy(), R, np.stack(face), fb.b.x, fb.b.z, 117.0), unsettled, [int(n) for n in fb.b.shape]


def part_numerics(p, train, held, log):
    centred = [cd for cd in held if cap_x(cd) == 117.0]
    res = {"grid": {}, "passes": {}}

    def summary(tab, unsettled, shape, seconds):
        return {"grid_nodes": shape, "seconds": round(seconds),
                "pressing_error_mm": pressing_error(tab, train),
                "rms_error_mm": {cd.code: MR.metrics(cd, M.predict_clip(cd, tab, p)[0])["rms_error_mm"] for cd in centred},
                "cap_force_N": {f"{d:g} mm": p.E * float(np.interp(d, tab.d, tab.R)) for d in (5.0, 10.0, 20.0, 30.0)},
                "largest_unsettled_stiffness_ratio_log": dict(zip((f"{d:g} mm" for d in M.DEPTHS), unsettled))}

    for fine in (7.5, 5.0, 3.75):
        t0 = time.time()
        s = summary(*cap_table(p, fine=fine), time.time() - t0)
        res["grid"][f"{fine:g} mm"] = s
        log(f"  grid {fine:g} mm {s['grid_nodes']}: pressing-frame error {s['pressing_error_mm']:.3f} mm, "
            f"{s['rms_error_mm']}, forces {({k: round(v, 1) for k, v in s['cap_force_N'].items()})} ({s['seconds']} s)")
    for iters, tol in ((10, 0.03), (80, 0.003)):
        t0 = time.time()
        s = summary(*cap_table(p, iters=iters, tol=tol), time.time() - t0)
        res["passes"][f"{iters}"] = s
        worst = max(s["largest_unsettled_stiffness_ratio_log"].values())
        log(f"  up to {iters} passes: pressing-frame error {s['pressing_error_mm']:.3f} mm, {s['rms_error_mm']}, "
            f"forces {({k: round(v, 1) for k, v in s['cap_force_N'].items()})}, worst unsettled element {worst:.2f} ({s['seconds']} s)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--part", choices=("fast", "all") + PARTS, default="fast",
                    help="fast: everything but numerics (the default)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)       # all-NaN frames where a dot was briefly hidden
    out = Path(a.out)
    p = load_params(out)
    cache = out / "model" / "cache"
    clips = MR.load_set(read_trials(Path(a.trials)), out / "tracks", None)
    presses = [c for c in clips.values() if c.kind == "press"]
    train = [c for c in presses if not c.held_out]
    held = [c for c in presses if c.held_out]
    weights = [c for c in clips.values() if c.kind == "weight" and c.load_kg and c.load_kg >= 0.9]
    dst = out / "model" / "model_checks.json"
    res = json.loads(dst.read_text()) if dst.exists() else {}
    log = lambda s: print(s, flush=True)  # noqa: E731
    todo = FAST if a.part == "fast" else PARTS if a.part == "all" else (a.part,)
    run = {"layer": lambda: part_layer(p, log), "travel": lambda: part_travel(presses, log),
           "offset": lambda: part_offset(p, train, cache, log),
           "stiffness": lambda: part_stiffness(p, weights, cache, out, log),
           "damping": lambda: part_damping(p, train, held, cache, log),
           "numerics": lambda: part_numerics(p, train, held, log)}
    for name in todo:
        log(f"{name}:")
        t0 = time.time()
        res[name] = run[name]()
        dst.write_text(json.dumps(res, indent=1, default=float))
        log(f"  ({time.time() - t0:.0f} s)")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
