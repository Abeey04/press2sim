"""Follow-up checks on the MuJoCo baseline (README: "Why our own simulator, and not MuJoCo").

Run after `python -m press2sim model` (it reads out/model/params.json and the tracks). Needs mujoco>=4.0.

    python scripts/mujoco_checks.py --trials trials.csv --out out              # all three parts, ~45 min on 2 cores
    python scripts/mujoco_checks.py --trials trials.csv --out out --part cycles

Parts:
  cycles  press to 30 mm, hold, lift clear: force against depth, inverted elements, the dent left behind,
          for MuJoCo's default material (StVK), StVK with its J2 plasticity, and its experimental
          Stable Neo-Hookean material. All with near-hard contact so the cap cannot sink into the foam.
  fine    StVK on 13 vertex layers instead of 8, to 29 mm: is the collapse a resolution effect?
  snh     the full comparison again with Stable Neo-Hookean: Poisson's ratio picked on the training
          presses, every press scored on the frames where MuJoCo's foam surface stays under the cap,
          next to StVK and our model on the same frames; then H1 through time.
Writes out/model/mujoco_checks.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from press2sim import model as M, model_report as MR, mujoco_foam as MJ, mujoco_report as MJR  # noqa: E402
from press2sim.cli import read_trials  # noqa: E402


def load_params(out: Path) -> M.Params:
    d = json.loads((out / "model" / "params.json").read_text())["params"]
    return M.Params(**{f.name: (tuple(d[f.name]) if isinstance(d.get(f.name), list) else d[f.name])
                       for f in fields(M.Params) if f.name in d})


def base_setup(p: M.Params, nu=0.3, **kw) -> MJ.MjSetup:
    return MJ.MjSetup(E=p.E, nu=nu, damping=max(p.tau_v, 1e-4), y0=p.y0, **kw)


def part_cycles(p, log):
    res = {}
    # each material at the Poisson's ratio that fits the training presses best (StVK 0.3, Stable Neo-Hookean 0)
    for name, nu, kw in (("StVK", 0.3, {}),
                         ("StVK + J2 plasticity, yield strain 0.07", 0.3, {"yield_strain": 0.07}),
                         ("Stable Neo-Hookean", 0.0, {"material": "snh"})):
        t0 = time.time()
        r = MJ.press_cycle(base_setup(p, nu=nu, solimp=MJ.STIFF_CONTACT, **kw))
        r["poisson"] = nu
        r["wall_s"] = round(time.time() - t0)
        res[name] = r
        log(f"cycle {name}: {json.dumps(r)}")
    return res


def part_fine(p, out, trials, log):
    base = base_setup(p)                                   # the README comparison's setup (default contact)
    fine = replace(base, count=(24, 14, 13))
    depths = [x for x in M.DEPTHS if x <= 29]
    d_f, r_f, u_f = MJ.static_table(fine, 117.0, depths=depths, log=log, cache=out / "model" / "cache")
    d_b, r_b, u_b = MJ.static_table(base, 117.0, log=log, cache=out / "model" / "cache")
    res = {"inversion_13_layers": MJR.inversion_report(r_f, u_f, d_f),
           "inversion_8_layers": MJR.inversion_report(r_b, u_b, d_b)[:len(d_f)],
           "surface_depth_13_layers": MJ.surface_follow(d_f, r_f, u_f, 117.0, p.y0).round(2).tolist(),
           "surface_depth_8_layers": MJ.surface_follow(d_b, r_b, u_b, 117.0, p.y0).round(2).tolist(), "rows": []}
    clips = MR.load_set(trials, out / "tracks", lambda s: None)
    for cd in clips.values():
        if cd.kind != "press" or abs(cd.x_c - 117.0) > 12:
            continue
        fr = M.contact_frames(cd)
        fr = fr[np.nan_to_num(cd.depth[fr], nan=99) <= d_f[-1]]
        if len(fr) < 5:
            continue
        d0 = max(0.0, float(np.nan_to_num(cd.depth[cd.k0])))
        row = {"clip": cd.code, "frames": int(len(fr))}
        for nm, (dt_, rg, ug) in (("8 layers", (d_b, r_b, u_b)), ("13 layers", (d_f, r_f, u_f))):
            Uk = MJR.mj_table_at(cd, dt_, rg, ug, 117.0)
            U = M.field_at(Uk, dt_, np.clip(cd.depth[fr], 0, dt_[-1])) - M.field_at(Uk, dt_, np.array([d0]))
            row[nm + " RMS error mm"] = M.rms((U - cd.disp[fr]).ravel())
        res["rows"].append(row)
        log(f"fine grid {row}")
    return res


def _snh_table(args):
    setup, x, cache = args
    t0 = time.time()
    return setup.nu, x, MJ.static_table(setup, x, cache=cache), time.time() - t0


def part_snh(p, out, trials, log, jobs=2):
    cache = out / "model" / "cache"
    clips = MR.load_set(trials, out / "tracks", lambda s: None)
    presses = [c for c in clips.values() if c.kind == "press"]
    xs = sorted({117.0 if abs(c.x_c - 117.0) <= 12 else round(c.x_c, 1) for c in presses})
    snh = {nu: base_setup(p, nu=nu, material="snh", solimp=MJ.STIFF_CONTACT) for nu in (0.0, 0.3)}
    work = [(snh[nu], x, cache) for nu in (0.0, 0.3) for x in xs]
    tabs = {}
    with Pool(jobs) as pool:
        for nu, x, res, wall in pool.imap_unordered(_snh_table, work):
            tabs[(nu, x)] = res
            log(f"SNH table nu={nu} x={x}: {wall:.0f} s")
    # MuJoCo's default material with the same near-hard contact, so only the material differs
    svk = {x: MJ.static_table(base_setup(p, nu=0.3, solimp=MJ.STIFF_CONTACT), x, cache=cache, log=log) for x in xs}
    out_ = {"contact solimp": MJ.STIFF_CONTACT, "tables": {}}
    for (nu, x), (d_s, r_s, u_s) in sorted(tabs.items()):
        vd, s = MJ.valid_depth(d_s, r_s, u_s, x, p.y0)
        inv = MJR.inversion_report(r_s, u_s, d_s, x_fe=x)
        out_["tables"][f"nu={nu}, x={x}"] = {"valid_to_mm": vd, "surface_depth_mm": s.round(2).tolist(),
                                            "inverted_layer_pairs": [r["inverted_layer_pairs_under_cap"] for r in inv]}
    for x in xs:
        vd, s = MJ.valid_depth(*svk[x], x, p.y0)
        out_["tables"][f"StVK nu=0.3, same contact, x={x}"] = {"valid_to_mm": vd, "surface_depth_mm": s.round(2).tolist()}
    def press_errors(cd, x, keep, tables):
        d0 = max(0.0, float(np.nan_to_num(cd.depth[cd.k0])))
        res = {}
        for nm, (dt_, rg, ug) in tables.items():
            Uk = MJR.mj_table_at(cd, dt_, rg, ug, x)
            U = M.field_at(Uk, dt_, np.clip(cd.depth[keep], 0, dt_[-1])) - M.field_at(Uk, dt_, np.array([d0]))
            res[nm] = (U - cd.disp[keep]).ravel()
        return res

    def x_of(cd):
        return 117.0 if abs(cd.x_c - 117.0) <= 12 else round(cd.x_c, 1)

    # Poisson's ratio: picked on the training presses, both values scored on the same frames
    # (those where both tables are still valid), as for StVK in the main comparison
    errs = {0.0: [], 0.3: []}
    for cd in presses:
        if cd.held_out:
            continue
        x = x_of(cd)
        vd = min(MJ.valid_depth(*tabs[(nu, x)], x, p.y0)[0] for nu in (0.0, 0.3))
        fr = M.contact_frames(cd)
        keep = fr[np.nan_to_num(cd.depth[fr], nan=99.0) <= vd]
        for nu, e in press_errors(cd, x, keep, {nu: tabs[(nu, x)] for nu in (0.0, 0.3)}).items():
            errs[nu].append(e)
    for nu in (0.0, 0.3):
        out_[f"training RMS on common frames, SNH nu={nu}"] = M.rms(np.concatenate(errs[nu]))
        log(f"SNH nu={nu}: {out_[f'training RMS on common frames, SNH nu={nu}']:.3f} mm on the training presses")
    best = min((0.0, 0.3), key=lambda n: out_[f"training RMS on common frames, SNH nu={n}"])
    out_["nu chosen on training presses"] = best
    rows = []
    for cd in presses:
        x = x_of(cd)
        d_s, r_s, u_s = tabs[(best, x)]
        vd, _ = MJ.valid_depth(d_s, r_s, u_s, x, p.y0)
        fr = M.contact_frames(cd)
        keep = fr[np.nan_to_num(cd.depth[fr], nan=99.0) <= vd]
        r = {"clip": cd.code, "set": "held out" if cd.held_out else "training", "x": x,
             "max_depth_mm": float(np.nanmax(cd.depth)), "SNH valid to mm": vd,
             "frames": int(len(fr)), "frames scored": int(len(keep))}
        if len(keep) >= 5:
            e = press_errors(cd, x, keep, {"MuJoCo StVK": svk[x], "MuJoCo SNH": (d_s, r_s, u_s)})
            for nm, v in e.items():
                r[nm + " RMS error mm"] = M.rms(v)
            U_ours = np.load(out / "model" / f"pred_{cd.code}.npz")["U"]
            r["ours RMS error mm"] = M.rms((U_ours[keep] - cd.disp[keep]).ravel())
            r["RMS movement mm"] = M.rms(cd.disp[keep].ravel())
        rows.append(r)
        log(f"  {r}")
    out_["rows"] = rows
    cd = M.load_clip(out / "tracks", "H1", "press")
    t1 = time.time()
    f_dyn = cache / f"mujoco_dynamic_H1_snh_nu{best}.npz"
    if f_dyn.exists():
        z = np.load(f_dyn)
        frames, rest, disp, unstable = z["frames"], z["rest"], z["disp"], bool(z["unstable"])
    else:
        frames, rest, disp, unstable = MJ.dynamic_run(snh[best], cd, 117.0, log=log)
        np.savez_compressed(f_dyn, frames=frames, rest=rest, disp=disp, unstable=unstable)
    U = MJR.dynamic_dots(cd, frames, rest, disp, 117.0)
    np.savez_compressed(out / "model" / "pred_H1_mujoco_snh.npz", t=cd.t, U=U)
    m = MR.metrics(cd, U)
    out_["H1 time domain"] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    out_["H1 time domain"].update(unstable=bool(unstable), wall_s=round(time.time() - t1))
    log(f"SNH H1 through time: {json.dumps(out_['H1 time domain'])}")
    return out_


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--part", choices=("all", "cycles", "fine", "snh"), default="all")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    p = load_params(out)
    trials = read_trials(Path(a.trials))
    dst = out / "model" / "mujoco_checks.json"
    res = json.loads(dst.read_text()) if dst.exists() else {}
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.part in ("all", "cycles"):
        res["cycles"] = part_cycles(p, log)
        dst.write_text(json.dumps(res, indent=1, default=float))
    if a.part in ("all", "fine"):
        res["fine_grid"] = part_fine(p, out, trials, log)
        dst.write_text(json.dumps(res, indent=1, default=float))
    if a.part in ("all", "snh"):
        res["stable_neo_hookean"] = part_snh(p, out, trials, log, a.jobs)
        dst.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
