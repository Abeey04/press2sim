"""Command line: python -m press2sim <command> ...

  times   RAW_DIR                      frame timing of every clip (dropped frames, true rate)
  calib   --trials trials.csv          checkerboard clips -> out/calib/<code>.json (+ C1 vs C2 check)
  track   --trials trials.csv          every non-calibration clip -> out/tracks/<code>.npz/.json + plots
  report  --trials trials.csv          noise floor, repeatability, summary table
  all     --trials trials.csv          calib, track, report in order
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

from .calib import Calibration, calibrate_clip, compare, face_correction
from .config import load_config
from .pipeline import KINDS, process_clip
from .video import frame_times

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mkv"}


def read_trials(path: Path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            if not r.get("code") or r["code"].startswith("#"):
                continue
            if not r.get("file"):
                print(f"note: {r['code']} has no file name in the trials file; skipping it")
                continue
            if r.get("kind") not in KINDS:
                raise SystemExit(f"trials: {r['code']} has kind '{r.get('kind')}', expected one of {KINDS}")
            rows.append(r)
    return rows


def _video(raw: Path, name: str) -> Path:
    p = raw / name
    if not p.exists():
        raise SystemExit(f"missing video {p}")
    return p


def cmd_times(a):
    raw = Path(a.raw)
    vids = sorted(p for p in raw.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        raise SystemExit(f"no videos in {raw}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'file':28s} {'frames':>7s} {'dur s':>6s} {'fps(avg)':>8s} {'fps(gap)':>8s} {'long gaps':>9s} {'max gap ms':>10s} rot")
    for v in vids:
        s = frame_times(v).summary()
        rows.append({"file": v.name, **s})
        print(f"{v.name:28s} {s['n_frames']:7d} {s['duration_s']:6.2f} {s['fps_average']:8.1f} "
              f"{s['fps_from_median_gap']:8.1f} {s['n_long_gaps']:9d} {s['max_gap_ms']:10.2f} {s['rotation_ccw']}")
    with open(out / "frame_times.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out / 'frame_times.csv'}. Timestamps from the file are used everywhere; "
          "'long gaps' are dropped frames and are harmless.")


def cmd_calib(a, cfg, trials):
    raw, out = Path(a.raw), Path(a.out)
    cals = {}
    from . import qa
    for r in trials:
        if r["kind"] != "calib":
            continue
        p = _video(raw, r["file"])
        print(f"calibrating {r['code']} ({p.name}) ...", flush=True)
        c = calibrate_clip(p, cfg, rotation_ccw=a.rotate)
        c.report["clip"] = r["code"]
        c.save(out / "calib" / f"{r['code']}.json")
        qa.calib_plot(c, p, out / "qa" / f"calib_{r['code']}.png", rotation_ccw=a.rotate)
        rep = c.report
        print(f"  board in {rep['frames_with_board']} sampled frames, averaged {rep['frames_averaged']}; "
              f"fit RMS {rep['fit_rms_mm']:.3f} mm, hold-out RMS {rep['holdout_rms_mm']:.3f} mm "
              f"(max {rep['holdout_max_mm']:.3f}); squares {rep['square_mm_measured_x']:.3f} x "
              f"{rep['square_mm_measured_z']:.3f} mm; {rep['px_per_mm']:.2f} px/mm; rotation {rep['rotation_ccw']}")
        if rep["holdout_rms_mm"] > 0.2:
            print("  WARNING: hold-out error above 0.2 mm. Was the card flat against the face and fully in view?")
        # heights on the face, checked against rulers on the foam itself (bare face in the same clip)
        statics = [_video(raw, x["file"]) for x in trials if x["kind"] == "static"]
        for src in [p] + statics:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # frames with the card in front: no dots
                    f = face_correction(c, cfg, src, frame_times(src).dt_median,
                                        row_span_mm=cfg.grid.measured_row_span_mm or None)
                break
            except RuntimeError as e:
                print(f"  face check on {src.name} failed: {e}")
                f = None
        if f is not None:
            c.report["face"] = f
            c.save(out / "calib" / f"{r['code']}.json")
            m = f["measured_card_mm"]
            refs = ", ".join(f"{k.replace('_', ' ')} x{v:.4f}" for k, v in f["references"].items())
            print(f"  bare face ({f['from_clip']}): block {m['length']:.1f} x {m['z_top'] - m['z_base']:.1f} mm "
                  f"(measured {cfg.block.length_mm:g} x {cfg.block.height_mm:g}), dot rows {m['dot_rows_span']:.2f} mm "
                  f"apart (ruler {cfg.grid.measured_row_span_mm:g}); base at z={m['z_base']:+.2f} mm on the card")
            print(f"  -> heights on the face scaled x{f['a_z']:.4f} ({refs}; spread {f['spread_pct'] or 0:.1f}%), "
                  f"widths left as they are (length check {f['length_check_pct']:+.1f}%)")
        cals[r["code"]] = c
    codes = list(cals)
    if len(codes) >= 2:
        cmp = compare(cals[codes[0]], cals[codes[-1]])
        (out / "calib" / "compare.json").write_text(json.dumps(cmp, indent=1))
        print(f"  {codes[0]} vs {codes[-1]}: card shifted {cmp['card_shift_x_mm']:+.2f} mm sideways (harmless), "
              f"{cmp['card_shift_z_mm']:+.2f} mm vertically, scale {cmp['scale_change_pct']:+.2f}%, "
              f"tilt {cmp['tilt_change_deg']:+.3f} deg, residual {cmp['residual_rms_mm']:.3f} mm -> {cmp['verdict']}")
    return cals


def _load_calib(out: Path, code: str) -> Calibration:
    f = out / "calib" / f"{code}.json"
    if not f.exists():
        raise SystemExit(f"no calibration {f}; run the calib command first")
    return Calibration.load(f)


def _track_one(args):
    path, code, kind, calib_path, cfg, out, rotate, plots = args
    t0 = time.time()
    try:
        m = process_clip(path, code, kind, Calibration.load(calib_path), cfg, out, rotation_ccw=rotate,
                         make_plots=plots, log=lambda *x: None)
        return code, kind, m, None, time.time() - t0
    except Exception as e:  # keep going; report at the end
        return code, kind, None, f"{type(e).__name__}: {e}", time.time() - t0


def _print_track(code, kind, m, err, dt):
    if err:
        print(f"{code:4s} {kind:9s} FAILED: {err}")
        return
    ev = m.get("events", {})
    depth = f", max depth {ev['max_depth_mm']:.1f} mm" if ev and ev.get("max_depth_mm") == ev.get("max_depth_mm") else ""
    ind = m.get("indenter")
    bottle = f", cap seen {100 * ind['cap_dot_fraction']:.0f}%, both dots {100 * ind['both_dots_fraction']:.0f}%, tilt <= {ind['tilt_deg_max']:.1f} deg" if ind else ""
    print(f"{code:4s} {kind:9s} rest at {m['rest']['t']:.2f} s, {m['grid']['found']}/105 dots, tracked "
          f"{100 * m['valid_fraction_after_rest']:.1f}%, rest noise {(m['rest']['noise_rms_mm'] or float('nan')):.3f} mm, "
          f"camera shift <= {m['camera_still'].get('max_shift_px', float('nan')):.2f} px{depth}{bottle} ({dt:.0f} s)",
          flush=True)


def cmd_track(a, cfg, trials):
    raw, out = Path(a.raw), Path(a.out)
    cal_code = a.calib or next((r["code"] for r in trials if r["kind"] == "calib"), None)
    if cal_code is None:
        raise SystemExit("no calibration clip in the trials file")
    calib_path = out / "calib" / f"{cal_code}.json"
    if not calib_path.exists():
        raise SystemExit(f"no calibration {calib_path}; run the calib command first")
    only = set(a.only.split(",")) if a.only else None
    jobs = [(_video(raw, r["file"]), r["code"], r["kind"], calib_path, cfg, out, a.rotate, not a.no_plots)
            for r in trials if r["kind"] != "calib" and (not only or r["code"] in only)]
    print(f"tracking {len(jobs)} clips with calibration {cal_code}, {a.jobs} at a time ...", flush=True)
    failures = []
    if a.jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # spawn, not fork: forking after the video and OpenCV thread pools start can deadlock the workers
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
            for f in as_completed([ex.submit(_track_one, j) for j in jobs]):
                res = f.result(); _print_track(*res)
                if res[3]: failures.append(res[:1] + res[3:4])
    else:
        for j in jobs:
            res = _track_one(j); _print_track(*res)
            if res[3]: failures.append(res[:1] + res[3:4])
    if failures:
        print("\nfailed clips:")
        for c, e in failures:
            print(f"  {c}: {e}")


def cmd_report(a, cfg, trials):
    from . import qa
    out = Path(a.out)
    td = out / "tracks"
    static = [r["code"] for r in trials if r["kind"] == "static"]
    noise = qa.noise_floor(td, static)
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs else []
    rep = qa.repeatability(td, pairs)
    rows = qa.summary_csv(td, out / "summary.csv")
    res = {"noise_floor": noise, "repeatability": rep}
    cmpf = out / "calib" / "compare.json"
    if cmpf.exists():
        res["calibration_compare"] = json.loads(cmpf.read_text())
    (out / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\nsummary of {len(rows)} clips: {out / 'summary.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="press2sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("times"); t.add_argument("raw"); t.add_argument("--out", default="out")
    for name in ("calib", "track", "report", "all", "model", "simulate", "mujoco"):
        s = sub.add_parser(name)
        s.add_argument("--trials", required=True, help="CSV with columns code,file,kind,...")
        s.add_argument("--raw", default="raw", help="folder with the original clips")
        s.add_argument("--out", default="out")
        s.add_argument("--config", default=None, help="JSON overrides for config.py values")
        s.add_argument("--rotate", type=int, default=None, help="force rotation (deg, counter-clockwise)")
        s.add_argument("--calib", default=None, help="calibration code to use for tracking (default: first calib row)")
        s.add_argument("--only", default=None, help="comma-separated clip codes")
        s.add_argument("--no-plots", action="store_true")
        s.add_argument("--pairs", default="R1:T05,R2:T08", help="repeat pairs for the repeatability floor")
        s.add_argument("--jobs", type=int, default=1, help="clips tracked in parallel (try 4)")
        s.add_argument("--no-video", action="store_true", help="model: skip the comparison video")
        s.add_argument("--keep-shape", action="store_true",
                       help="model: keep the shape and support from out/model/start.json, refit only damping and E")
    a = ap.parse_args(argv)
    if a.cmd == "times":
        return cmd_times(a)
    cfg = load_config(a.config)
    trials = read_trials(Path(a.trials))
    if a.cmd in ("calib", "all"):
        cmd_calib(a, cfg, trials)
    if a.cmd in ("track", "all"):
        cmd_track(a, cfg, trials)
    if a.cmd in ("report", "all"):
        cmd_report(a, cfg, trials)
    if a.cmd == "model":
        cmd_model(a, cfg, trials)
    if a.cmd == "simulate":
        cmd_simulate(a, cfg, trials)
    if a.cmd == "mujoco":
        cmd_mujoco(a, cfg, trials)


def cmd_mujoco(a, cfg, trials):
    """The same block in MuJoCo, scored like our model; a time-domain run and video for a held-out press."""
    from dataclasses import fields
    import numpy as np
    from . import animate, mujoco_foam as MJ, mujoco_report as MJR
    from .model import Params, load_clip
    out = Path(a.out)
    d = json.loads((out / "model" / "params.json").read_text())["params"]
    p = Params(**{f.name: (tuple(d[f.name]) if isinstance(d.get(f.name), list) else d[f.name]) for f in fields(Params) if f.name in d})
    summary, (setup, d_mj, rest_mj, disp_mj), tab_lin = MJR.run(trials, out, p, nus=(0.0, 0.3))
    summary["inversion"] = MJR.inversion_report(rest_mj, disp_mj, d_mj)
    (out / "model" / "mujoco_comparison.json").write_text(json.dumps(summary, indent=1, default=float))
    calib_code = a.calib or next(r["code"] for r in trials if r["kind"] == "calib")
    calib = Calibration.load(out / "calib" / f"{calib_code}.json")
    rot = calib.report.get("rotation_ccw") if a.rotate is None else a.rotate
    files = {r["code"]: r["file"] for r in trials}
    for code in (a.only.split(",") if a.only else ["H1"]):
        cd = load_clip(out / "tracks", code, "press")
        x_fe = 117.0 if abs(cd.x_c - 117.0) <= 12 else round(cd.x_c, 1)
        f = out / "model" / "cache" / f"mujoco_dynamic_{code}.npz"
        if f.exists():
            z = np.load(f); frames, rest, disp = z["frames"], z["rest"], z["disp"]
        else:
            print(f"MuJoCo time-domain run for {code} (about 130x slower than real time) ...")
            frames, rest, disp, _ = MJ.dynamic_run(setup, cd, x_fe, log=print)
            np.savez_compressed(f, frames=frames, rest=rest, disp=disp)
        U_mj = MJR.dynamic_dots(cd, frames, rest, disp, x_fe)
        pred = np.load(out / "model" / f"pred_{code}.npz")
        from . import model_report as MR
        U_lin, _ = MJR.quasi_static(cd, tab_lin.at(cd.rest, cd.x_c, cd.z_top), tab_lin.d)
        MJR.profile_plot(cd, {"ours": pred["U"], "our solver, MuJoCo's physics": U_lin, "MuJoCo": U_mj},
                         out / "model" / f"{code}_vs_mujoco_profiles.png")
        np.savez_compressed(out / "model" / f"pred_{code}_mujoco.npz", t=cd.t, U=U_mj)
        dst = out / "model" / f"{code}_ours_vs_mujoco.mp4"
        print(f"video {dst.name} ...")
        animate.compare_video(code, out, _video(Path(a.raw), files[code]), calib, rot, dst, p, (frames, rest, disp), U_mj,
                              out / "model" / "cache")


def cmd_simulate(a, cfg, trials):
    """Side-by-side videos: the real press and the simulated 3D block (after `model`)."""
    from dataclasses import fields
    from .animate import simulate_video
    from .model import Params
    out = Path(a.out)
    d = json.loads((out / "model" / "params.json").read_text())["params"]
    p = Params(**{f.name: (tuple(d[f.name]) if isinstance(d.get(f.name), list) else d[f.name]) for f in fields(Params) if f.name in d})
    calib_code = a.calib or next(r["code"] for r in trials if r["kind"] == "calib")
    calib = Calibration.load(out / "calib" / f"{calib_code}.json")
    rot = calib.report.get("rotation_ccw") if a.rotate is None else a.rotate
    files = {r["code"]: r["file"] for r in trials}
    codes = a.only.split(",") if a.only else [r["code"] for r in trials if r["kind"] == "press"
                                               and str(r.get("held_out", "")).lower() == "yes"]
    for code in codes:
        dst = out / "model" / f"{code}_simulation.mp4"
        print(f"simulating {code} -> {dst.name}")
        simulate_video(code, out, _video(Path(a.raw), files[code]), calib, rot, dst, p, out / "model" / "cache", step=5)


def cmd_model(a, cfg, trials):
    """Fit the 3D block model on the training clips and predict the held-out ones."""
    from . import model_report as MR
    out = Path(a.out)
    start = None
    sfile = out / "model" / "start.json"
    if sfile.exists():                       # warm start from an earlier fit
        from dataclasses import fields
        from .model import Params
        d = json.loads(sfile.read_text())
        start = Params(**{f.name: (tuple(d[f.name]) if isinstance(d.get(f.name), list) else d[f.name])
                          for f in fields(Params) if f.name in d})
        print(f"starting from {sfile}")
    p, report, preds, clips = MR.run(trials, out, Path(a.raw), log=print, video=False, start=start,
                                     refit_shape=not (a.keep_shape and start is not None))
    if a.no_video:
        return
    calib_code = a.calib or next(r["code"] for r in trials if r["kind"] == "calib")
    calib = Calibration.load(out / "calib" / f"{calib_code}.json")
    rot = calib.report.get("rotation_ccw") if a.rotate is None else a.rotate
    files = {r["code"]: r["file"] for r in trials}
    for cd in clips.values():
        if cd.held_out and cd.kind == "press":
            meta = json.loads((out / "tracks" / f"{cd.code}.json").read_text())
            U, q, Fz, c, U0 = preds[cd.code]
            dst = out / "model" / f"{cd.code}_model_vs_video.mp4"
            print(f"video {dst.name} ...")
            MR.make_video(cd, U, _video(Path(a.raw), files[cd.code]), calib, rot, meta, dst, q=q)


if __name__ == "__main__":
    sys.exit(main())
