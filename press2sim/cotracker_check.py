"""Tracking with CoTracker3, a pretrained point tracker (needs PyTorch; a GPU is strongly advised).

Two uses:

- A cross-check on the centroid tracker, which is the measurement. This runs CoTracker3 on the same clip
  from the same rest-frame dot positions and reports how far the two disagree (in mm). A large
  disagreement on a dot flags a tracking problem worth looking at.

      python -m press2sim.cotracker_check --code T05 --raw raw --out out [--device cuda]

  Or against the truth, on the rendered clips of scripts/make_synthetic.py (after the pipeline has run
  on them), which shows which of the two trackers is right where they disagree:

      python -m press2sim.cotracker_check --synthetic syn

- A stand-in for the centroid tracker in the whole pipeline: `as_tracker` gives `pipeline.process_clip`
  the same points followed by CoTracker instead. `notebooks/03_walkthrough.ipynb` does this to show what an
  off-the-shelf tracker gives, and what the purpose-built one adds.

CoTracker is CC-BY-NC licensed; credit it in the README if you report its numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .calib import Calibration
from .video import Clip, frame_times


def run(model, frames, queries_xy, step, device, torch):
    """Feed frames (iterator of HxWx3 uint8 RGB) to an online CoTracker; return (T, N, 2) tracks and (T, N) visibility."""
    frames = iter(frames)
    cache, loaded = {}, 0

    def get(i):
        nonlocal loaded
        while loaded <= i:
            try:
                cache[loaded] = next(frames)
            except StopIteration:
                return None
            loaded += 1
        return cache.get(i)

    def chunk(a, b):
        imgs = [f for f in (get(i) for i in range(a, b)) if f is not None]
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].float().to(device)
        return x

    q = torch.tensor(np.c_[np.zeros(len(queries_xy)), queries_xy], dtype=torch.float32, device=device)[None]
    model(video_chunk=chunk(0, 2 * step), is_first_step=True, queries=q, add_support_grid=True)
    tracks = vis = None
    ind = 0
    while True:
        if get(ind + step) is None:          # fewer than step frames left: done
            break
        tracks, vis = model(video_chunk=chunk(ind, ind + 2 * step))
        for k in [k for k in cache if k < ind + step]:
            del cache[k]
        ind += step
    if tracks is None:
        raise RuntimeError("clip too short for CoTracker")
    n = len(queries_xy)          # the support grid's points come back after the queries: drop them
    tr = tracks[0].cpu().numpy()
    return tr[:, :n], vis[0].cpu().numpy().astype(bool).reshape(tr.shape[0], -1)[:, :n]


def load_model(device="cuda"):
    """CoTracker3's online model from PyTorch Hub (downloaded on first use, about 100 MB)."""
    import torch
    return torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", verbose=False).to(device)


def track_points(clip: Clip, start: int, pts_px: np.ndarray, model, device="cuda", max_side=768, pad=80, every=1):
    """Follow pts_px (N, 2), given in frame `start`, to the end of the clip with an online CoTracker.

    Only the region around the points is given to the model (three pads above them, where the bottle
    comes from). every: give the model one frame in `every` and fill the frames between by linear
    interpolation in time; at 240 fps the dots barely move from one frame to the next.
    Returns positions (T, N, 2) in full-frame pixels and visibility (T, N), T counted from `start`.
    """
    import torch
    x0 = int(max(0, np.nanmin(pts_px[:, 0]) - pad)); x1 = int(np.nanmax(pts_px[:, 0]) + pad)
    y0 = int(max(0, np.nanmin(pts_px[:, 1]) - 3 * pad)); y1 = int(np.nanmax(pts_px[:, 1]) + pad)
    s = min(1.0, max_side / max(x1 - x0, y1 - y0))

    def frames():
        for i, t, img in clip.frames(color=True, start=start, step=every):
            crop = img[y0:y1, x0:x1, ::-1]
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            yield np.ascontiguousarray(crop)

    tr, vis = run(model, frames(), (pts_px - [x0, y0]) * s, model.step, device, torch)
    tr = tr / s + [x0, y0]
    if every == 1:
        return tr, vis
    t = frame_times(clip.path).t[start:]
    ts = t[::every][:len(tr)]                                    # the frames the model saw
    k = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
    w = np.clip((t - ts[k]) / (ts[k + 1] - ts[k]), 0, 1)[:, None, None]
    out = (1 - w) * tr[k] + w * tr[k + 1]
    seen = vis[k] & vis[k + 1] & (t <= ts[-1])[:, None]
    return out, seen


def clean(P: np.ndarray, V: np.ndarray, start: int, n_grid: int, tolerance=0.012):
    """Which CoTracker points to trust, as a mask (T, N). Two rules, both about the bottle's two dots.

    P (T, N, 2) pixels and V (T, N) CoTracker's own visibility; the last two points are the cap dot and
    the tape dot when N = n_grid + 2. The bottle is rigid, so its two dots stay the same distance apart.
    CoTracker's cap dot slides up the cap as the cap sinks into the foam, and the distance shrinks: where
    it differs from its rest value by more than `tolerance` (a fraction), the cap dot is dropped and the
    pipeline places the cap from the tape dot alone (indenter.cap_path). Where the tape dot itself is
    not seen, the bottle is taken as gone.
    """
    keep = V.copy()
    if P.shape[1] == n_grid + 2:
        cap, tape = n_grid, n_grid + 1
        sep = np.linalg.norm(P[:, tape] - P[:, cap], axis=1)
        rest = np.nanmedian(sep[start:start + 40])
        keep[:, cap] &= np.abs(sep / rest - 1) < tolerance
        keep[:, cap] &= keep[:, tape]
    return keep


def as_tracker(model, device="cuda", every=1, raw: dict | None = None):
    """A stand-in for tracker.track: the same points, followed by CoTracker instead of the centroid tracker.

    Pass the result as `tracker=` to pipeline.process_clip. raw: a dict that receives CoTracker's own
    output (positions and visibility, before cleaning) under the clip's file name, for a closer look.
    It does not pass frames to process_clip's camera-still check (`on_frame`), so that check is skipped,
    and it reports no blob areas.
    """
    def track(clip, calib, cfg, start, init_px, rest_area, rest_peak, grid_shape=(7, 15), **_):
        times = frame_times(clip.path).t
        T, N = len(times), len(init_px)
        have = np.isfinite(init_px[:, 0])
        pts, vis = track_points(clip, start, init_px[have], model, device, every=every)
        n = min(len(pts), T - start)
        P = np.full((T, N, 2), np.nan)
        V = np.zeros((T, N), bool)
        P[start:start + n, have] = pts[:n]
        V[start:start + n, have] = vis[:n]
        if raw is not None:
            raw[clip.path.name] = {"pts_px": P.copy(), "visible": V.copy(), "start": start}
        V = clean(P, V, start, grid_shape[0] * grid_shape[1])
        P[~V] = np.nan
        return times, P, V, np.full((T, N), np.nan)
    return track


def against_truth(syn: Path, model, device="cuda", every=(1, 4), rest_frames=20):
    """CoTracker3 and the centroid tracker on the rendered clips, whose truth is known.

    syn: a folder made by scripts/make_synthetic.py and then run through the pipeline (syn/out). Both
    trackers start from the same rest-frame positions. Scored on what the model is fitted to: each dot's
    movement from its rest position, in mm. Returns one row per clip and tracker.
    """
    import time
    syn = Path(syn)
    truth_all = np.load(syn / "truth.npz")
    calib = Calibration.load(syn / "out" / "calib" / "C1.json")
    rows = []
    for code in ("S1", "T05"):
        A = np.load(syn / "out" / "tracks" / f"{code}.npz")
        k0 = int(A["k0"])
        truth = truth_all[f"{code}_dots"][k0:]
        true_move = truth - truth[:rest_frames].mean(axis=0)              # the rendered foam is still here
        clip = Clip(syn / "raw" / f"SYN_{code}.MOV", calib.report.get("rotation_ccw"))
        runs = {"centroid": (A["pts_px"][k0:, :105], None)}
        for e in every:
            t0 = time.time()
            P, _ = track_points(clip, k0, A["pts_px"][k0].astype(float), model, device, every=e)
            runs[f"CoTracker3, 1 frame in {e}"] = (P[:, :105], time.time() - t0)
        for name, (P, seconds) in runs.items():
            mm = calib.px_to_mm(np.asarray(P, float))
            err = (mm - np.nanmean(mm[:rest_frames], axis=0)) - true_move[:len(mm)]     # (frames, dots, 2)
            dist = np.linalg.norm(err, axis=-1)
            # RMS over both components, as everywhere else in press2sim (model.rms); percentiles of the distance
            rows.append({"clip": code, "tracker": name, "rms_mm": float(np.sqrt(np.nanmean(err ** 2))),
                         "p95_mm": float(np.nanpercentile(dist, 95)), "worst_mm": float(np.nanmax(dist)),
                         "seconds": None if seconds is None else round(seconds, 1)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", help="clip to cross-check against the centroid tracker's tracks in out/")
    ap.add_argument("--synthetic", default=None, help="instead: score both trackers against the truth of the rendered "
                                                         "clips in this folder (scripts/make_synthetic.py, then the pipeline)")
    ap.add_argument("--file", default=None, help="clip file name (default: read from out/tracks/<code>.json)")
    ap.add_argument("--raw", default="raw"); ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-side", type=int, default=768, help="downscale the cropped region to this size")
    a = ap.parse_args(argv)
    if a.synthetic:
        rows = against_truth(Path(a.synthetic), load_model(a.device), a.device)
        (Path(a.synthetic) / "cotracker_vs_truth.json").write_text(json.dumps(rows, indent=1))
        print(f"{'clip':5s} {'tracker':26s} {'RMS (mm)':>9s} {'95th pct':>9s} {'worst':>7s}   error in each dot's movement from rest")
        for r in rows:
            print(f"{r['clip']:5s} {r['tracker']:26s} {r['rms_mm']:9.3f} {r['p95_mm']:9.3f} {r['worst_mm']:7.2f}")
        return
    if not a.code:
        ap.error("give --code or --synthetic")
    out = Path(a.out)
    meta = json.loads((out / "tracks" / f"{a.code}.json").read_text())
    A = np.load(out / "tracks" / f"{a.code}.npz")
    calib = Calibration.load(out / "calib" / f"{meta['calibration']}.json")
    k0 = int(A["k0"])
    P = A["pts_px"]
    q_px = P[k0]
    keep = np.isfinite(q_px[:, 0])
    clip = Clip(Path(a.raw) / (a.file or meta["file"]), calib.report.get("rotation_ccw"))
    tr, vis = track_points(clip, k0, q_px[keep], load_model(a.device), a.device, max_side=a.max_side)
    T = min(len(tr), len(P) - k0)
    ct = np.full((T, len(q_px), 2), np.nan)
    ct[:, keep] = tr[:T]
    cv_ = np.zeros((T, len(q_px)), bool)
    cv_[:, keep] = vis[:T]
    ours = P[k0:k0 + T]
    both = np.isfinite(ours[..., 0]) & cv_
    d_mm = np.linalg.norm(calib.px_to_mm(ct) - calib.px_to_mm(ours), axis=-1)
    d = d_mm[both]
    per_dot = np.nanmedian(np.where(both, d_mm, np.nan), axis=0)
    rep = {"code": a.code, "frames": int(T), "compared": int(both.sum()),
           "median_mm": float(np.median(d)), "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max()),
           "worst_dots": [int(i) for i in np.argsort(np.nan_to_num(per_dot, nan=-1))[::-1][:5]]}
    np.savez_compressed(out / "tracks" / f"{a.code}_cotracker.npz", pts_px=ct.astype(np.float32), visible=cv_, k0=k0)
    (out / "tracks" / f"{a.code}_cotracker.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
