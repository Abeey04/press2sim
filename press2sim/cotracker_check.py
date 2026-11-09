"""Optional cross-check with CoTracker3 (pretrained point tracker, needs PyTorch; a GPU is strongly advised).

The centroid tracker is the measurement; CoTracker3 is an independent second opinion. This runs the
online model on the same clip, starting from the same rest-frame dot positions, and reports how far
the two disagree (in mm). Large disagreement on a dot flags a tracking problem worth looking at.

usage: python -m press2sim.cotracker_check --code T05 --raw raw --out out [--device cuda]
CoTracker is CC-BY-NC licensed; credit it in the README if you report its numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .calib import Calibration
from .video import Clip


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
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy().astype(bool).reshape(tracks.shape[1], -1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--file", default=None, help="clip file name (default: read from out/tracks/<code>.json)")
    ap.add_argument("--raw", default="raw"); ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-side", type=int, default=768, help="downscale the cropped region to this size")
    a = ap.parse_args(argv)
    import torch
    out = Path(a.out)
    meta = json.loads((out / "tracks" / f"{a.code}.json").read_text())
    A = np.load(out / "tracks" / f"{a.code}.npz")
    calib = Calibration.load(out / "calib" / f"{meta['calibration']}.json")
    k0 = int(A["k0"])
    P = A["pts_px"]
    q_px = P[k0]
    keep = np.isfinite(q_px[:, 0])
    pad = 80
    x0 = int(max(0, np.nanmin(q_px[:, 0]) - pad)); x1 = int(np.nanmax(q_px[:, 0]) + pad)
    y0 = int(max(0, np.nanmin(q_px[:, 1]) - 3 * pad)); y1 = int(np.nanmax(q_px[:, 1]) + pad)
    s = min(1.0, a.max_side / max(x1 - x0, y1 - y0))
    clip = Clip(Path(a.raw) / (a.file or meta["file"]), calib.report.get("rotation_ccw"))

    def frames():
        for i, t, img in clip.frames(color=True, start=k0):
            crop = img[y0:y1, x0:x1, ::-1]
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            yield np.ascontiguousarray(crop)

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(a.device)
    q = (q_px[keep] - [x0, y0]) * s
    tr, vis = run(model, frames(), q, model.step, a.device, torch)
    T = min(len(tr), len(P) - k0)
    ct = np.full((T, len(q_px), 2), np.nan)
    ct[:, keep] = tr[:T] / s + [x0, y0]
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
