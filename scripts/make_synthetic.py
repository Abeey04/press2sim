"""Render synthetic clips with known ground truth, to test the pipeline end to end.

The scene copies the real setup: 234 x 73 mm face with a 15 x 7 grid of 4 mm dots, the bottle
(yellow cap, two dots) pressing at the middle, a checkerboard card for calibration, a sticky-note
slate at the start of each clip, mains flicker, sensor noise, HEVC compression, and iPhone-style
dropped frames (uneven timestamps). The true dot and cap positions are saved next to each clip.

usage: python scripts/make_synthetic.py OUT_DIR [--rotate180]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

W, H = 1920, 1080
SHIFT = 4                        # OpenCV sub-pixel drawing: coordinates in 1/16 px
S = 1 << SHIFT
CARD_LEFT = 23.0                 # card x of the foam's left end (card is 280 wide, centred)
L, HT = 234.0, 73.0
CX = 117.0                       # press centre, foam x


def camera_h():
    """Card/face plane (x right, z up, mm) -> pixels for a slightly turned pinhole camera."""
    f, cx, cy = 1650.0, 960.0, 540.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    C = np.array([CARD_LEFT + CX - 18.0, -470.0, 40.0])         # camera centre in world mm
    R0 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)     # world -> camera axes
    def rot(axis, deg):
        a = np.radians(deg); c, s = np.cos(a), np.sin(a)
        if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = rot("z", 0.3) @ rot("x", 1.5) @ rot("y", 0.6) @ R0
    M = K @ R
    Hm = np.column_stack([M[:, 0], M[:, 2], -M @ C])            # columns for x, z, 1
    return Hm / Hm[2, 2]


HM = camera_h()


def to_px(p):
    p = np.asarray(p, float)
    q = np.c_[p.reshape(-1, 2), np.ones(p.size // 2)] @ HM.T
    return (q[:, :2] / q[:, 2:3]).reshape(p.shape)


def poly(img, pts_mm, color):
    q = np.round(to_px(pts_mm) * S).astype(np.int32)
    cv2.fillPoly(img, [q], color, lineType=cv2.LINE_AA, shift=SHIFT)


def ellipse(img, c_mm, rx_mm, rz_mm, color):
    c = to_px(c_mm)
    ex = np.linalg.norm(to_px(c_mm + np.array([rx_mm, 0])) - c)
    ez = np.linalg.norm(to_px(c_mm + np.array([0, rz_mm])) - c)
    cv2.ellipse(img, (int(round(c[0] * S)), int(round(c[1] * S))), (int(round(ex * S)), int(round(ez * S))),
                0, 0, 360, color, -1, cv2.LINE_AA, SHIFT)


def background(rng):
    img = np.zeros((H, W, 3), np.uint8)
    yline = int(to_px(np.array([CARD_LEFT, 0]))[1]) - 110       # where the table meets the wall
    grad = np.linspace(0, 1, yline)[:, None, None]
    img[:yline] = (np.array([196, 210, 214]) * (0.92 + 0.08 * grad)).astype(np.uint8)
    img[yline:] = (176, 172, 174)
    for _ in range(160):                                          # texture on table and wall: static features
        x, y = int(rng.uniform(0, W)), int(rng.uniform(yline + 5, H))
        cv2.circle(img, (x, y), int(rng.uniform(2, 6)), tuple(int(v) for v in rng.uniform(120, 160, 3)), -1, cv2.LINE_AA)
    for _ in range(40):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, yline))
        cv2.circle(img, (x, y), int(rng.uniform(3, 8)), tuple(int(v) for v in rng.uniform(150, 185, 3)), -1, cv2.LINE_AA)
    return img


def rest_dots():
    xs = 47 + 10 * np.arange(15) + np.array([0, .3, -.4, .2, .5, -.3, 0, .4, -.2, .1, .3, -.5, .2, 0, .3])
    zs = 8 + 10 * np.arange(7) + np.array([0, -.2, .1, .3, -.1, .2, 0])
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X, Z], -1).reshape(-1, 2)                   # foam mm, row-major from the bottom


def deform(p, d):
    """Indentation field for press depth d (mm) at x = CX."""
    x, z = p[:, 0], p[:, 1]
    r = np.abs(x - CX)
    Sx = np.where(r <= 30, 1.0, np.exp(-((r - 30) / 25.0) ** 2))
    uz = -d * Sx * (z / HT) ** 1.3
    ux = 0.08 * d * np.tanh((x - CX) / 20.0) * (z / HT) * Sx
    duz_dz = -d * Sx * 1.3 * (z / HT) ** 0.3 / HT
    return np.c_[x + ux, z + uz], np.clip(1 + duz_dz, 0.35, 1.0)


def press_profile(t, t_press=1.4, down=0.6, hold=1.0, z_rest=81.0, z_low=44.0):
    """Cap bottom height and foam dent depth vs time (snap release, slow recovery)."""
    t1 = t_press + down; t2 = t1 + hold; t3 = t2 + 0.12
    if t < t_press:
        zc = z_rest
    elif t < t1:
        zc = z_rest + (z_low - z_rest) * (t - t_press) / down
    elif t < t2:
        zc = z_low
    elif t < t3:
        zc = z_low + (130 - z_low) * (t - t2) / 0.12
    else:
        zc = 130.0
    if t < t2:
        dent = max(0.0, HT - zc)
    else:
        s = t - t2
        dent = (HT - z_low) * (0.3 * np.exp(-s / 0.02) + 0.7 * np.exp(-s / 0.25))
    return zc, dent


def draw_bottle(img, zc, tilt_deg=1.0):
    a = np.radians(tilt_deg)
    Rm = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])   # top leans right for positive tilt
    base = np.array([CX, zc])
    def tr(p): return (np.asarray(p, float) - base) @ Rm.T + base + np.array([CARD_LEFT, 0])
    poly(img, tr([[CX - 30.5, zc + 19], [CX + 30.5, zc + 19], [CX + 30.5, zc + 105], [CX - 30.5, zc + 105]]), (226, 229, 231))
    poly(img, tr([[CX - 6, zc + 49], [CX + 6, zc + 49], [CX + 6, zc + 61], [CX - 6, zc + 61]]), (245, 246, 247))
    poly(img, tr([[CX - 30, zc], [CX + 30, zc], [CX + 30, zc + 19], [CX - 30, zc + 19]]), (25, 210, 236))
    for h in (8.5, 55.0):
        c = tr([[CX, zc + h]])[0]
        ellipse(img, c, 2.0, 2.0, (30, 30, 30))
    cap = tr([[CX, zc + 8.5]])[0] - np.array([CARD_LEFT, 0])
    tape = tr([[CX, zc + 55]])[0] - np.array([CARD_LEFT, 0])
    return cap, tape


def draw_slate(img, rng):
    poly(img, np.array([[60, 5], [140, 5], [140, 85], [60, 85]]) + [CARD_LEFT, 0], (90, 235, 245))
    for k in range(5):
        a = np.array([70 + 12 * k, 30.0]); b = a + [8, 30]
        cv2.line(img, tuple(np.round(to_px(a + [CARD_LEFT, 0])).astype(int)),
                 tuple(np.round(to_px(b + [CARD_LEFT, 0])).astype(int)), (40, 40, 40), 4, cv2.LINE_AA)


def draw_face(img, tex_mask, tex):
    poly(img, np.array([[0, 0], [L, 0], [L, HT], [0, HT]]) + [CARD_LEFT, 0], (210, 193, 163))
    np.copyto(img, np.clip(img.astype(np.int16) + tex, 0, 255).astype(np.uint8), where=tex_mask[..., None])


def draw_board(img):
    poly(img, np.array([[0, 0], [280, 0], [280, 130], [0, 130]]), (238, 240, 241))
    for i in range(26):
        for j in range(11):
            if (i + j) % 2 == 0:
                x, z = 10 + 10 * i, 10 + 10 * j
                poly(img, np.array([[x, z], [x + 10, z], [x + 10, z + 10], [x, z + 10]]), (28, 28, 28))


def finish(img, t, rng):
    f = 1.0 + 0.03 * np.sin(2 * np.pi * 100 * t)                # 100 Hz mains flicker
    out = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8) * f + rng.normal(0, 2.0, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def frame_times(duration, rng, fps=240.0, drop=0.18):
    n = int(duration * fps)
    keep = rng.random(n) > drop
    keep[0] = True
    return np.arange(n)[keep] / fps


def write(path, frames_iter, rotate180):
    tb = Fraction(1, 2400)
    tmp = path.with_suffix(".tmp.mov") if rotate180 else path
    with av.open(str(tmp), "w") as c:
        s = c.add_stream("libx265", rate=240)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.codec_context.time_base = tb
        s.options = {"crf": "14", "preset": "ultrafast", "x265-params": "log-level=error"}
        for t, img in frames_iter:
            if rotate180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            fr = av.VideoFrame.from_ndarray(img, format="bgr24")
            fr.pts = int(round(t * 2400)); fr.time_base = tb
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    if rotate180:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "180", "-i", str(tmp), "-c", "copy",
                        str(path)], check=True)
        tmp.unlink()


def make(out: Path, rotate180=False, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    bg = background(rng)
    face = np.zeros((H, W), np.uint8)
    poly(face, np.array([[0, 0], [L, 0], [L, HT], [0, HT]]) + [CARD_LEFT, 0], 255)
    tex_mask = face > 0
    tex = rng.normal(0, 6, (H, W, 1)).astype(np.int16)
    D = rest_dots()
    truth = {}

    def scene(t, dent, zc, slate, board=False, bottle=True):
        img = bg.copy()
        draw_face(img, tex_mask, tex)
        p, sq = deform(D, dent)
        for (x, z), s_ in zip(p, sq):
            ellipse(img, np.array([x + CARD_LEFT, z]), 2.0, 2.0 * s_, (35, 35, 38))
        cap = tape = (np.nan, np.nan)
        if bottle:
            cap, tape = draw_bottle(img, zc)
        if board:
            draw_board(img)
        if slate:
            draw_slate(img, rng)
        return finish(img, t, rng), p, cap, tape

    # C1: foam visible, then the card for 2 s, then removed
    ts = frame_times(3.0, rng)
    def gen_c():
        for t in ts:
            img, *_ = scene(t, 0, 81, False, board=(0.5 <= t <= 2.5), bottle=False)
            yield t, img
    write(out / "raw" / "SYN_C1.MOV", gen_c(), rotate180)

    # S1: slate, then nothing touching
    ts = frame_times(2.0, rng)
    gt = []
    def gen_s():
        for t in ts:
            img, p, *_ = scene(t, 0, 81, t < 0.4, bottle=False)
            gt.append(p)
            yield t, img
    write(out / "raw" / "SYN_S1.MOV", gen_s(), rotate180)
    truth["S1"] = {"t": ts, "dots": np.array(gt)}

    # T05: slate, rest, press to 29 mm, hold, snap release, recovery
    ts = frame_times(4.0, rng)
    gt, gcap, gtape, gz = [], [], [], []
    def gen_t():
        for t in ts:
            zc, dent = press_profile(t)
            img, p, cap, tape = scene(t, dent, zc, t < 0.4)
            gt.append(p); gcap.append(cap); gtape.append(tape); gz.append(zc)
            yield t, img
    write(out / "raw" / "SYN_T05.MOV", gen_t(), rotate180)
    truth["T05"] = {"t": ts, "dots": np.array(gt), "cap_dot": np.array(gcap), "tape_dot": np.array(gtape),
                    "cap_bottom_z": np.array(gz)}

    np.savez(out / "truth.npz", H=HM, card_left=CARD_LEFT,
             **{f"{k}_{n}": v for k, d in truth.items() for n, v in d.items()})
    with open(out / "trials.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "file", "kind", "depth_mm", "speed", "hold_s", "load_kg", "held_out", "notes"])
        w.writerow(["C1", "SYN_C1.MOV", "calib", "", "", "", "", "", "synthetic"])
        w.writerow(["S1", "SYN_S1.MOV", "static", "", "", "", "", "", "synthetic"])
        w.writerow(["T05", "SYN_T05.MOV", "press", "29", "slow", "1", "", "no", "synthetic"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rotate180", action="store_true", help="store frames upside down with a 180-degree display tag")
    a = ap.parse_args()
    make(Path(a.out), a.rotate180)
    print("wrote", a.out)
