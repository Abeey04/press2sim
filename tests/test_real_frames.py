"""Dot and bottle detection on two real frames from the session (rest, and a 29 mm press)."""
from pathlib import Path

import cv2
import numpy as np

from press2sim import dots
from press2sim.calib import Calibration
from press2sim.config import Config

DATA = Path(__file__).parent / "data"
PPM = 3.42   # approximate scale of these frames; a real run uses the checkerboard instead


def approx_calib(cfg):
    H = np.array([[1 / PPM, 0, -519 / PPM], [0, -1 / PPM, 761.5 / PPM], [0, 0, 1]])
    return Calibration(H, np.linalg.inv(H), np.zeros((10, 25, 2)), cfg.board.corner_xz_mm(), {"px_per_mm": PPM})


def test_full_grid_found_and_labelled():
    cfg = Config()
    g = cv2.imread(str(DATA / "real_rest.png"), cv2.IMREAD_GRAYSCALE)
    px, mm, rep = dots.find_grid(g, approx_calib(cfg), cfg)
    assert rep["found"] == 105
    assert np.all(np.diff(px[..., 1], axis=0) < 0)     # rows run upward
    assert np.all(np.diff(px[..., 0], axis=1) > 0)     # columns run rightward
    assert abs(rep["pitch_z_mean"] - 10.0) < 0.3


def test_bottle_dots_at_rest():
    cfg = Config()
    g = cv2.imread(str(DATA / "real_rest.png"), cv2.IMREAD_GRAYSCALE)
    b = dots.find_bottle_dots(g, approx_calib(cfg), cfg, face_top_mm=73.0)
    assert b is not None
    assert 43 < b["separation_mm"] < 50, b["separation_mm"]
    assert b["cap_px"][1] > b["tape_px"][1]               # cap dot below the tape dot
    assert abs(b["cap_px"][0] - 1006) < 4 and abs(b["cap_px"][1] - 455) < 4
