"""The tracker on a small rendered clip: two dots in a column squeezed together until their blobs merge,
then let go, as the dots beside the cap are in a deep press.

A blob belongs to one dot only. Before this was enforced in the tracker's local fallback search, the dot
that lost its blob in the merge was placed on its neighbour's, stayed valid, and after the release kept
following the neighbour, a row's spacing from where it should be.
"""
import numpy as np
import pytest

from press2sim import dots, tracker
from press2sim.config import Config

PPM = 3.74                                  # the session's scale (px per mm on the face)
ROWS, COLS = 3, 3
MOVER = 1 * COLS + 1                        # the middle dot; it rises towards the one above (rows count upward)


class Clip:
    """Rendered frames at 240 fps: the middle dot rises towards the one above, holds, and comes back."""

    def __init__(self, cfg, gap_px, n_rest=10, n_move=20, n_hold=10):
        self.diam = cfg.grid.dot_diameter_mm * PPM
        pitch = cfg.grid.pitch_mm * PPM
        rest = np.array([[60 + c * pitch, 150 - r * pitch] for r in range(ROWS) for c in range(COLS)])
        lift = np.r_[np.zeros(n_rest), np.arange(1, n_move + 1) / n_move, np.ones(n_hold),
                     1 - np.arange(1, n_move + 1) / n_move, np.zeros(n_rest)] * (pitch - gap_px)
        self.truth = np.repeat(rest[None], len(lift), axis=0)
        self.truth[:, MOVER, 1] -= lift

    def image(self, k, bg=190.0, ink=60.0):
        yy, xx = np.mgrid[0:200, 0:200].astype(float)
        img = np.full(yy.shape, bg)
        for cx, cy in self.truth[k]:
            cover = np.clip(self.diam / 2 - np.hypot(xx - cx, yy - cy) + 0.5, 0, 1)   # soft edge: sub-pixel centres
            img = np.minimum(img, bg - (bg - ink) * cover)
        return img.astype(np.uint8)

    def frames(self, start=0, **_):
        for k in range(start, len(self.truth)):
            yield k, k / 240.0, self.image(k)


class Calib:
    report = {"px_per_mm": PPM}


@pytest.mark.parametrize("gap_px", [14.0, 10.0, 6.0])     # centre to centre at the closest; the dots are 15 px wide
def test_one_dot_per_blob_and_back_after_merge(gap_px):
    cfg = Config()
    clip = Clip(cfg, gap_px)
    roi = (0, 0, 200, 200)
    C0 = dots.contrast_image(clip.image(0), roi, clip.diam)
    comp, area, peak = dots.blobs(C0, 0.4 * np.percentile(C0, 99.7), clip.diam)
    near = [int(np.argmin(np.linalg.norm(comp - p, axis=1))) for p in clip.truth[0]]
    t, P, V, _ = tracker.track(clip, Calib(), cfg, 0, clip.truth[0].copy(), area[near], peak[near],
                               grid_shape=(ROWS, COLS))
    thr = cfg.tuning.contrast_fraction * float(np.median(peak[near]))      # the tracker's own threshold
    merged = 0
    for k in range(len(t)):
        Ck = dots.contrast_image(clip.image(k), roi, clip.diam)
        *_, lab, _ = dots.blobs(Ck, thr, clip.diam, labels=True)
        on = [int(lab[int(round(y)), int(round(x))]) for x, y in P[k, V[k]]]
        assert 0 not in on and len(set(on)) == len(on), f"frame {k}: two valid dots on one blob"
        merged += len(set(lab[int(round(y)), int(round(x))] for x, y in clip.truth[k])) < ROWS * COLS
    assert merged > 0                                     # the two dots did merge into one blob
    assert V[-1].all()
    assert np.abs(P[-1] - clip.truth[-1]).max() < 0.5    # and each is back on its own dot
