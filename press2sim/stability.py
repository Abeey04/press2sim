"""Did the camera stay still? Track static background features through a clip."""
from __future__ import annotations

import cv2
import numpy as np


class BackgroundCheck:
    """Optical flow on corners in the table area in front of the foam and the far sides of the frame.

    The foam, the bottle and everything above the foam (hands, shadows) are excluded. The median
    feature motion over time is the camera motion relative to the table, in pixels.
    """

    def __init__(self, face_box_px, n_features=40, every=8):
        self.box = face_box_px       # (x0, y0, x1, y1) of the foam face
        self.n = n_features
        self.every = every
        self.ref = None
        self.p0 = None
        self.k = 0
        self.t, self.dx, self.dy = [], [], []

    def _mask(self, shape):
        H, W = shape
        x0, y0, x1, y1 = self.box
        m = np.zeros(shape, np.uint8)
        m[min(H, y1 + 20):, :] = 255                                   # table in front of the foam
        top = max(0, y0 - 20)
        m[top:, :max(0, x0 - 120)] = 255                               # far left
        m[top:, min(W, x1 + 120):] = 255                               # far right
        return m

    def __call__(self, idx, t, g):
        if self.ref is None:
            self.ref = g.copy()
            p = cv2.goodFeaturesToTrack(g, self.n, 0.01, 25, mask=self._mask(g.shape), blockSize=9)
            self.p0 = None if p is None else p.astype(np.float32)
            return
        self.k += 1
        if self.p0 is None or self.k % self.every:
            return
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.ref, g, self.p0, None, winSize=(31, 31), maxLevel=2,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        good = st.ravel() == 1
        if good.sum() < 5:
            return
        d = (p1 - self.p0).reshape(-1, 2)[good]
        self.t.append(t); self.dx.append(float(np.median(d[:, 0]))); self.dy.append(float(np.median(d[:, 1])))

    def result(self):
        if not self.t:
            return {"features": 0 if self.p0 is None else int(len(self.p0)), "note": "too few static features"}
        dx, dy = np.array(self.dx), np.array(self.dy)
        mag = np.hypot(dx, dy)
        return {"features": int(len(self.p0)), "max_shift_px": float(mag.max()),
                "median_shift_px": float(np.median(mag)), "final_shift_px": float(mag[-1]),
                "t": self.t, "dx": self.dx, "dy": self.dy}
