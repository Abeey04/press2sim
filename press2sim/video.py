"""Reading iPhone clips with their true per-frame timestamps.

iPhone slow motion at "240 fps" often records fewer frames than that, dropping some and leaving
longer gaps in the timestamps. Everything downstream therefore uses each frame's own presentation
time from the file (PyAV), never frame index divided by a nominal rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

_ROT = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def rotate(img: np.ndarray, deg_ccw: int) -> np.ndarray:
    """Rotate counter-clockwise by deg_ccw (multiple of 90), matching how players display the clip."""
    k = _ROT[int(round(deg_ccw)) % 360]
    return img if k is None else cv2.rotate(img, k)


def display_rotation(path: str | Path) -> int:
    """Counter-clockwise rotation a player applies to show the clip upright (from the display matrix)."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for fr in c.decode(s):
            return int(round(fr.rotation or 0)) % 360
    return 0


@dataclass
class TimeInfo:
    n_frames: int
    t: np.ndarray              # seconds from the first frame
    dt_median: float
    fps_from_median_gap: float
    fps_average: float
    n_long_gaps: int           # gaps longer than 1.5 x the median gap (dropped frames)
    max_gap: float
    width: int
    height: int
    rotation_ccw: int
    codec: str

    def summary(self) -> dict:
        return {
            "n_frames": self.n_frames, "duration_s": float(self.t[-1]) if self.n_frames else 0.0,
            "dt_median_ms": 1e3 * self.dt_median, "fps_from_median_gap": self.fps_from_median_gap,
            "fps_average": self.fps_average, "n_long_gaps": self.n_long_gaps,
            "max_gap_ms": 1e3 * self.max_gap, "width": self.width, "height": self.height,
            "rotation_ccw": self.rotation_ccw, "codec": self.codec,
        }


def frame_times(path: str | Path) -> TimeInfo:
    """Presentation timestamps of every frame, read from the container without decoding."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        pts = [p.pts for p in c.demux(s) if p.pts is not None]
        w, h, codec = s.codec_context.width, s.codec_context.height, s.codec_context.name
    rot = display_rotation(path)
    if rot in (90, 270):
        w, h = h, w
    pts = np.sort(np.asarray(pts, dtype=np.int64))
    t = (pts - pts[0]) * tb if len(pts) else np.zeros(0)
    gaps = np.diff(t)
    if len(gaps):
        med = float(np.median(gaps))
        long_gaps = int(np.sum(gaps > 1.5 * med))
        fps_avg = (len(t) - 1) / t[-1] if t[-1] > 0 else float("nan")
        mx = float(gaps.max())
    else:
        med, long_gaps, fps_avg, mx = float("nan"), 0, float("nan"), float("nan")
    return TimeInfo(len(t), t, med, 1.0 / med if med == med and med > 0 else float("nan"),
                    float(fps_avg), long_gaps, mx, w, h, rot, codec)


class Clip:
    """Sequential decoder yielding (index, time_s, image) in presentation order, upright."""

    def __init__(self, path: str | Path, rotation_ccw: int | None = None):
        self.path = Path(path)
        self.rotation_ccw = display_rotation(self.path) if rotation_ccw is None else int(rotation_ccw) % 360

    def frames(self, color: bool = False, start: int = 0, stop: int | None = None, step: int = 1):
        fmt = "bgr24" if color else "gray"
        t0 = None
        with av.open(str(self.path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            tb = float(s.time_base)
            for i, fr in enumerate(c.decode(s)):
                if fr.pts is None:
                    continue
                if t0 is None:
                    t0 = fr.pts
                if stop is not None and i >= stop:
                    break
                if i < start or (i - start) % step:
                    continue
                img = rotate(fr.to_ndarray(format=fmt), self.rotation_ccw)
                yield i, (fr.pts - t0) * tb, img

    def frame(self, index: int, color: bool = False):
        for i, t, img in self.frames(color=color, start=index, stop=index + 1):
            return t, img
        raise IndexError(f"{self.path.name} has no frame {index}")
