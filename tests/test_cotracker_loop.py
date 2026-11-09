"""The CoTracker online loop must cover every frame, whatever the clip length (no torch needed)."""
import numpy as np

from press2sim.cotracker_check import run


class _T:
    def __init__(s, a): s.a = np.asarray(a)
    def permute(s, *ax): return _T(np.transpose(s.a, ax))
    def __getitem__(s, k): return _T(s.a[k])
    def float(s): return _T(s.a.astype(np.float32))
    def to(s, d): return s
    def cpu(s): return s
    def numpy(s): return s.a
    @property
    def shape(s): return s.a.shape


class _Torch:
    float32 = np.float32
    @staticmethod
    def from_numpy(a): return _T(a)
    @staticmethod
    def tensor(a, dtype=None, device=None): return _T(np.asarray(a, dtype))


class _Model:
    step = 8
    def __init__(s): s.calls = []
    def __call__(s, video_chunk, is_first_step=False, queries=None, add_support_grid=False):
        if is_first_step:
            s.q = queries.a[0, :, 1:]
            return None, None
        s.calls.append(video_chunk.shape[1])
        n = (len(s.calls) - 1) * s.step + video_chunk.shape[1]
        return _T(np.repeat(s.q[None, None], n, axis=1)), _T(np.ones((1, n, len(s.q)), bool))


def test_every_frame_covered():
    for n in (9, 16, 17, 40, 41, 47, 48, 1001):
        frames = (np.zeros((12, 20, 3), np.uint8) for _ in range(n))
        tr, vis = run(_Model(), frames, np.array([[3.0, 4.0], [5.0, 6.0]]), 8, "cpu", _Torch)
        assert tr.shape == (n, 2, 2) and vis.shape == (n, 2)
