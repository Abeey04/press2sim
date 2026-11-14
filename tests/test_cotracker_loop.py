"""The CoTracker online loop must cover every frame, whatever the clip length, and return only the points
asked for (no torch needed)."""
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
    """Like CoTracker3's online model: with add_support_grid it also follows a grid of extra points, which
    come back after the queries."""
    step = 8
    support = 36
    def __init__(s): s.calls = []
    def __call__(s, video_chunk, is_first_step=False, queries=None, add_support_grid=False):
        if is_first_step:
            q = queries.a[0, :, 1:]
            s.q = np.vstack([q, np.full((s.support, 2), -1.0)]) if add_support_grid else q
            return None, None
        s.calls.append(video_chunk.shape[1])
        n = (len(s.calls) - 1) * s.step + video_chunk.shape[1]
        return _T(np.repeat(s.q[None, None], n, axis=1)), _T(np.ones((1, n, len(s.q)), bool))


def test_every_frame_covered_and_only_the_queries_returned():
    queries = np.array([[3.0, 4.0], [5.0, 6.0]])
    for n in (9, 16, 17, 40, 41, 47, 48, 1001):
        frames = (np.zeros((12, 20, 3), np.uint8) for _ in range(n))
        tr, vis = run(_Model(), frames, queries, 8, "cpu", _Torch)
        assert tr.shape == (n, 2, 2) and vis.shape == (n, 2)
        assert np.allclose(tr, queries)                     # the support grid's points are dropped
