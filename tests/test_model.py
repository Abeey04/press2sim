"""The 3D block model against answers known in closed form."""
import numpy as np

from press2sim import fem
from press2sim import model as M


def test_plate_matches_plain_compression():
    # with Poisson's ratio 0 a plate pressed into the block gives force = E A / H per mm
    fb = fem.FoamBlock(0.0, 10.0, 1.0, ("plate",), fine=20.0)
    assert abs(fb.solve_depth(1.0) / (234 * 139.5 / 73 * 1e-6) - 1) < 1e-6


def test_support_acts_as_a_spring_in_series():
    k = 0.02                                   # support stiffness per unit E (1/mm)
    fb = fem.FoamBlock(0.0, 10.0, 1.0, ("plate",), fine=20.0, support=k)
    expected = 234 * 139.5 / (73 + 1 / k) * 1e-6
    assert abs(fb.solve_depth(1.0) / expected - 1) < 1e-6


def test_crushable_foam_softens():
    lin = fem.FoamBlock(0.1, 10.0, 1.0, ("cap", 117.0, 0.0, 60.0, 60.0), fine=12.0)
    foam = fem.FoamBlock(0.1, 0.05, 0.05, ("cap", 117.0, 0.0, 60.0, 60.0), fine=12.0)
    assert foam.solve_depth(20.0) < 0.6 * lin.solve_depth(20.0)


def test_shape_search_can_switch_the_soft_top_on(monkeypatch):
    # the soft top needs a thickness and a softness together; a start without it must still reach it
    from press2sim import fit as F

    def loss(p, clips, cache, frames=None):
        has_top = p.skin_t > 0 and p.skin_k < 1
        return 1.0 + abs(p.skin_k - 0.07) if has_top else 2.0

    monkeypatch.setattr(F, "shape_loss", loss)
    best, best_loss, _ = F.fit_shape([], None, M.Params(), names=("skin_k", "skin_t"))
    assert best.skin_t > 0 and best.skin_k < 1 and best_loss < 1.01


def _toy_press(p):
    """A press, hold and fast lift on a toy block whose face sinks most under the cap; dots as the model moves them."""
    d = np.linspace(0, 20, 21)
    xn, zn = np.linspace(0, 234, 40), np.linspace(0, 73, 12)
    X, Z = np.meshgrid(xn, zn, indexing="ij")
    face = np.zeros((len(d), len(xn), len(zn), 3))
    face[..., 2] = -d[:, None, None] * np.exp(-((X - 117) / 45.0) ** 2) * (Z / 73.0)
    tab = M.Table(d, d.copy(), face, xn, zn, 117.0)
    t = np.arange(0, 4.0, 1 / 240)
    depth = np.interp(t, [0, 0.5, 1.0, 2.0, 2.08, 4.0], [-5, 0, 12, 12, -20, -20])
    cols, rows = np.meshgrid(47 + 10.0 * np.arange(15), 8 + 10.0 * np.arange(7))
    rest = np.c_[cols.ravel(), rows.ravel()]
    cd = M.ClipData("toy", "press", t, rest, np.zeros((len(t), 105, 2)), 0, depth=depth, x_c=117.0, t_end=float(t[-1]))
    U, q, F, c = M.predict_clip(cd, tab, p)
    k = np.where((t > 2.0) & ~c)[0][0]                         # the model foam lets go of the cap here
    cd.disp, cd.events = U, {"t_contact": 0.5, "t_lift": 2.0, "t_release": float(t[k])}
    return cd, tab


def test_damping_fit_recovers_a_known_relaxation():
    # the dent under the cap after the foam lets go is enough to find the relaxing fraction and its time
    from press2sim import fit as F
    true = M.Params(g=(0.2,), tau=(0.5,), tau_v=0.005)
    cd, tab = _toy_press(true)
    assert len(F.dent_dots(cd)) == 7 and F.dent_frames(cd).sum() > 300      # columns 87 to 147 mm, under the cap
    p, loss, none = F.fit_time([cd], tab, M.Params(g=(0.08,), tau=(1.5,), tau_v=0.005))   # the dashpot is not fitted
    assert abs(p.g[0] - 0.2) < 0.01 and abs(p.tau[0] - 0.5) < 0.03
    assert loss < 0.01 < none


def test_cap_seen_for_a_single_frame_is_dropped():
    # a stray blob standing in for a bottle dot for one frame must not set the cap's height
    from press2sim.config import Config
    from press2sim.indenter import cap_path
    t = np.arange(12) / 240.0
    cap = np.full((12, 2), np.nan)
    tape = np.full((12, 2), np.nan)
    cap[:5] = [100.0, 90.0]
    tape[:5] = [100.0, 136.5]
    tape[8] = [100.0, 60.0]                    # one frame, long after the bottle was last seen
    cp = cap_path(t, cap, tape, Config(), rest_sel=np.arange(12) < 3)
    assert np.isfinite(cp["z_mm"][:5]).all() and not np.isfinite(cp["z_mm"][5:]).any()
    assert cp["source"][8] == 0


def test_qlv_instant_and_long_term_response():
    # linear elastic force R(q) = q: a step force F gives q = F/E at once and F/(E g_inf) eventually
    d = np.linspace(0, 10, 101)
    p = M.Params(g=(0.3, 0.2), tau=(0.05, 0.5), E=2.0, tau_v=0.0)
    t = np.arange(0, 6, 0.002)
    q, F, _ = M.qlv(t, d, d.copy(), p, force=np.where(t > 0, 1.0, 0.0))
    assert abs(q[1] - 1.0 / p.E) < 0.02
    assert abs(q[-1] - 1.0 / (p.E * p.g_inf)) < 0.01


def test_dashpot_sets_the_rise_time():
    # spring and dashpot in parallel (Kelvin-Voigt): q = F/E (1 - exp(-t / tau_v))
    d = np.linspace(0, 10, 101)
    p = M.Params(g=(1e-9,), tau=(1.0,), E=1.0, tau_v=0.2)
    t = np.arange(0, 2, 0.001)
    q, F, _ = M.qlv(t, d, d.copy(), p, force=np.where(t > 0, 1.0, 0.0))
    assert abs(q[np.searchsorted(t, 0.2)] - (1 - np.exp(-1))) < 0.01


def test_qlv_lets_go_of_a_rising_cap():
    # cap pressed 5 mm and pulled away fast: the foam stays below it and creeps back afterwards
    d = np.linspace(0, 10, 101)
    p = M.Params(g=(0.3, 0.3), tau=(0.1, 1.0), E=1.0, tau_v=0.02)
    t = np.arange(0, 4, 0.004)
    delta = np.where(t < 2.0, 5.0, np.maximum(5.0 - (t - 2.0) * 200.0, -10))
    q, F, c = M.qlv(t, d, d.copy(), p, delta=delta)
    k = np.searchsorted(t, 2.1)
    assert c[np.searchsorted(t, 1.0)] and not c[k]
    assert q[k] > max(delta[k], 0) and q[-1] < 0.5 * q[k]
