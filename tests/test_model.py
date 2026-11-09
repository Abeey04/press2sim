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
