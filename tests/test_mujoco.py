"""The MuJoCo version of the block builds and its vertex grid is indexed correctly (skipped without mujoco)."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from press2sim import mujoco_foam as MJ


def test_block_builds_and_grid_is_ordered():
    s = MJ.MjSetup(count=(6, 4, 3))
    m, d, v0, order = MJ.build(s, 117.0)
    rest = v0[order] * 1000.0
    assert rest.shape == (6, 4, 3, 3)
    assert np.allclose(rest[-1, 0, 0], [234.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(rest[0, -1, -1], [0.0, 139.5, 73.0], atol=1e-6)
    assert np.all(np.diff(rest[:, 0, 0, 0]) > 0) and np.all(np.diff(rest[0, 0, :, 2]) > 0)


def test_material_options_build():
    """The follow-up checks' variants compile: Stable Neo-Hookean (spec API only), J2 plasticity, hard contact."""
    for kw in ({"material": "snh"}, {"yield_strain": 0.07}, {"solimp": MJ.STIFF_CONTACT}):
        m, d, v0, order = MJ.build(MJ.MjSetup(count=(6, 4, 3), **kw), 117.0)
        assert (order >= 0).all()
    m, *_ = MJ.build(MJ.MjSetup(count=(6, 4, 3), yield_strain=0.07), 117.0)
    assert m.flex_yield[0, 0] == pytest.approx(0.07)


def test_valid_depth_stops_where_the_surface_leaves_the_cap():
    s = MJ.MjSetup(count=(6, 4, 3))
    _, _, v0, order = MJ.build(s, 117.0)
    rest = v0[order] * 1000.0
    d_tab = np.array([0.0, 5.0, 10.0, 15.0])
    disp = np.zeros((4,) + rest.shape)
    for k, sink in enumerate([0.0, 5.0, 10.0, 21.0]):          # the last one has collapsed past the cap
        disp[k][:, :, -1, 2] = -sink
    vd, surf = MJ.valid_depth(d_tab, rest, disp, 117.0)
    assert vd == 10.0 and surf[-1] == pytest.approx(21.0)
