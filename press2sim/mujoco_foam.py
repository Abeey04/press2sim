"""The same foam block in MuJoCo, as a baseline for the purpose-built model.

MuJoCo (3.x/4.x) has deformable "flex" bodies: here a 3D grid of vertices over the whole block with
linear elasticity (Young's modulus, Poisson's ratio, stiffness-proportional damping), the base pinned
to the table and the cap as a kinematic (mocap) box that follows the measured cap depth, touching the
foam through MuJoCo's soft contacts. The fitted E goes in unchanged, with the model's dashpot (1.24 ms,
not fitted) as MuJoCo's damping; Poisson's ratio is chosen the same fair way as for our model (best fit
on the training presses).

What MuJoCo cannot represent: the foam's cells buckling (its material is linear), the soft top
layer, the compliant support, and the force relaxation. The comparison shows what those cost.

Numerics: the foam is very light (78 g) and comparatively stiff, so its elastic waves are fast
(~75 m/s); MuJoCo needs its implicit "discrete" integrator, and the mass is scaled up 100x with gravity
off so that a 1 ms step works. The presses take tenths of a second to seconds, far slower than the
block's elastic response even with the heavier mass, so the motion stays quasi-static.

MjSetup also carries the options scripts/mujoco_checks.py tries: MuJoCo's experimental Stable
Neo-Hookean material (which MJCF cannot select, so it is set through the spec API), its J2 plasticity,
and near-hard contact.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import model as M

try:
    import mujoco
except ImportError:                        # pragma: no cover
    mujoco = None


@dataclass
class MjSetup:
    E: float = 178e3              # Pa
    nu: float = 0.0
    damping: float = 0.0012       # s, stiffness-proportional (MuJoCo flex "damping")
    count: tuple = (24, 14, 8)    # vertices along x, y, z
    dt: float = 0.001
    mass_scale: float = 100.0
    friction: float = 2.0
    y0: float = 0.0               # cap's front edge behind the face (mm)
    # follow-up checks (defaults = the comparison in the README's table)
    material: str = "stvk"        # "stvk": MuJoCo's default Saint Venant-Kirchhoff; "snh": its experimental
                                  # Stable Neo-Hookean, which MJCF cannot select (set through the spec API)
    yield_strain: float = 0.0     # >0: MuJoCo's J2 (von Mises) plasticity with this yield strain
    solimp: str = ""              # contact impedance for cap and foam; "" = MuJoCo's default

STIFF_CONTACT = "0.99 0.9999 0.001 0.5 2"     # near-hard contact, needed once the foam pushes back hard


def build(setup: MjSetup, x_c_mm: float):
    if mujoco is None:
        raise RuntimeError("pip install mujoco")
    L, D, H = (v / 1000 for v in M.BLOCK)
    nx, ny, nz = setup.count
    st = 2 * setup.dt
    fr = f"{setup.friction} 0.005 0.0001"
    ys = f' yield="{setup.yield_strain}"' if setup.yield_strain > 0 else ""
    si = f' solimp="{setup.solimp}"' if setup.solimp else ""
    xml = f"""
<mujoco model="foam">
  <option timestep="{setup.dt}" gravity="0 0 0" integrator="discrete"/>
  <worldbody>
    <flexcomp name="foam" type="grid" dim="3" count="{nx} {ny} {nz}"
              spacing="{L / (nx - 1)} {D / (ny - 1)} {H / (nz - 1)}" pos="{L / 2} {D / 2} {H / 2}"
              radius="0.0005" mass="{0.078 * setup.mass_scale}" rgba="0.55 0.75 0.95 1">
      <contact condim="3" selfcollide="none" solref="{st} 1" friction="{fr}"{si}/>
      <edge equality="false"/>
      <elasticity young="{setup.E}" poisson="{setup.nu}" damping="{setup.damping}"{ys}/>
      <pin gridrange="0 0 0 {nx - 1} {ny - 1} 0"/>
    </flexcomp>
    <body name="cap" mocap="true" pos="{x_c_mm / 1000} {(setup.y0 + 30) / 1000} {H + 0.03}">
      <geom type="box" size="0.03 0.03 0.0095" rgba="1 0.82 0.1 1" solref="{st} 1" friction="{fr}"{si}/>
    </body>
  </worldbody>
</mujoco>"""
    if setup.material == "snh":
        spec = mujoco.MjSpec.from_string(xml)
        spec.flexes[0].elastic3d = 1          # experimental Stable Neo-Hookean (needs the discrete integrator)
        m = spec.compile()
    elif setup.material == "stvk":
        m = mujoco.MjModel.from_xml_string(xml)
    else:
        raise ValueError(f"unknown material {setup.material!r}")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    v0 = d.flexvert_xpos.copy()
    # grid index of every vertex
    ix = np.rint(v0[:, 0] / (L / (nx - 1))).astype(int)
    iy = np.rint(v0[:, 1] / (D / (ny - 1))).astype(int)
    iz = np.rint(v0[:, 2] / (H / (nz - 1))).astype(int)
    order = np.full((nx, ny, nz), -1)
    order[ix, iy, iz] = np.arange(len(v0))
    assert (order >= 0).all(), "vertex grid could not be indexed"
    return m, d, v0, order


def _cap_z(H_m, depth_mm):
    return H_m + 0.0095 - depth_mm / 1000.0


class MjRun:
    """Drive the cap through a depth history; keep the whole grid's movement (mm) when asked."""

    def __init__(self, setup: MjSetup, x_c_mm: float):
        self.setup = setup
        self.m, self.d, self.v0, self.order = build(setup, x_c_mm)
        self.H = M.BLOCK[2] / 1000
        self.x_c = x_c_mm
        self.unstable = False
        self.d.mocap_pos[0] = [x_c_mm / 1000, (setup.y0 + 30) / 1000, _cap_z(self.H, -2.0)]
        # top vertices under the cap's footprint: MuJoCo's soft contact lets the cap sink into them a
        # little, so the cap is driven by where the foam surface under it is, not by the cap itself
        top = self.order[:, :, -1].ravel()
        xy = self.v0[top, :2] * 1000.0
        inside = (np.abs(xy[:, 0] - x_c_mm) < 28) & (xy[:, 1] >= setup.y0) & (xy[:, 1] <= setup.y0 + 58)
        self.under = top[inside]
        self.pen = 0.0

    def surface_depth(self):
        """Depth of the foam surface under the cap (mm), from its top vertices."""
        return (self.H - float(np.mean(self.d.flexvert_xpos[self.under, 2]))) * 1000.0

    def step_to(self, depth_mm: float, n: int = 1):
        for _ in range(n):
            target = depth_mm if np.isfinite(depth_mm) else -60.0
            if target > 0.3:
                # surface lagging the commanded depth = the contact's give: push the cap that much further
                gap = target - self.surface_depth()
                self.pen = float(np.clip(self.pen + 0.2 * gap, 0.0, 8.0))
            else:
                self.pen *= 0.9
            self.d.mocap_pos[0][2] = _cap_z(self.H, target + self.pen)
            mujoco.mj_step(self.m, self.d)
        if self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            self.unstable = True

    def grid_disp(self):
        """(nx, ny, nz, 3) movement of every vertex, mm."""
        return (self.d.flexvert_xpos - self.v0)[self.order] * 1000.0

    def grid_rest(self):
        return self.v0[self.order] * 1000.0


def face_at(grid_rest_mm, grid_disp_mm, rest_xz: np.ndarray):
    """Bilinear movement (x, z) of points on the front face (y = 0) from the vertex grid."""
    xs = grid_rest_mm[:, 0, 0, 0]
    zs = grid_rest_mm[0, 0, :, 2]
    U = grid_disp_mm[:, 0]                                  # (nx, nz, 3)
    x, z = rest_xz[:, 0], rest_xz[:, 1]
    xi = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    zi = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
    fx = np.clip((x - xs[xi]) / (xs[xi + 1] - xs[xi]), 0, 1)[:, None]
    fz = np.clip((z - zs[zi]) / (zs[zi + 1] - zs[zi]), 0, 1)[:, None]
    u = (1 - fx) * (1 - fz) * U[xi, zi] + fx * (1 - fz) * U[xi + 1, zi] + (1 - fx) * fz * U[xi, zi + 1] + fx * fz * U[xi + 1, zi + 1]
    return u[:, [0, 2]]


def static_table(setup: MjSetup, x_c_mm: float, depths=M.DEPTHS, speed_mm_s=60.0, settle_s=0.08, log=None,
                 cache: Path | None = None):
    """MuJoCo's equilibrium at each depth: the cap is pushed in slowly and held at every tabulated depth.

    Returns depths reached, the vertex grid at rest (mm) and its movement at each depth (K, nx, ny, nz, 3).
    """
    import hashlib, json
    key = hashlib.md5(json.dumps([setup.E, setup.nu, setup.damping, list(setup.count), setup.dt, setup.mass_scale,
                                  setup.friction, setup.y0, x_c_mm, list(map(float, depths)), speed_mm_s,
                                  settle_s] + ([] if (setup.material, setup.yield_strain, setup.solimp) == ("stvk", 0.0, "")
                                                else [setup.material, setup.yield_strain, setup.solimp])).encode()).hexdigest()[:16]
    f = None if cache is None else Path(cache) / f"mujoco_{key}.npz"
    if f is not None and f.exists():
        z = np.load(f)
        return z["d"], z["rest"], z["disp"]
    run = MjRun(setup, x_c_mm)
    dt = setup.dt
    out_d, out_u = [0.0], [np.zeros_like(run.grid_rest())]
    depth = -2.0
    t0 = time.time()
    for target in depths[1:]:
        while depth < target:
            depth = min(target, depth + speed_mm_s * dt)
            run.step_to(depth)
        acc = []
        for k in range(int(settle_s / dt)):
            run.step_to(depth)
            if k >= int(0.75 * settle_s / dt):
                acc.append(run.grid_disp())
        if run.unstable or not np.all(np.isfinite(acc)):
            if log:
                log(f"  MuJoCo went unstable near {target:.0f} mm: table stops at {out_d[-1]:.0f} mm")
            break
        out_d.append(float(target)); out_u.append(np.mean(acc, axis=0))
        if log:
            log(f"  MuJoCo static: {target:4.1f} mm ({time.time() - t0:.0f} s)")
    res = (np.array(out_d), run.grid_rest(), np.stack(out_u))
    if f is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, d=res[0], rest=res[1], disp=res[2])
    return res


def dynamic_run(setup: MjSetup, cd: M.ClipData, x_c_fe: float, t_start=None, t_stop=None, log=None):
    """Drive MuJoCo's cap with a clip's measured depth over time; movement of every vertex at each frame.

    Returns the frame indices simulated, the vertex grid at rest and its movement per frame (mm).
    """
    run = MjRun(setup, x_c_fe)
    dt = setup.dt
    ev = cd.events
    t_a = ev["t_contact"] - 0.3 if t_start is None else t_start
    t_b = cd.t_end if t_stop is None else t_stop
    frames = np.where((cd.t >= t_a) & (cd.t <= t_b))[0]
    depth = np.where(np.isfinite(cd.depth), cd.depth, -60.0)
    # settle at the starting depth
    for _ in range(int(0.15 / dt)):
        run.step_to(max(depth[frames[0]], -2.0))
    t_sim = cd.t[frames[0]]
    out = []
    t0 = time.time()
    for n, k in enumerate(frames):
        while t_sim < cd.t[k] - 1e-9:
            t_sim += dt
            run.step_to(float(np.interp(t_sim, cd.t, depth)))
        out.append(run.grid_disp().astype(np.float32))
        if log and n % 400 == 0:
            log(f"  MuJoCo {cd.code}: {cd.t[k] - cd.t[frames[0]]:.2f} s of {cd.t[frames[-1]] - cd.t[frames[0]]:.2f} s "
                f"({time.time() - t0:.0f} s)")
    return frames, run.grid_rest(), np.stack(out), run.unstable


def cap_force(m, d) -> float:
    """Total normal contact force on the cap (N); the cap is the only thing the foam touches."""
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        tot += f6[0]
    return tot


def surface_follow(d_tab, grid_rest_mm, grid_disp_mm, x_c_mm, y0=0.0):
    """Mean depth (mm) of the foam surface under the cap's footprint at each tabulated cap depth."""
    xs = grid_rest_mm[:, 0, 0, 0]
    ys = grid_rest_mm[0, :, 0, 1]
    under = (np.abs(xs[:, None] - x_c_mm) < 28) & (ys[None, :] >= y0) & (ys[None, :] <= y0 + 58)
    return np.array([-float(grid_disp_mm[k][:, :, -1, 2][under].mean()) for k in range(len(d_tab))])


def valid_depth(d_tab, grid_rest_mm, grid_disp_mm, x_c_mm, y0=0.0, tol_mm=2.0):
    """Deepest tabulated depth up to which the surface under the cap stayed within tol of the cap.

    Past it the table means something else: with StVK the elements under the cap collapse and the
    surface sinks below the cap; with a stiffer material the contact gives way and the cap sinks
    into the foam."""
    s = surface_follow(d_tab, grid_rest_mm, grid_disp_mm, x_c_mm, y0)
    last = 0
    for k in range(len(d_tab)):
        if abs(s[k] - d_tab[k]) > tol_mm:
            break
        last = k
    return float(d_tab[last]), s


def press_cycle(setup: MjSetup, x_c_mm: float = 117.0, depth_mm=30.0, speed_mm_s=100.0, hold_s=0.05, rest_s=0.3):
    """Press to depth, hold, lift clear and wait: force against surface depth, collapse and what stays."""
    run = MjRun(setup, x_c_mm)
    dt = setup.dt
    hist = []

    def step(target):
        run.step_to(target)
        hist.append((target, run.surface_depth(), cap_force(run.m, run.d)))

    z = -2.0
    while z < depth_mm:
        z = min(depth_mm, z + speed_mm_s * dt); step(z)
    for _ in range(int(hold_s / dt)):
        step(depth_mm)
    rest, disp = run.grid_rest(), run.grid_disp()
    ix = int(np.argmin(np.abs(rest[:, 0, 0, 0] - x_c_mm)))
    zc = rest[ix, :, :, 2] + disp[ix, :, :, 2]
    inverted = int(np.sum(np.diff(zc, axis=1) < 0))
    surface_at_max = run.surface_depth()
    while z > -5.0:
        z = max(-5.0, z - speed_mm_s * dt); step(z)
    for _ in range(int(rest_s / dt)):
        step(-5.0)
    h = np.array(hist)
    load = h[: int(np.ceil((depth_mm + 2.0) / (speed_mm_s * dt)))]
    load = load[load[:, 0] > 0]
    reached = float(load[:, 1].max())
    f_at = {f"{dd:g} mm": float(np.interp(dd, np.maximum.accumulate(load[:, 1]), load[:, 2]))
            for dd in (5, 10, 15, 20, 25, 30) if dd <= reached}
    return {"force_N_at_surface_depth": f_at, "peak_force_N": float(h[:, 2].max()),
            "surface_depth_at_hold_mm": float(surface_at_max), "inverted_layer_pairs_at_hold": inverted,
            "dent_after_release_mm": float(run.surface_depth()), "unstable": bool(run.unstable)}
