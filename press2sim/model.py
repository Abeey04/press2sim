"""Part 02: a 3D model of the foam block with effective stiffness and damping.

Space: fem.FoamBlock. The block is a grid of hexahedral voxels with a crushable-foam material
(linear up to a strain eps_y, then a soft plateau of relative slope h, stiffening again as the cells
close), pressed by a flat rigid square (the cap) or a rigid plate (the pad and weights). For a given
indentation it gives the movement of every dot on the dotted face and the force per unit Young's
modulus E.

Time: Fung's quasi-linear viscoelasticity (QLV) with a dashpot in parallel. The force on the foam is

    F(t) = E [ g_inf R(q) + sum_i h_i + tau_v dR(q)/dt ],      dh_i/dt = g_i dR(q)/dt - h_i / tau_i,

with R(q) the elastic force at indentation q from the FE and g_inf + sum_i g_i = 1. The h_i make the
force relax while the cap holds still; the dashpot (tau_v) limits how fast the foam can spring back.
While the cap touches the foam, q is the cap's depth. When the foam would have to pull on the cap
(F < 0), the two part and q follows F = 0: the foam rises on its own, no faster than the dashpot
lets it. Under a dead weight (the K clips), F = m g.

Parameters: shape (nu, eps_y, h), the cap's setback y0 from the dotted face, damping
(g_1, tau_1, g_2, tau_2) and stiffness E. The model is driven only by the cap's measured motion (or a
weight's mass); the dots are always predicted, never replayed from the clip being predicted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import fem

G_ACC = 9.81
BLOCK = (234.0, 139.5, 73.0)
CAP = (60.0, 60.0)                              # cap footprint along x and y (mm)
DEPTHS = np.array([0.0, 1.5, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0, 56.0])
FINE = 7.5                                      # FE grid spacing near the face and top (mm)
PLATE_DEPTHS = np.array([0.0, 0.25, 0.5, 1.0, 1.6, 2.5])


@dataclass
class Params:
    nu: float = 0.1               # Poisson's ratio (small strain)
    eps_y: float = 0.06           # strain where the foam's cells start to buckle
    h: float = 0.10               # plateau slope / E
    eps_d: float = 0.7            # strain where the crushed cells start to touch (densification)
    y0: float = 0.0               # cap's front edge behind the dotted face (mm)
    g: tuple = (0.15,)            # relaxing fraction(s) of the stiffness
    tau: tuple = (0.3,)           # their time constant(s) (s)
    tau_v: float = 0.00124        # dashpot in parallel: viscous force = E tau_v dR/dt (s). Not fitted: these
                                  # clips cannot tell 0 from 20 ms, so it stays at what an earlier fit returned
    E: float = 30e3               # Young's modulus at small strain (Pa)
    support: float = 0.02         # stiffness of what the block stands on, per unit E (1/mm)
    skin_t: float = 0.0           # softer layer at the top surface: thickness (mm, 0 = none)
    skin_k: float = 1.0           # ... and its stiffness relative to the rest

    @property
    def g_inf(self):
        return 1.0 - float(sum(self.g))

    def shape_key(self):
        return (round(self.nu, 4), round(self.eps_y, 5), round(self.h, 5), round(self.eps_d, 4), round(self.y0, 3),
                round(self.support, 5), round(self.skin_t, 3), round(self.skin_k, 4))


# ---------------------------------------------------------------------------------------------- data

@dataclass
class ClipData:
    code: str
    kind: str
    t: np.ndarray                 # (T,) s
    rest: np.ndarray              # (N, 2) dot rest positions: x from the block's left end, z from its base (mm)
    disp: np.ndarray              # (T, N, 2) measured movement from rest (mm), NaN where not tracked
    k0: int
    depth: np.ndarray | None = None   # (T,) cap depth below the undisturbed top (mm), NaN where unknown
    x_c: float | None = None
    z_top: float = BLOCK[2]
    events: dict = field(default_factory=dict)
    load_kg: float | None = None
    t_end: float = 0.0            # last usable time (before the tap that stops the recording)


def load_clip(tracks_dir: Path, code: str, kind: str, load_kg=None, grid=105) -> ClipData:
    A = np.load(Path(tracks_dir) / f"{code}.npz")
    meta = json.loads((Path(tracks_dir) / f"{code}.json").read_text())
    t = A["t"]
    disp = A["disp_mm"][:, :grid].astype(float)
    rest = A["rest_mm"][:grid].astype(float)
    k0 = int(A["k0"])
    cd = ClipData(code, kind, t, rest, disp, k0, events=meta.get("events", {}), load_kg=load_kg)
    cd.t_end = float(t[-1]) - 0.6
    if "cap_z_mm" in A and "foam_top_mm" in A:
        cd.z_top = float(A["foam_top_mm"])
        cd.depth = cd.z_top - A["cap_z_mm"].astype(float)
        cd.x_c = float(meta["indenter"]["rest_x_mm"])
    return cd


# ------------------------------------------------------------------------------------- FE depth tables

@dataclass
class Table:
    d: np.ndarray                 # (K,) indentation (mm)
    R: np.ndarray                 # (K,) elastic force per unit E (N/Pa)
    face: np.ndarray              # (K, nx, nz, 3) face-node movement (mm)
    xn: np.ndarray
    zn: np.ndarray
    x_c: float                    # cap centre in the FE (mm from the left end); NaN for the plate

    def at(self, rest: np.ndarray, x_c=None, z_top=BLOCK[2]) -> np.ndarray:
        """(K, N, 2) movement of dots at rest positions `rest` (x, z), for every tabulated depth.

        Dots are shifted so the cap sits where it was in the clip, and heights scaled so the clip's
        measured top matches the block's.
        """
        x = rest[:, 0] + (0.0 if x_c is None or not np.isfinite(self.x_c) else self.x_c - x_c)
        z = rest[:, 1] * BLOCK[2] / z_top
        xi = np.clip(np.searchsorted(self.xn, x) - 1, 0, len(self.xn) - 2)
        zi = np.clip(np.searchsorted(self.zn, z) - 1, 0, len(self.zn) - 2)
        fx = np.clip((x - self.xn[xi]) / (self.xn[xi + 1] - self.xn[xi]), 0, 1)[None, :, None]
        fz = np.clip((z - self.zn[zi]) / (self.zn[zi + 1] - self.zn[zi]), 0, 1)[None, :, None]
        F = self.face
        out = (1 - fx) * (1 - fz) * F[:, xi, zi] + fx * (1 - fz) * F[:, xi + 1, zi] + \
              (1 - fx) * fz * F[:, xi, zi + 1] + fx * fz * F[:, xi + 1, zi + 1]
        bad = ~np.isfinite(x) | ~np.isfinite(z)
        out[:, bad] = np.nan
        return out[..., [0, 2]]


def build_table(p: Params, load=("cap", 117.0), cache: Path | None = None, depths=None, fine=FINE) -> Table:
    """FE solutions for increasing depth, cached on disk by parameters."""
    kind = load[0]
    if depths is None:
        depths = PLATE_DEPTHS if kind == "plate" else DEPTHS
    if kind == "cap":
        spec = ("cap", float(load[1]), p.y0, CAP[0], CAP[1])
    elif kind == "disc":
        spec = ("disc", float(load[1]), float(load[2]), float(load[3]))
    else:
        spec = ("plate",)
    skin = None if p.skin_t <= 0 or p.skin_k >= 1 else (p.skin_t, p.skin_k)
    key = hashlib.md5(json.dumps([p.nu, p.eps_y, p.h, p.eps_d, p.support, spec, list(map(float, depths)), fine]
                                 + ([list(skin)] if skin else [])).encode()).hexdigest()[:16]
    if cache is not None:
        f = Path(cache) / f"table_{key}.npz"
        if f.exists():
            z = np.load(f)
            return Table(z["d"], z["R"], z["face"], z["xn"], z["zn"], float(z["x_c"]))
    fb = fem.FoamBlock(p.nu, p.eps_y, p.h, spec, block=BLOCK, fine=fine, support=p.support, eps_d=p.eps_d, skin=skin)
    R, face = [0.0], [np.zeros((len(fb.b.x), len(fb.b.z), 3))]
    for d in depths[1:]:
        R.append(fb.solve_depth(float(d)))
        face.append(fb.face_nodes().copy())
    R = np.maximum.accumulate(np.array(R)) + np.arange(len(R)) * 1e-12     # strictly increasing
    tab = Table(np.asarray(depths, float), R, np.stack(face), fb.b.x, fb.b.z,
                float(spec[1]) if kind in ("cap", "disc") else np.nan)
    if cache is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(cache) / f"table_{key}.npz", d=tab.d, R=tab.R, face=tab.face, xn=tab.xn,
                            zn=tab.zn, x_c=tab.x_c)
    return tab


def field_at(Uk: np.ndarray, d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Movement at depths q (T,) from tabulated Uk (K, N, 2) at depths d (K,): piecewise linear."""
    q = np.clip(np.nan_to_num(q, nan=0.0), d[0], d[-1])
    i = np.clip(np.searchsorted(d, q) - 1, 0, len(d) - 2)
    w = ((q - d[i]) / (d[i + 1] - d[i]))[:, None, None]
    return (1 - w) * Uk[i] + w * Uk[i + 1]


# ------------------------------------------------------------------------------------------ dynamics

def qlv(t, d_tab, R_tab, p: Params, delta=None, force=None):
    """Indentation q(t), force F(t)/E and contact flag under the cap's depth `delta` or a force (N).

    delta: (T,) cap depth, NaN or None where there is no cap. force: (T,) newtons (dead weight).
    """
    g = np.asarray(p.g, float)
    tau = np.asarray(p.tau, float)
    g_inf = p.g_inf
    T = len(t)
    q_out = np.zeros(T); F_out = np.zeros(T); c_out = np.zeros(T, bool)
    h = np.zeros(len(g)); Rq = 0.0
    for k in range(T):
        dt = t[k] - t[k - 1] if k else 0.0
        if dt > 0:
            e = np.exp(-dt / tau)
            a = tau / dt * (1 - e)
        else:
            e = np.ones_like(tau); a = np.ones_like(tau)
        target = 0.0 if force is None else force[k] / p.E
        v = p.tau_v / dt if dt > 0 else 0.0
        R_free = (target - np.dot(h, e) + np.dot(g, a) * Rq + v * Rq) / (g_inf + np.dot(g, a) + v)
        R_free = max(R_free, 0.0)
        q_free = float(np.interp(R_free, R_tab, d_tab))
        dk = -np.inf if delta is None or not np.isfinite(delta[k]) else float(delta[k])
        if force is None and q_free < dk:
            q_new = min(dk, d_tab[-1]); R_new = float(np.interp(q_new, d_tab, R_tab)); c_out[k] = True
        else:
            q_new, R_new = q_free, float(np.interp(q_free, d_tab, R_tab))
        h = h * e + g * a * (R_new - Rq)
        q_out[k] = q_new
        F_out[k] = g_inf * R_new + h.sum() + v * (R_new - Rq)
        Rq = R_new
    return q_out, F_out, c_out


def predict_clip(cd: ClipData, tab: Table, p: Params):
    """Predicted dot movement (T, N, 2) for a press clip, from the cap's depth alone."""
    Uk = tab.at(cd.rest, cd.x_c, cd.z_top)
    delta = cd.depth.copy()
    delta[:cd.k0] = np.nan
    q, F, c = qlv(cd.t, tab.d, tab.R, p, delta=delta)
    q_rest = q[cd.k0]
    U = field_at(Uk, tab.d, q) - field_at(Uk, tab.d, np.array([q_rest]))
    U[:cd.k0] = np.nan
    return U, q, F, c


def predict_weight(cd: ClipData, tab: Table, p: Params, t_on: float, t_off=None, ramp_s: float = 0.15):
    """Predicted dot movement under a dead weight put on at t_on and lifted off at t_off (pad clips)."""
    Uk = tab.at(cd.rest, None, cd.z_top)
    F = cd.load_kg * G_ACC * np.clip((cd.t - t_on) / ramp_s, 0, 1)
    if t_off is not None:
        F = F * (1 - np.clip((cd.t - t_off) / ramp_s, 0, 1))
    q, Fe, _ = qlv(cd.t, tab.d, tab.R, p, force=F)
    U = field_at(Uk, tab.d, q)
    return U, q


# ------------------------------------------------------------------------------------------ errors

def rms(a):
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else np.nan


def window(cd: ClipData, start=None, end=None):
    t0 = cd.t[cd.k0] if start is None else start
    t1 = cd.t_end if end is None else end
    return (cd.t >= t0) & (cd.t <= t1)


def contact_frames(cd: ClipData, n=16, min_depth=0.8):
    """Frames spread over the depths the cap went through while pressing (loading and hold)."""
    ev = cd.events
    sel = window(cd, ev.get("t_contact", cd.t[cd.k0]), ev.get("t_lift", cd.t_end))
    sel &= np.isfinite(cd.depth) & (cd.depth > min_depth)
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return idx
    order = idx[np.argsort(cd.depth[idx])]
    pick = np.unique(np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int))
    return order[pick]


@dataclass
class FullTable:
    """Whole-block solutions for increasing depth, for drawing the simulated block in 3D."""
    d: np.ndarray                 # (K,) indentation (mm)
    R: np.ndarray                 # (K,) elastic force per unit E (N/Pa)
    u: np.ndarray                 # (K, nx, ny, nz, 3) node movement (mm)
    xn: np.ndarray
    yn: np.ndarray
    zn: np.ndarray
    load: tuple

    def at_depth(self, q: float) -> np.ndarray:
        q = float(np.clip(q, self.d[0], self.d[-1]))
        i = int(np.clip(np.searchsorted(self.d, q) - 1, 0, len(self.d) - 2))
        w = (q - self.d[i]) / (self.d[i + 1] - self.d[i])
        return (1 - w) * self.u[i] + w * self.u[i + 1]


def build_full(p: Params, load=("cap", 117.0), cache: Path | None = None, depths=None, fine=FINE) -> FullTable:
    """Like build_table, keeping every node (cached separately)."""
    kind = load[0]
    if depths is None:
        depths = PLATE_DEPTHS if kind == "plate" else DEPTHS
    if kind == "cap":
        spec = ("cap", float(load[1]), p.y0, CAP[0], CAP[1])
    elif kind == "disc":
        spec = ("disc", float(load[1]), float(load[2]), float(load[3]))
    else:
        spec = ("plate",)
    skin = None if p.skin_t <= 0 or p.skin_k >= 1 else (p.skin_t, p.skin_k)
    key = hashlib.md5(json.dumps(["full", p.nu, p.eps_y, p.h, p.eps_d, p.support, spec, list(map(float, depths)), fine]
                                 + ([list(skin)] if skin else [])).encode()).hexdigest()[:16]
    f = None if cache is None else Path(cache) / f"full_{key}.npz"
    if f is not None and f.exists():
        z = np.load(f)
        return FullTable(z["d"], z["R"], z["u"], z["xn"], z["yn"], z["zn"], spec)
    fb = fem.FoamBlock(p.nu, p.eps_y, p.h, spec, block=BLOCK, fine=fine, support=p.support, eps_d=p.eps_d, skin=skin)
    R, U = [0.0], [np.zeros((*fb.b.shape, 3))]
    for d in depths[1:]:
        R.append(fb.solve_depth(float(d)))
        U.append(fb.u.reshape(*fb.b.shape, 3).copy())
    R = np.maximum.accumulate(np.array(R)) + np.arange(len(R)) * 1e-12
    tab = FullTable(np.asarray(depths, float), R, np.stack(U).astype(np.float32), fb.b.x, fb.b.y, fb.b.z, spec)
    if f is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, d=tab.d, R=tab.R, u=tab.u, xn=tab.xn, yn=tab.yn, zn=tab.zn)
    return tab
