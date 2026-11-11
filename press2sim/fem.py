"""A 3D finite-element model of the foam block: a voxel (hex) grid with linear elasticity.

The block (length x along the dotted face, depth y into the block with y = 0 the dotted face,
height z from the table) is cut into box elements on a tensor-product grid that is finer near the
dotted face and the top. The grid lines are placed exactly on the edges of whatever presses on the
top, so a footprint never falls between nodes. Stiffness is computed for E = 1 Pa; displacement
fields under prescribed displacements do not depend on E (only on Poisson's ratio), forces scale
with it.

The tracked dots are the "particles" of this model: each sits on the y = 0 face and moves with the
grid by bilinear interpolation of the face nodes, as particles are carried by a voxel grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:                                    # algebraic multigrid: ~10x faster than a direct solve here
    import pyamg
except ImportError:                     # pragma: no cover
    pyamg = None

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def _axis(length: float, fine: float, coarse: float, fine_until: float, must=()) -> np.ndarray:
    """Grid lines from 0 to length: spacing `fine` up to fine_until, then `coarse`, plus forced lines."""
    a = np.arange(0.0, min(fine_until, length), fine)
    b = np.arange(min(fine_until, length), length, coarse)
    pts = np.unique(np.r_[a, b, length, [m for m in must if 0 < m < length]])
    # merge lines closer than 35% of the local spacing (keeps forced lines, drops their neighbours)
    keep = [pts[0]]
    forced = set(np.round([m for m in must if 0 < m < length] + [0.0, length], 6))
    for p in pts[1:]:
        h = fine if p <= fine_until else coarse
        if p - keep[-1] < 0.35 * h:
            if round(p, 6) in forced and round(keep[-1], 6) not in forced:
                keep[-1] = p
            continue
        keep.append(p)
    return np.array(keep)


def _hex_stiffness(hx, hy, hz, nu):
    """Element stiffness (n, 24, 24) of box elements with sides hx, hy, hz (arrays), E = 1."""
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1.0 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    n = len(hx)
    K = np.zeros((n, 24, 24))
    jac = np.stack([hx / 2, hy / 2, hz / 2], 1)                    # (n, 3)
    detJ = jac.prod(1)
    for xi in _GP:
        for eta in _GP:
            for zeta in _GP:
                g = np.array([xi, eta, zeta])
                # dN/dxi for the 8 nodes: (8, 3)
                dN = np.empty((8, 3))
                for a, c in enumerate(_CORNERS):
                    f = 1 + c * g
                    dN[a] = 0.125 * c * np.array([f[1] * f[2], f[0] * f[2], f[0] * f[1]])
                dx = dN[None, :, :] / jac[:, None, :]                # (n, 8, 3)
                B = np.zeros((n, 6, 24))
                B[:, 0, 0::3] = dx[:, :, 0]
                B[:, 1, 1::3] = dx[:, :, 1]
                B[:, 2, 2::3] = dx[:, :, 2]
                B[:, 3, 0::3] = dx[:, :, 1]; B[:, 3, 1::3] = dx[:, :, 0]
                B[:, 4, 1::3] = dx[:, :, 2]; B[:, 4, 2::3] = dx[:, :, 1]
                B[:, 5, 0::3] = dx[:, :, 2]; B[:, 5, 2::3] = dx[:, :, 0]
                K += np.einsum("nij,jk,nkl->nil", B.transpose(0, 2, 1), D, B) * detJ[:, None, None]
    return K


@dataclass
class Solution:
    u: np.ndarray            # (nx, ny, nz, 3) nodal displacement per unit imposed displacement
    force: float             # total vertical reaction on the loaded nodes, N per (Pa of E x mm imposed)


class Block:
    """Hex grid of the block; footprint edges in x and y become grid lines."""

    def __init__(self, length=234.0, depth=139.5, height=73.0, nu=0.25, x_lines=(), y_lines=(),
                 fine=6.0, coarse=12.0, z_lines=()):
        self.L, self.D, self.H, self.nu = length, depth, height, nu
        self.x = _axis(length, fine, fine, length, x_lines)
        self.y = _axis(depth, fine, coarse, 80.0, y_lines)
        self.z = _axis(height, fine, fine, height, z_lines)
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        self.shape = (nx, ny, nz)
        self.n_nodes = nx * ny * nz
        idx = np.arange(self.n_nodes).reshape(nx, ny, nz)
        i, j, k = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        conn = np.stack([idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
                         idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1]], 1)
        hx, hy, hz = np.diff(self.x)[i], np.diff(self.y)[j], np.diff(self.z)[k]
        Ke = _hex_stiffness(hx, hy, hz, nu)
        dof = (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(len(conn), 24)
        rows = np.repeat(dof, 24, axis=1).ravel()
        cols = np.tile(dof, (1, 24)).ravel()
        n = 3 * self.n_nodes
        self.K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
        self.idx = idx

    def nodes_xyz(self):
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        return np.stack([X, Y, Z], -1)

    def solve(self, loaded: np.ndarray, stick: bool = True) -> Solution:
        """Unit downward displacement of the `loaded` top nodes (bool (nx, ny)); the base is fixed.

        stick: the loaded nodes cannot slide sideways (rough cap or pad on foam).
        """
        nx, ny, nz = self.shape
        n = 3 * self.n_nodes
        fixed = np.zeros(n, bool)
        val = np.zeros(n)
        base = self.idx[:, :, 0].ravel()
        for c in range(3):
            fixed[3 * base + c] = True
        top = self.idx[:, :, nz - 1][loaded]
        fixed[3 * top + 2] = True
        val[3 * top + 2] = -1.0
        if stick:
            fixed[3 * top] = True
            fixed[3 * top + 1] = True
        free = ~fixed
        Kff = self.K[free][:, free].tocsc()
        rhs = -self.K[free][:, fixed] @ val[fixed]
        u = val.copy()
        u[free] = spla.spsolve(Kff, rhs)
        reaction = self.K @ u
        force = float(-reaction[3 * top + 2].sum())         # N per (Pa x mm): compressive positive
        return Solution(u.reshape(nx, ny, nz, 3), force)

    def face(self, sol: Solution, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Displacement (x, z components) at points (x, z) on the dotted face y = 0, bilinear."""
        U = sol.u[:, 0, :, :]                                # (nx, nz, 3)
        xi = np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2)
        zi = np.clip(np.searchsorted(self.z, z) - 1, 0, len(self.z) - 2)
        fx = np.clip((x - self.x[xi]) / (self.x[xi + 1] - self.x[xi]), 0, 1)
        fz = np.clip((z - self.z[zi]) / (self.z[zi + 1] - self.z[zi]), 0, 1)
        out = ((1 - fx) * (1 - fz))[:, None] * U[xi, zi] + (fx * (1 - fz))[:, None] * U[xi + 1, zi] + \
              ((1 - fx) * fz)[:, None] * U[xi, zi + 1] + (fx * fz)[:, None] * U[xi + 1, zi + 1]
        return out[:, [0, 2]]


def cap_field(nu: float, x_c: float, y0: float, width: float = 60.0, depth: float = 60.0,
              block=(234.0, 139.5, 73.0), stick=True, fine=6.0):
    """FE solution for a flat square cap (width along x, depth along y) pushed 1 mm into the top.

    The cap's front edge is y0 mm behind the dotted face. Returns (Block, Solution).
    """
    L, D, H = block
    xs = (x_c - width / 2, x_c + width / 2)
    ys = (y0, y0 + depth)
    b = Block(L, D, H, nu, x_lines=xs, y_lines=ys, fine=fine)
    X, Y = np.meshgrid(b.x, b.y, indexing="ij")
    loaded = (X >= xs[0] - 1e-6) & (X <= xs[1] + 1e-6) & (Y >= ys[0] - 1e-6) & (Y <= ys[1] + 1e-6)
    return b, b.solve(loaded, stick=stick)


def disc_field(nu: float, x_c: float, y_c: float, radius: float, block=(234.0, 139.5, 73.0), stick=True, fine=6.0):
    """FE solution for a round flat contact (a fingertip) pushed 1 mm into the top."""
    L, D, H = block
    b = Block(L, D, H, nu, x_lines=(x_c - radius, x_c, x_c + radius), y_lines=(max(0.0, y_c - radius), y_c + radius),
              fine=min(fine, radius / 1.5))
    X, Y = np.meshgrid(b.x, b.y, indexing="ij")
    loaded = (X - x_c) ** 2 + (Y - y_c) ** 2 <= radius ** 2 + 1e-6
    return b, b.solve(loaded, stick=stick)


def plate_field(nu: float, block=(234.0, 139.5, 73.0), stick=True, fine=6.0):
    """FE solution for a rigid plate covering the whole top, pushed 1 mm down."""
    L, D, H = block
    b = Block(L, D, H, nu, fine=fine)
    loaded = np.ones((len(b.x), len(b.y)), bool)
    return b, b.solve(loaded, stick=stick)


def _center_B(hx, hy, hz):
    """Strain-displacement matrix (n, 6, 24) at the element centre (engineering shear strains)."""
    jac = np.stack([hx / 2, hy / 2, hz / 2], 1)
    dN = 0.125 * _CORNERS                                       # dN/dxi at xi = 0
    dx = dN[None, :, :] / jac[:, None, :]
    n = len(hx)
    B = np.zeros((n, 6, 24))
    B[:, 0, 0::3] = dx[:, :, 0]; B[:, 1, 1::3] = dx[:, :, 1]; B[:, 2, 2::3] = dx[:, :, 2]
    B[:, 3, 0::3] = dx[:, :, 1]; B[:, 3, 1::3] = dx[:, :, 0]
    B[:, 4, 1::3] = dx[:, :, 2]; B[:, 4, 2::3] = dx[:, :, 1]
    B[:, 5, 0::3] = dx[:, :, 2]; B[:, 5, 2::3] = dx[:, :, 0]
    return B


def foam_stress(eps, eps_y, h, eps_d=0.7, k_d=6.0):
    """Uniaxial compressive stress / E of the foam: linear up to eps_y, then a soft plateau of slope h,
    stiffening again as the cells close (densification) from eps_d. eps >= 0 is compression."""
    eps = np.asarray(eps, float)
    s = np.where(eps <= eps_y, eps, eps_y + h * (eps - eps_y))
    return s + np.where(eps > eps_d, k_d * (eps - eps_d) ** 2, 0.0)


class FoamBlock:
    """The block under one kind of load, solved for increasing depth with a crushable-foam material.

    Elements get a secant modulus from their compressive strain (foam_stress), iterated to
    convergence at each depth; with eps_y large the model is plain linear elasticity. Everything is
    per unit Young's modulus: forces come out in N per Pa of E.
    load: ("cap", x_c, y0, width, depth) | ("disc", x_c, y_c, radius) | ("plate",)
    """

    def __init__(self, nu, eps_y, h, load, block=(234.0, 139.5, 73.0), fine=6.0, stick=True, support=None, eps_d=0.7,
                 skin=None):
        L, D, H = block
        kind = load[0]
        if kind == "cap":
            _, x_c, y0, w, d = load
            xl, yl = (x_c - w / 2, x_c + w / 2), (y0, y0 + d)
        elif kind == "disc":
            _, x_c, y_c, r = load
            xl, yl = (x_c - r, x_c, x_c + r), (max(0.0, y_c - r), y_c, y_c + r)
            fine = min(fine, max(3.0, r / 1.5))
        else:
            xl, yl = (), ()
        # skin = (thickness mm, relative stiffness): a much softer top layer. It stands in for whatever keeps the
        # first millimetres of the cap's travel from reaching the dots (docs/model.md, Limits)
        z_extra = () if skin is None else (H - skin[0],)
        b = Block(L, D, H, nu, x_lines=xl, y_lines=yl, fine=fine, z_lines=z_extra)
        self.b, self.load, self.eps_y, self.h, self.eps_d = b, load, eps_y, h, eps_d
        X, Y = np.meshgrid(b.x, b.y, indexing="ij")
        if kind == "cap":
            loaded = (X >= xl[0] - 1e-6) & (X <= xl[1] + 1e-6) & (Y >= yl[0] - 1e-6) & (Y <= yl[1] + 1e-6)
        elif kind == "disc":
            loaded = (X - x_c) ** 2 + (Y - y_c) ** 2 <= r ** 2 + 1e-6
        else:
            loaded = np.ones_like(X, bool)
        nx, ny, nz = b.shape
        idx = b.idx
        i, j, k = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        conn = np.stack([idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
                         idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1]], 1)
        hx, hy, hz = np.diff(b.x)[i], np.diff(b.y)[j], np.diff(b.z)[k]
        self.Ke0 = _hex_stiffness(hx, hy, hz, nu).reshape(len(conn), -1)
        self.Bc = _center_B(hx, hy, hz)
        zc = 0.5 * (b.z[k] + b.z[k + 1])
        self.alpha = np.ones(len(conn)) if skin is None else np.where(zc > H - skin[0], skin[1], 1.0)
        self.dof = (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(len(conn), 24)
        n = 3 * b.n_nodes
        rows = np.repeat(self.dof, 24, axis=1).ravel(); cols = np.tile(self.dof, (1, 24)).ravel()
        key = rows.astype(np.int64) * n + cols
        ukey, self.pos = np.unique(key, return_inverse=True)
        self.K_rows, self.K_cols, self.n = (ukey // n).astype(np.int32), (ukey % n).astype(np.int32), n
        fixed = np.zeros(n, bool)
        self.val = np.zeros(n)
        base = idx[:, :, 0].ravel()
        for c in range(3 if support is None else 2):          # the base cannot slide
            fixed[3 * base + c] = True
        self.spring = None
        if support is not None:
            # compliant support under the base: springs of stiffness `support` * E per mm^2 of base
            ax = np.diff(b.x); ay = np.diff(b.y)
            wx = np.r_[ax, 0] / 2 + np.r_[0, ax] / 2
            wy = np.r_[ay, 0] / 2 + np.r_[0, ay] / 2
            area = np.outer(wx, wy).ravel()                      # base node tributary areas (mm^2)
            self.spring = sp.csr_matrix((support * area, (3 * base + 2, 3 * base + 2)), shape=(n, n))
        top = idx[:, :, nz - 1][loaded]
        fixed[3 * top + 2] = True
        self.val[3 * top + 2] = -1.0
        if stick:
            fixed[3 * top] = True; fixed[3 * top + 1] = True
        self.fixed, self.free, self.top = fixed, ~fixed, top
        self.E = self.alpha.copy()
        self.u = np.zeros(n)
        self.n_el = len(conn)

    def _K(self, E):
        data = np.bincount(self.pos, weights=(E[:, None] * self.Ke0).ravel(), minlength=len(self.K_rows))
        K = sp.csr_matrix((data, (self.K_rows, self.K_cols)), shape=(self.n, self.n))
        return K if self.spring is None else K + self.spring

    def _solve(self, K, delta, x0=None, M=None):
        free, fixed = self.free, self.fixed
        Kff = K[free][:, free].tocsr()
        rhs = -K[free][:, fixed] @ (delta * self.val[fixed])
        if pyamg is not None:
            if M is None:
                M = pyamg.smoothed_aggregation_solver(Kff).aspreconditioner(cycle="V")
            uf, _ = spla.cg(Kff, rhs, x0=x0, rtol=1e-7, maxiter=300, M=M)
        else:
            uf = spla.spsolve(Kff.tocsc(), rhs)
        u = delta * self.val
        u[free] = uf
        return u, M

    def strain_c(self, u):
        e = np.einsum("nij,nj->ni", self.Bc, u[self.dof])       # (n, 6) engineering strains
        T = np.empty((len(e), 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = e[:, 0], e[:, 1], e[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = e[:, 3] / 2
        T[:, 1, 2] = T[:, 2, 1] = e[:, 4] / 2
        T[:, 0, 2] = T[:, 2, 0] = e[:, 5] / 2
        return np.clip(-np.linalg.eigvalsh(T)[:, 0], 0, None)    # largest compressive principal strain

    def solve_depth(self, delta, iters=10, tol=0.03):
        """Equilibrium at indentation `delta` (mm), starting from the current state. Returns force (N/Pa)."""
        E = self.E.copy()
        scale = np.abs(self.u[self.fixed]).max() if self.u.any() else 0.0
        u = self.u * (delta / scale) if scale > 0 else None
        M = None
        for it in range(iters):
            K = self._K(E)
            u, M = self._solve(K, delta, x0=None if u is None else u[self.free], M=M if it % 4 else None)
            ec = self.strain_c(u)
            target = self.alpha * np.where(ec > 1e-9, foam_stress(ec, self.eps_y, self.h, self.eps_d) / np.maximum(ec, 1e-9), 1.0)
            E_new = np.sqrt(E * target)                           # relaxed secant update
            done = np.max(np.abs(np.log(E_new / E))) < tol
            E = E_new
            if done:
                break
        K = self._K(E)
        self.E, self.u = E, u
        reaction = K @ u
        return float(-reaction[3 * self.top + 2].sum()) * 1e-6     # N per Pa of E (mm -> m: 1e-6)

    def face_nodes(self):
        """Current displacement of the y = 0 face nodes, (nx, nz, 3) mm."""
        return self.u.reshape(*self.b.shape, 3)[:, 0, :, :]
