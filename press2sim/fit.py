"""Fit the block model to the training clips, then predict the held-out ones.

Order matters and keeps each step small:
1. shape (nu, eps_y, h, y0): only frames where the cap is pressing. With the cap's depth imposed,
   quasi-linear viscoelasticity leaves the deformation's shape purely elastic, so these four
   parameters can be fitted without knowing the damping;
2. damping (g, tau): the dent the cap leaves behind in the training presses, measured at the top-row
   dots under the cap from the moment the foam lets go of it. The dashpot is not fitted;
3. stiffness E: the dead-weight clips, where the force is known (m g). Hand presses cannot give E:
   the hand's force was never measured.
The held-out clips (a fresh press, an off-centre press, a fingertip) are never used for fitting.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from . import model as M

SHAPE_BOUNDS = {"nu": (0.0, 0.45), "eps_y": (0.02, 0.4), "h": (0.003, 1.0), "y0": (-10.0, 25.0), "eps_d": (0.4, 0.95),
                "skin_t": (1.0, 9.0), "skin_k": (0.02, 1.0)}
SHAPE_STEPS = {"nu": 0.05, "eps_y": 0.02, "h": 0.04, "y0": 3.0, "eps_d": 0.06, "skin_t": 1.0, "skin_k": 0.03}
# Where the shape search starts when no earlier fit is given. These values were set by hand from earlier
# exploratory fits, with the soft top already on: the search changes one parameter at a time, and the soft
# top only exists once it has both a thickness and a softness.
SHAPE_START = {"nu": 0.0, "eps_y": 0.06, "h": 0.01, "eps_d": 0.64, "y0": -3.0, "skin_t": 3.0, "skin_k": 0.1}


def _log(msg, log):
    if log:
        log(msg)


# ------------------------------------------------------------------------------------------ shape

def shape_loss(p: M.Params, clips, cache, frames=None) -> float:
    tab = M.build_table(p, ("cap", 117.0), cache=cache)
    errs = []
    for cd in clips:
        Uk = tab.at(cd.rest, cd.x_c, cd.z_top)
        fr = frames[cd.code] if frames else M.contact_frames(cd)
        d_rest = max(0.0, float(np.nan_to_num(cd.depth[cd.k0])))
        U = M.field_at(Uk, tab.d, cd.depth[fr]) - M.field_at(Uk, tab.d, np.array([d_rest]))
        errs.append((U - cd.disp[fr]).ravel())
    return M.rms(np.concatenate(errs))


def fit_shape(clips, cache, p0: M.Params, passes=2, log=None,
              names=("skin_k", "skin_t", "y0", "eps_y", "h", "eps_d", "nu"), steps=None):
    """Coordinate search: each parameter tried one step either side of the best, step halved per pass."""
    frames = {cd.code: M.contact_frames(cd) for cd in clips}
    best = p0
    best_loss = shape_loss(best, clips, cache, frames)
    _log(f"  shape start: RMS {best_loss:.3f} mm  {best.shape_key()}", log)
    steps = dict(SHAPE_STEPS, **(steps or {}))
    history = [(best.shape_key(), best_loss)]
    if best.skin_t <= 0 or best.skin_k >= 1:
        # no soft top yet: changing its thickness or its softness alone leaves the model as it is, so the
        # one-at-a-time search below could never switch it on. Try the two together first.
        cand = replace(best, skin_t=SHAPE_START["skin_t"], skin_k=SHAPE_START["skin_k"])
        loss = shape_loss(cand, clips, cache, frames)
        history.append((cand.shape_key(), loss))
        _log(f"  shape with a soft top ({cand.skin_t:g} mm at {cand.skin_k:g} E): RMS {loss:.3f} mm", log)
        if loss < best_loss:
            best, best_loss = cand, loss
    for ps in range(passes):
        for name in names:
            lo, hi = SHAPE_BOUNDS[name]
            for sgn in (-1, 1):
                v = float(np.clip(getattr(best, name) + sgn * steps[name], lo, hi))
                if abs(v - getattr(best, name)) < 1e-9:
                    continue
                cand = replace(best, **{name: v})
                t0 = time.time()
                loss = shape_loss(cand, clips, cache, frames)
                history.append((cand.shape_key(), loss))
                _log(f"  shape {name}={v:.4g}: RMS {loss:.3f} mm ({time.time() - t0:.0f} s)", log)
                if loss < best_loss:
                    best, best_loss = cand, loss
                    break                       # keep going in this direction next pass
            steps[name] *= 0.5
    return best, best_loss, history


# ------------------------------------------------------------------------------------------ damping

def _time_vec(p):
    g1, = p.g; t1, = p.tau
    return np.array([np.log(g1 / (1 - g1)), np.log(t1)])


def _vec_time(v, p):
    g1 = 1 / (1 + np.exp(-v[0]))
    return replace(p, g=(float(g1),), tau=(float(np.exp(v[1])),))


def recovery_frames(cd, before=0.05, after=2.5):
    """From just before the cap starts to lift until the foam has mostly come back."""
    tl = cd.events.get("t_lift")
    if tl is None:
        return np.zeros(len(cd.t), bool)
    return (cd.t >= tl - before) & (cd.t <= min(tl + after, cd.t_end))


def recovery_loss(p: M.Params, clips, tab):
    """RMS error (mm) of all dots around the lift. Reported, not fitted: the dent is a small part of it."""
    errs = []
    for cd in clips:
        U, q, F, c = M.predict_clip(cd, tab, p)
        sel = recovery_frames(cd)
        errs.append((U[sel] - cd.disp[sel]).ravel())
    return M.rms(np.concatenate(errs))


TOP_ROW = np.arange(6 * 15, 7 * 15)


def dent_dots(cd, half_width=M.CAP[0] / 2):
    """The top-row dots under the cap, where the foam is left dented when the cap lifts."""
    return TOP_ROW[np.abs(cd.rest[TOP_ROW, 0] - cd.x_c) <= half_width]


def dent_frames(cd):
    """From the moment the foam lets go of the cap (as measured) to the end of the clip."""
    tr = cd.events.get("t_release")
    if tr is None:
        return np.zeros(len(cd.t), bool)
    return (cd.t >= tr) & (cd.t <= cd.t_end)


def dent_error(cd, U):
    """Error (frames, dots) in the vertical movement of the dots under the cap, after the foam lets go."""
    sel = dent_frames(cd)
    j = dent_dots(cd)
    return U[sel][:, j, 1] - cd.disp[sel][:, j, 1]


def time_loss(p: M.Params, clips, tab):
    """RMS error (mm) of the dent the cap leaves behind, over the clips."""
    return M.rms(np.concatenate([dent_error(cd, M.predict_clip(cd, tab, p)[0]).ravel() for cd in clips]))


def fit_time(clips, tab, p0: M.Params, log=None, maxiter=300):
    """Damping from where it shows: the dent under the cap after the foam lets go of it.

    While the cap presses, its depth sets the dots and damping changes nothing. Once the foam lets go, how
    fast the dent closes is all damping. Scored over all 105 dots that signal is drowned by the shape error
    elsewhere (such a score cannot tell 12% relaxation from none), so the score here is the vertical movement
    of the top-row dots under the cap, from the measured release to the end of the clip.

    Two numbers are fitted: the relaxing fraction g and its time constant tau. The dashpot tau_v is left as
    it is in p0. Neither the dent nor the moment of release can tell a dashpot of 0 from one of 20 ms (the
    dent score moves by under 1%, the release time not at all; scripts/model_checks.py shows both), and a
    fitted value would only add an unfounded viscous force while the cap presses fast.
    """
    clips = [cd for cd in clips if cd.events.get("t_release") is not None]
    if len(p0.g) != 1:
        p0 = replace(p0, g=(0.15,), tau=(0.3,))
    f = lambda v: time_loss(_vec_time(v, p0), clips, tab)
    v0 = _time_vec(p0)
    res = minimize(f, v0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5,
                                                          "initial_simplex": v0 + np.vstack([np.zeros(2), np.eye(2) * 0.8])})
    p = _vec_time(res.x, p0)
    none = time_loss(replace(p, g=(1e-9,), tau_v=1e-9), clips, tab)
    _log(f"  damping: relaxing fraction {p.g[0]:.3f} with tau {p.tau[0]:.3f} s (dashpot held at {p.tau_v * 1e3:.2f} ms); "
         f"dent under the cap after release RMS {res.fun:.3f} mm (no damping {none:.3f} mm, {res.nfev} runs)", log)
    return p, float(res.fun), float(none)


# ------------------------------------------------------------------------------------------ stiffness

def plate_amplitude(cd: M.ClipData, tab: M.Table):
    """Measured plate sink w(t) (mm): the dot field projected on the model's plate pattern."""
    Uk = tab.at(cd.rest, None, cd.z_top)
    phi = Uk[1] / tab.d[1]                        # pattern per mm at small sink
    D = np.nan_to_num(cd.disp)
    m = np.isfinite(cd.disp).all(-1)
    num = np.einsum("tnc,nc->t", D * m[..., None], phi)
    den = np.einsum("tn,nc->t", m.astype(float), phi ** 2)
    return num / np.maximum(den, 1e-12)


def load_onset(cd: M.ClipData, w: np.ndarray, frac=0.15):
    """When the weight goes on and, if it does, comes off again (from the measured sink w)."""
    sel = M.window(cd)
    wm = np.nanmax(w[sel])
    idx = np.where(sel & (w > frac * wm))[0]
    if not len(idx):
        return float(cd.t[cd.k0]), None
    t_on = float(cd.t[idx[0]])
    # plateau level a second after loading; off when the sink falls below half of it afterwards
    plat = sel & (cd.t > t_on + 1.0) & (cd.t < t_on + 3.0)
    level = float(np.nanmedian(w[plat])) if plat.any() else wm
    after = np.where(sel & (cd.t > t_on + 1.0) & (w < 0.5 * level))[0]
    t_off = float(cd.t[after[0]]) - 0.1 if len(after) else None
    return t_on, t_off


def fit_E(kclips, tab, p: M.Params, log=None):
    data = []
    for cd in kclips:
        w = plate_amplitude(cd, tab)
        t_on, t_off = load_onset(cd, w)
        sel = M.window(cd, t_on - 0.3)
        data.append((cd, w, t_on, t_off, sel))

    def loss(logE):
        pe = replace(p, E=float(np.exp(logE)))
        r = []
        for cd, w, t_on, t_off, sel in data:
            _, q = M.predict_weight(cd, tab, pe, t_on, t_off)
            r.append(q[sel] - w[sel])
        return M.rms(np.concatenate(r))

    res = minimize_scalar(loss, bounds=(np.log(1e3), np.log(1e7)), method="bounded", options={"xatol": 1e-4})
    E = float(np.exp(res.x))
    per = {}
    for cd, w, t_on, t_off, sel in data:
        _, q = M.predict_weight(cd, tab, replace(p, E=E), t_on, t_off)
        end = (t_off if t_off is not None else cd.t_end)
        late = sel & (cd.t > end - 1.0) & (cd.t <= end)
        per[cd.code] = {"load_kg": cd.load_kg, "t_on_s": t_on, "t_off_s": t_off,
                        "sink_measured_mm": float(np.nanmedian(w[late])),
                        "sink_model_mm": float(np.nanmedian(q[late])), "rms_mm": M.rms(q[sel] - w[sel])}
    _log(f"  stiffness: E = {E / 1e3:.1f} kPa (weights RMS {res.fun:.3f} mm)", log)
    return E, float(res.fun), per, {cd.code: (w, t_on, t_off) for cd, w, t_on, t_off, _ in data}


# ------------------------------------------------------------------------------------------ support

def loaded_field(cd: M.ClipData, w: np.ndarray):
    """Average movement over the last 2 s the weight is on, and that stretch's time span."""
    t_on, t_off = load_onset(cd, w)
    end = t_off if t_off is not None else cd.t_end
    run = np.where(M.window(cd) & (cd.t >= max(t_on + 1.0, end - 2.0)) & (cd.t <= end))[0]
    return np.nanmean(cd.disp[run], axis=0), (float(cd.t[run[0]]), float(cd.t[run[-1]]))


def support_loss(p: M.Params, kclips, cache):
    tab = M.build_table(p, ("plate",), cache=cache)
    errs, scale = [], []
    for cd in kclips:
        w = plate_amplitude(cd, tab)
        D, _ = loaded_field(cd, w)
        phi = tab.at(cd.rest, None, cd.z_top)[1] / tab.d[1]
        m = np.isfinite(D).all(-1)
        a = float(np.sum(D[m] * phi[m]) / np.sum(phi[m] ** 2))
        errs.append((D[m] - a * phi[m]).ravel() / max(abs(a), 1e-6))     # relative to the sink
        scale.append(a)
    return M.rms(np.concatenate(errs)), scale


def fit_support(kclips, cache, p0: M.Params, log=None):
    f = lambda lk: support_loss(replace(p0, support=float(np.exp(lk))), kclips, cache)[0]
    res = minimize_scalar(f, bounds=(np.log(1e-3), np.log(1.0)), method="bounded", options={"xatol": 0.05})
    p = replace(p0, support=float(np.exp(res.x)))
    loss_rigid, _ = support_loss(replace(p0, support=1e3), kclips, cache)
    _log(f"  support: {p.support:.4f} E per mm (pattern misfit {res.fun:.3f} of the sink; "
         f"{loss_rigid:.3f} if it were rigid)", log)
    return p, float(res.fun), float(loss_rigid)
