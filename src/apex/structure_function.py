"""Depth-electrode structure functions and the reference-free spacing test."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j0

_U = np.exp(np.linspace(np.log(1e-4), np.log(1e4), 1500))
_LU = np.log(_U)


def ratio_slope(f, D_short, D_long, band=(2.0, 40.0), mask=None) -> float:
    """Spacing-test statistic: slope of log(D(delta_short)/D(delta_long)) on log f. 0 iff gamma = 0; > 0 if gamma > 0."""
    m = (f >= band[0]) & (f <= band[1]) & (D_short > 0) & (D_long > 0)
    if mask is not None:
        m &= mask
    return float(np.polyfit(np.log(f[m]), np.log(D_short[m] / D_long[m]), 1)[0])


def matern_khat(u, nu):
    """d = 2 Matern spatial spectrum with kappa(0) = 1."""
    return 4 * np.pi * nu * (1 + u ** 2) ** (-(nu + 1.0))


def sf_matern_blur(delta, lam, sigma, nu):
    """D / (2 P0) for Matern(nu) coherence (length lam) seen through a Gaussian blur exp(-sigma^2 k^2). d = 2."""
    lam = np.asarray(lam, float)[..., None]
    delta = np.asarray(delta, float)[..., None]
    integ = matern_khat(_U, nu) * np.exp(-(sigma * _U / lam) ** 2) * (1 - j0(_U * delta / lam)) * _U * _U / (2 * np.pi)
    return np.trapezoid(integ, _LU, axis=-1)


def joint_fit(f, D, deltas, band=(2.0, 40.0), n_freq=60, starts=None):
    """Proposition 5' estimator. D: [n_sep, nf] aperiodic structure functions; deltas: [n_sep] in mm.

    Model: log D_j(f) = log 2 + a - chi_s log f + log sf_matern_blur(delta_j, lam0 f^-gam, sigma, nu).
    Fit on n_freq log-spaced frequencies (interpolated) in band. Returns dict with chi_s, gam, nu, sigma, lam0, cost.
    """
    m = (f >= band[0]) & (f <= band[1])
    fq = np.exp(np.linspace(np.log(band[0]), np.log(band[1]), n_freq))
    Y = np.stack([np.interp(np.log(fq), np.log(f[m]), np.log(Dj[m])) for Dj in D])
    lfq = np.log(fq)
    deltas = np.asarray(deltas, float)

    def model(th):
        a, chi, ll0, g, lnu, lsig = th
        lam = np.exp(ll0) * fq ** (-g)
        return np.log(2) + a - chi * lfq[None] + np.log(sf_matern_blur(deltas[:, None], lam[None], np.exp(lsig), np.exp(lnu)) + 1e-300)

    def res(th):
        return (Y - model(th)).ravel()

    a0 = float(np.mean(Y))
    lo = [a0 - 30, -1.0, np.log(0.5), -0.5, np.log(0.2), np.log(0.1)]
    hi = [a0 + 30, 5.0, np.log(500.0), 1.5, np.log(30.0), np.log(30.0)]
    if starts is None:
        starts = [(ll0, g0, lnu, ls) for ll0 in (np.log(8.0), np.log(30.0)) for g0 in (0.0, 0.4)
                  for lnu in (np.log(0.5), np.log(2.0)) for ls in (np.log(1.0), np.log(5.0))]
    best = None
    for ll0, g0, lnu, ls in starts:
        x0 = [a0, 1.5, ll0, g0, lnu, ls]
        try:
            r = least_squares(res, x0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    a, chi, ll0, g, lnu, lsig = best.x
    rms = float(np.sqrt(np.mean(best.fun ** 2)))
    at_bound = [n for n, v, l, h in zip(["a", "chi", "lam0", "gam", "nu", "sigma"], best.x, lo, hi) if abs(v - l) < 1e-6 or abs(v - h) < 1e-6]
    return dict(chi_s=float(chi), gam=float(g), nu=float(np.exp(lnu)), sigma=float(np.exp(lsig)), lam0=float(np.exp(ll0)),
                rms=rms, at_bound="|".join(at_bound))
