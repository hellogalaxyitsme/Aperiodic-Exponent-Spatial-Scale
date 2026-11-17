"""High-density scalp metrics: interior channels, Hjorth Laplacian, P-2 and pair slope.

metrics_v2(S, fs, pos) on average-referenced segments S [seg, ch, t]:
  P-2: avg − Hjorth(6), all channels and INTERIOR channels only (neighbours surround the channel)
  P-3: slope of chi_n on n(n+1) for n <= 5 (capped; 83-128 channels resolve degree ~5 reliably) and n <= 8
  pairs: slope of chi_D on ln(distance) (random pairs, IRASA)
  Per band, D(delta) = a + b (1 - exp(-delta/ell)) with intercept a >= 0 = incoherent (zero-distance) term;
          incoherent fraction at the nearest-neighbour distance, exponent of a(f), and the pair slope after removing a(f)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import pairs, scalp, spectra

BAND = (2.0, 20.0)


def interior_mask(pos, k=6, tol=0.35):
    """Channel is interior if the centroid of its k nearest neighbours lies within tol * (mean neighbour distance)
    of the channel, tangentially (i.e. neighbours surround it)."""
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    m = np.zeros(len(pos), bool)
    c0 = pos.mean(0)
    for i in range(len(pos)):
        nb = np.argsort(D[i])[1:k + 1]
        off = pos[nb].mean(0) - pos[i]
        radial = (pos[i] - c0) / np.linalg.norm(pos[i] - c0)
        tang = off - (off @ radial) * radial
        m[i] = np.linalg.norm(tang) < tol * D[i, nb].mean()
    return m


def _deg_slope(chi_n, nmax):
    n = np.arange(1, nmax + 1)
    return float(np.polyfit(n * (n + 1.0), chi_n[:nmax], 1)[0])


def _incoherent_fit(dist, Dk):
    """Fit D = a + b (1 - exp(-dist/ell)) for one band. Returns a, b, ell."""
    s = np.median(Dk)
    r = least_squares(lambda th: (th[0] + th[1] * (1 - np.exp(-dist / th[2])) - Dk) / s,
                      [0.2 * s, s, 50.0], bounds=([0, 0, 5.0], [np.inf, np.inf, 400.0]), loss="soft_l1", f_scale=0.3)
    return r.x


def _bexp(method, centers, S, band):
    """Band exponent; 'mtpf' (multitaper, peak-free) excludes 7-16 Hz (alpha and sigma) from the log-log fit."""
    if method == "mtpf":
        keep = ~((centers >= 7.0) & (centers <= 16.0))
        return pairs.band_exponent(centers[keep], S[..., keep], band)
    return pairs.band_exponent(centers, S, band)


def _psd_at(X, fs, centers, hset, nper, method):
    """PSD of X [seg, ch, t] at band centres -> [seg, ch, nb]; IRASA fractal part or plain multitaper (NW = 3)."""
    if method == "irasa":
        return pairs.irasa_channels(X, fs, centers, hset, nper)
    f, P = spectra.multitaper_psd(X, fs, NW=3.0)
    m = f > 0
    return np.exp(np.apply_along_axis(lambda y: np.interp(np.log(centers), np.log(f[m]), y), -1, np.log(np.maximum(P[..., m], 1e-300))))


def metrics_v2(S, fs, pos, n_pairs=600, seed=0, method="irasa"):
    n = S.shape[1]
    edges, centers = pairs.log_bands(BAND[0], BAND[1], 30)
    hset = np.round(np.arange(1.1, 1.95, 0.05), 3)
    nper = int(2 * fs)
    out = dict(n_ch=n, n_seg=int(S.shape[0]), method=method)
    # P-2
    Lap = scalp.hjorth_matrix(pos, k=6)
    inner = interior_mask(pos)
    chi_avg = _bexp(method, centers, _psd_at(S, fs, centers, hset, nper, method).mean(0), BAND)
    chi_lap = _bexp(method, centers, _psd_at(np.einsum("oc,sct->sot", Lap, S), fs, centers, hset, nper, method).mean(0), BAND)
    out.update(chi_avg=float(chi_avg.mean()), P2_gap_all=float((chi_avg - chi_lap).mean()),
               P2_gap_interior=float((chi_avg - chi_lap)[inner].mean()), n_interior=int(inner.sum()))
    # P-3
    Y, deg = scalp.sph_basis(pos, 8)
    Yr, dg = Y[:, deg >= 1], deg[deg >= 1]
    coef = np.linalg.solve(Yr.T @ Yr + 1e-3 * np.trace(Yr.T @ Yr) / Yr.shape[1] * np.eye(Yr.shape[1]), Yr.T)
    Pc = _psd_at(np.einsum("kc,sct->skt", coef, S), fs, centers, hset, nper, method).mean(0)
    chi_n = _bexp(method, centers, np.stack([Pc[dg == k].sum(0) for k in range(1, 9)]), BAND)
    out.update(P3_slope_n5=_deg_slope(chi_n, 5), P3_slope_n8=_deg_slope(chi_n, 8), chi_n=" ".join(f"{v:.3f}" for v in chi_n))
    # pairs
    rng = np.random.default_rng(seed)
    ia, ib = np.triu_indices(n, 1)
    pick = rng.choice(len(ia), size=min(n_pairs, len(ia)), replace=False)
    acc = []
    for s0 in range(0, len(pick), 100):
        pk = pick[s0:s0 + 100]
        if method == "irasa":   # average the linear fractal PSD, then interpolate
            f, frac, _ = spectra.irasa(S[:, ia[pk]] - S[:, ib[pk]], fs, hset=hset, nperseg=nper)
            acc.append(frac.mean(0))
        else:
            acc.append(_psd_at(S[:, ia[pk]] - S[:, ib[pk]], fs, centers, hset, nper, method).mean(0))
    if method == "irasa":
        fm = f > 0
        Dc = np.exp(np.apply_along_axis(lambda y: np.interp(np.log(centers), np.log(f[fm]), y), -1, np.log(np.concatenate(acc, 0)[..., fm])))
    else:
        Dc = np.concatenate(acc, 0)
    dist = np.linalg.norm(pos[ia[pick]] - pos[ib[pick]], axis=1)
    x = np.log(dist) - np.log(dist).mean()
    chi_D = _bexp(method, centers, Dc, BAND)
    out["pair_slope"] = float((chi_D - chi_D.mean()) @ x / (x @ x))
    fits = np.array([_incoherent_fit(dist, Dc[:, k]) for k in range(len(centers))])     # [nb, 3]
    a = fits[:, 0]
    nn = np.median(np.sort(np.linalg.norm(pos[:, None] - pos[None], axis=-1), axis=1)[:, 1])
    Dnn = a + fits[:, 1] * (1 - np.exp(-nn / fits[:, 2]))
    out["incoh_frac_at_nn"] = float(np.median(a / Dnn))
    out["nn_dist_mm"] = float(nn)
    ok = a > 0
    out["incoh_exponent"] = float(-np.polyfit(np.log(centers[ok]), np.log(a[ok]), 1)[0]) if ok.sum() >= 10 else np.nan
    Dcorr = Dc - a[None]
    good = np.all(Dcorr > 0, axis=1)
    chi_c = _bexp(method, centers, Dcorr[good], BAND)
    xc = np.log(dist[good]) - np.log(dist[good]).mean()
    out["pair_slope_incoh_removed"] = float((chi_c - chi_c.mean()) @ xc / (xc @ xc))
    out["frac_pairs_valid_after_removal"] = float(good.mean())
    return out
