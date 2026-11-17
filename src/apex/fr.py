"""Helpers: mixed (Welch) spectra, specparam (fixed/knee) with peak removal, local exponents."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import welch

SP_SETTINGS = dict(peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05)
R2_MIN = 0.8


def welch_mixed(X, fs, fmax=100.0):
    """X [..., seg, k, t] or [seg, k, t] -> f, P [k, nf] averaged over segments (axis 0). 2-s Hann, 50 % overlap."""
    nper = int(2 * fs)
    f, P = welch(X, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann", detrend="constant", axis=-1)
    keep = (f > 0) & (f <= fmax)
    return f[keep], P[..., keep].mean(0)


def interp_line(f, P, line_hz, width=2.0):
    """Log-log interpolate across line-noise harmonics (P [..., nf])."""
    P = np.array(P, float, copy=True)
    bad = np.zeros_like(f, bool)
    for h in np.arange(line_hz, f.max() + 1, line_hz):
        bad |= np.abs(f - h) <= width
    if bad.any():
        lf = np.log(f)
        lp = np.log(np.maximum(P, 1e-300))
        good = ~bad
        lp[..., bad] = np.apply_along_axis(lambda y: np.interp(lf[bad], lf[good], y[good]), -1, lp)
        P = np.exp(lp)
    return P


def sp_fit(f, P, mode="fixed", frange=(2.0, 40.0)):
    """specparam fit; returns dict(exponent, knee, offset, r2, peaks [[cf, pw, bw], ...]) or NaNs."""
    from specparam import SpectralModel
    nan = dict(exponent=np.nan, knee=np.nan, offset=np.nan, r2=np.nan, peaks=np.zeros((0, 3)))
    fm = SpectralModel(aperiodic_mode=mode, verbose=False, **SP_SETTINGS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fm.fit(np.asarray(f, float), np.asarray(P, float), list(frange))
        except Exception:
            return nan
        try:
            ap = np.atleast_1d(fm.get_params("aperiodic"))
            pk = np.atleast_2d(fm.get_params("peak"))
        except Exception:
            try:
                ap = np.atleast_1d(fm.results.params.aperiodic.params)
                pk = np.atleast_2d(fm.results.params.peak.params)
            except Exception:
                return nan
        try:
            r2 = float(fm.get_metrics("gof_rsquared"))
        except Exception:
            try:
                r2 = float(fm.get_metrics("gof", "rsquared"))
            except Exception:
                r2 = np.nan
    if pk.size == 0 or np.all(np.isnan(pk)):
        pk = np.zeros((0, 3))
    exp_ = float(ap[-1])
    knee = float(ap[1]) if mode == "knee" and ap.size == 3 else np.nan
    return dict(exponent=exp_, knee=knee, offset=float(ap[0]), r2=r2, peaks=pk)


def sp_exponents(f, P, mode, frange):
    """Vectorised over rows of P [n, nf]: exponent with the R2 filter applied (NaN when R2 < R2_MIN)."""
    out = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        r = sp_fit(f, P[i], mode, frange)
        if np.isfinite(r["r2"]) and r["r2"] >= R2_MIN:
            out[i] = r["exponent"]
    return out


def peak_removed_log10(f, P, frange):
    """log10 P with fitted specparam peaks (fixed mode over frange) subtracted as Gaussians. Returns (log10 P_rm, n_peaks)."""
    r = sp_fit(f, P, "fixed", frange)
    lp = np.log10(np.maximum(P, 1e-300))
    for cf, pw, bw in r["peaks"]:
        if np.isfinite(cf):
            sd = max(bw / 2.0, 1e-3)
            lp = lp - pw * np.exp(-(f - cf) ** 2 / (2 * sd ** 2))
    return lp, len(r["peaks"])


def local_exponents(f, log10P, edges):
    """Local exponent per sub-band: -slope of log10 P on log10 f within [edges[k], edges[k+1]]. log10P [..., nf] -> [..., nb]."""
    out = []
    lf = np.log10(f)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f <= hi)
        x = lf[m] - lf[m].mean()
        y = log10P[..., m]
        out.append(-((y - y.mean(-1, keepdims=True)) @ x) / (x @ x))
    return np.stack(out, -1)


def slope(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    return float((y - y.mean()) @ x / (x @ x)) if (x @ x) > 0 else np.nan
