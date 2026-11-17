"""Aperiodic exponent estimators: specparam fixed mode and log-log fit of the IRASA fractal part."""
from __future__ import annotations

import warnings

import numpy as np


def loglog_exponent(f, P, band=(2.0, 40.0), mask=None) -> float:
    """Exponent chi = -slope of log P on log f over band (optionally restricted by a boolean mask)."""
    m = (f >= band[0]) & (f <= band[1]) & np.isfinite(P) & (P > 0)
    if mask is not None:
        m &= mask
    return float(-np.polyfit(np.log(f[m]), np.log(P[m]), 1)[0])


def specparam_fit(f, P, band=(2.0, 40.0), peak_width_limits=(1.0, 8.0), max_n_peaks=6):
    """Fixed-mode specparam. Returns dict(exponent, offset, r2, peaks[[cf, pw, bw], ...]) or NaNs on failure."""
    try:
        from specparam import SpectralModel
    except ImportError:   # local machine without specparam
        return dict(exponent=np.nan, offset=np.nan, r2=np.nan, peaks=np.zeros((0, 3)))
    fm = SpectralModel(aperiodic_mode="fixed", verbose=False, peak_width_limits=list(peak_width_limits),
                       max_n_peaks=max_n_peaks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fm.fit(np.asarray(f, float), np.asarray(P, float), list(band))
            off, exp_ = fm.get_params("aperiodic")
            pk = np.atleast_2d(fm.get_params("periodic"))
            if pk.size == 0 or np.all(np.isnan(pk)):
                pk = np.zeros((0, 3))
            try:
                r2 = float(fm.get_metrics("gof", "rsquared"))
            except Exception:
                r2 = np.nan
            return dict(exponent=float(exp_), offset=float(off), r2=r2, peaks=pk)
        except Exception:
            return dict(exponent=np.nan, offset=np.nan, r2=np.nan, peaks=np.zeros((0, 3)))


def peak_free_mask(f, peaks_list, n_bw: float = 1.0) -> np.ndarray:
    """Boolean mask of bins farther than n_bw * (full bandwidth) from any specparam peak centre in peaks_list."""
    m = np.ones_like(f, dtype=bool)
    for pk in peaks_list:
        for cf, _, bw in np.atleast_2d(pk):
            if np.isfinite(cf):
                m &= np.abs(f - cf) > n_bw * bw
    return m
