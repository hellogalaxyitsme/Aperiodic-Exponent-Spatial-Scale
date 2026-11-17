"""Spectral estimators: multitaper PSD, IRASA fractal component, noise floor."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly, welch, decimate
from scipy.signal.windows import dpss


def multitaper_psd(x: np.ndarray, fs: float, NW: float = 4.0, K: int | None = None, fmax: float | None = None):
    """One-sided multitaper PSD along the last axis. x: [..., n]. Returns (f, P[..., nf]) in units^2/Hz."""
    n = x.shape[-1]
    K = int(2 * NW - 1) if K is None else K
    tapers = dpss(n, NW, K)                                   # [K, n], unit energy
    f = np.fft.rfftfreq(n, 1 / fs)
    keep = slice(None) if fmax is None else slice(0, int(np.searchsorted(f, fmax, side="right")))
    f = f[keep]
    lead = x.shape[:-1]
    xf = x.reshape(-1, n)
    out = np.empty((xf.shape[0], f.size))
    for i in range(xf.shape[0]):                              # loop keeps memory bounded
        X = np.fft.rfft(xf[i][None, :] * tapers, axis=-1)[:, keep]
        out[i] = (np.abs(X) ** 2).mean(0) / fs
    out[:, 1:] *= 2
    if fmax is None and n % 2 == 0:
        out[:, -1] /= 2
    return f, out.reshape(*lead, f.size)


def irasa(x: np.ndarray, fs: float, hset=None, nperseg: int | None = None):
    """IRASA (Wen & Liu 2016). x: [..., n]. Returns (f, fractal[..., nf], mixed[..., nf]).

    For each h: resample by h and 1/h (rational approximation), Welch PSD at the ORIGINAL fs, geometric mean.
    Fractal = median over h. Valid up to fs / (2 * max(h)) and down to roughly max(h) * f_res.
    """
    if hset is None:
        hset = np.round(np.arange(1.1, 1.95, 0.05), 3)
    nperseg = int(2 * fs) if nperseg is None else nperseg
    kw = dict(fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann", detrend="constant", axis=-1)
    f, mixed = welch(x, **kw)
    G = []
    for h in hset:
        fr = Fraction(float(h)).limit_denominator(20)
        up = resample_poly(x, fr.numerator, fr.denominator, axis=-1)
        dn = resample_poly(x, fr.denominator, fr.numerator, axis=-1)
        _, pu = welch(up, **kw)
        _, pd = welch(dn, **kw)
        G.append(np.sqrt(pu * pd))
    fractal = np.median(np.stack(G, 0), axis=0)
    return f, fractal, mixed


def decimate_to(x: np.ndarray, fs: float, fs_new: float) -> np.ndarray:
    q = int(round(fs / fs_new))
    if abs(fs / q - fs_new) < 1e-6:
        return decimate(x, q, ftype="fir", zero_phase=True, axis=-1)
    fr = Fraction(fs_new / fs).limit_denominator(4096)          # e.g. 4096 -> 500 Hz: 125/1024
    assert abs(fs * fr.numerator / fr.denominator - fs_new) < 1e-6
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1)


def noise_floor(f: np.ndarray, P: np.ndarray, band=(600.0, 900.0)) -> np.ndarray:
    """Median PSD in a high band (robust to line harmonics). P: [..., nf] -> [...]. An upper bound on a white floor."""
    m = (f >= band[0]) & (f <= band[1])
    return np.median(P[..., m], axis=-1)
