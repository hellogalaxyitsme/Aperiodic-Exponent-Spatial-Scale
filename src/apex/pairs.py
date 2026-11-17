"""All-pairs reference-free structure functions and band exponents on log-spaced frequency bands.

D_ij(f) = E|x_i - x_j|^2 is independent of any common reference. For n channels we compute it for all n(n-1)/2 pairs
from multitaper cross-spectra (exact identity D_ij = P_i + P_j - 2 Re C_ij) and from IRASA on the time-domain differences.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import dpss

from . import spectra


def log_bands(lo=2.0, hi=40.0, n=40):
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), n + 1))
    return edges, np.sqrt(edges[:-1] * edges[1:])


def pair_index(n):
    iu = np.triu_indices(n, 1)
    return iu[0], iu[1]


def mt_allpairs(X, fs, edges, NW=4.0):
    """X: [trials, ch, n]. Returns P [trials, ch, nb] (band-mean PSD) and D [trials, npairs, nb]."""
    ntr, nc, n = X.shape
    K = int(2 * NW - 1)
    tap = dpss(n, NW, K)
    f = np.fft.rfftfreq(n, 1 / fs)
    nb = len(edges) - 1
    # bins per band; a band narrower than one bin borrows the nearest bin
    binsets = []
    for b in range(nb):
        idx = np.where((f >= edges[b]) & (f < edges[b + 1]))[0]
        if idx.size == 0:
            idx = np.array([np.argmin(np.abs(f - np.sqrt(edges[b] * edges[b + 1])))])
        binsets.append(idx)
    hi = max(s.max() for s in binsets) + 1
    a, b_ = pair_index(nc)
    P = np.empty((ntr, nc, nb))
    D = np.empty((ntr, len(a), nb))
    for t in range(ntr):
        Xk = np.fft.rfft(X[t][:, None, :] * tap[None], axis=-1)[..., :hi]     # [ch, K, nf]
        for bi, idx in enumerate(binsets):
            Z = Xk[:, :, idx].reshape(nc, -1)                                    # [ch, K*bins]
            C = (Z @ Z.conj().T).real / Z.shape[1] * 2 / fs                      # one-sided PSD units
            d = np.diag(C)
            P[t, :, bi] = d
            D[t, :, bi] = d[a] + d[b_] - 2 * C[a, b_]
    return P, D


def irasa_allpairs(X, fs, centers, hset=None, nperseg=None, chunk=64):
    """IRASA fractal PSD of every pairwise difference, evaluated at band centres (log-log interpolation).
    X: [trials, ch, n] -> [trials, npairs, nb]."""
    ntr, nc, n = X.shape
    a, b = pair_index(nc)
    out = np.empty((ntr, len(a), len(centers)))
    for s in range(0, len(a), chunk):
        sl = slice(s, s + chunk)
        d = X[:, a[sl], :] - X[:, b[sl], :]
        f, frac, _ = spectra.irasa(d, fs, hset=hset, nperseg=nperseg)
        m = f > 0
        lf, lp = np.log(f[m]), np.log(np.maximum(frac[..., m], 1e-300))
        out[:, sl, :] = np.exp(np.apply_along_axis(lambda y: np.interp(np.log(centers), lf, y), -1, lp))
    return out


def irasa_channels(X, fs, centers, hset=None, nperseg=None):
    f, frac, _ = spectra.irasa(X, fs, hset=hset, nperseg=nperseg)
    m = f > 0
    lp = np.log(np.maximum(frac[..., m], 1e-300))
    return np.exp(np.apply_along_axis(lambda y: np.interp(np.log(centers), np.log(f[m]), y), -1, lp))


def band_exponent(centers, S, band=(2.0, 40.0)):
    """Vectorised log-log exponent over the last axis. S: [..., nb] -> [...]."""
    m = (centers >= band[0]) & (centers <= band[1])
    x = np.log(centers[m])
    x = x - x.mean()
    y = np.log(S[..., m])
    return -(y - y.mean(-1, keepdims=True)) @ x / (x @ x)
