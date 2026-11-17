"""Unit tests for src/apex on small synthetic cases (mirrors verification/ T8, T12). Run: python tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apex import aperiodic, spectra, stats, structure_function as sfn  # noqa: E402

rng = np.random.default_rng(0)


def colored(n, fs, chi, ntr):
    f = np.fft.rfftfreq(n, 1 / fs)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-chi / 2)
    Z = (rng.normal(size=(ntr, f.size)) + 1j * rng.normal(size=(ntr, f.size))) * amp
    return np.fft.irfft(Z, n=n, axis=-1)


def test_multitaper_white():
    fs, n = 500.0, 4000
    x = rng.normal(0, 2.0, (20, n))
    f, P = spectra.multitaper_psd(x, fs)
    m = (f > 5) & (f < 200)
    assert abs(P[:, m].mean() / (2 * 4.0 / fs) - 1) < 0.03, P[:, m].mean()


def test_irasa_recovers_exponent_under_oscillation():
    fs, n = 500.0, 4000
    # Without oscillations IRASA is unbiased; with a strong alpha burst it leaks into the fractal part near 10/h
    # (known limitation, Gerster et al. 2022) but stays closer than the raw spectrum. Hence the peak-free MT variant.
    x = colored(n, fs, 1.5, 40)
    f, frac, _ = spectra.irasa(x, fs)
    assert abs(aperiodic.loglog_exponent(f, frac.mean(0), (2, 40)) - 1.5) < 0.03
    t = np.arange(n) / fs
    y = x + 1.0 * x.std() * np.sin(2 * np.pi * 10 * t)[None] * (np.sin(2 * np.pi * 0.3 * t) > 0)[None]
    f, frac, mixed = spectra.irasa(y, fs)
    e_ir = aperiodic.loglog_exponent(f, frac.mean(0), (2, 40))
    e_raw = aperiodic.loglog_exponent(f, mixed.mean(0), (2, 40))
    assert abs(e_ir - 1.5) < 0.12, e_ir
    assert abs(e_raw - 1.5) > abs(e_ir - 1.5)


def field_line(n, fs, ntr, xs_mm, chi, lam0, gam, kern="exp"):
    """Time series at positions xs on a homogeneous 1-D field with coherence kappa(|dx|/lam(f)), lam = lam0 f^-gam."""
    f = np.fft.rfftfreq(n, 1 / fs)
    nc = len(xs_mm)
    D = np.abs(np.subtract.outer(xs_mm, xs_mm))
    Z = np.zeros((ntr, nc, f.size), complex)
    for i in range(1, f.size):
        lam = lam0 * f[i] ** (-gam)
        C = np.exp(-D / lam) if kern == "exp" else np.exp(-(D / lam) ** 2 / 2)
        L = np.linalg.cholesky(C + 1e-9 * np.eye(nc)) * f[i] ** (-chi / 2)
        w = rng.normal(size=(ntr, nc)) + 1j * rng.normal(size=(ntr, nc))
        Z[:, :, i] = w @ L.T
    return np.fft.irfft(Z, n=n, axis=-1)


def test_ratio_slope_detects_gamma_and_null():
    fs, n, ntr = 250.0, 2000, 60
    xs = np.array([0.0, 5.3, 10.6])
    for gam, sign in ((0.4, +1), (0.0, 0)):
        X = field_line(n, fs, ntr, xs, 1.5, 15.0, gam)
        f, P1 = spectra.multitaper_psd(X[:, 0] - X[:, 1], fs)
        _, P2 = spectra.multitaper_psd(X[:, 0] - X[:, 2], fs)
        s = sfn.ratio_slope(f, P1.mean(0), P2.mean(0), (2, 40))
        if sign > 0:
            assert s > 0.05, s
        else:
            assert abs(s) < 0.03, s


def test_rational_resample_preserves_exponent():
    x = colored(32768, 4096.0, 1.5, 4)
    y = spectra.decimate_to(x, 4096.0, 500.0)
    assert y.shape[-1] == 4000
    f, P = spectra.multitaper_psd(y, 500.0)
    assert abs(aperiodic.loglog_exponent(f, P.mean(0), (2, 40)) - 1.5) < 0.05


def test_signflip_exact():
    r = stats.signflip_test([1, 2, 3, 4, 5], "greater")
    assert abs(r["p"] - 1 / 32) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
