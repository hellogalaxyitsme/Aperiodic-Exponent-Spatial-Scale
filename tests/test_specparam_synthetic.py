"""Method check on SYNTHETIC spectra only: specparam wrapper recovers known exponents; peak removal works.
Usage: python3 analysis/fr_selftest.py"""
import sys

import numpy as np

sys.path.insert(0, "src")
from apex import fr  # noqa: E402

f = np.arange(0.5, 100.01, 0.5)
rng = np.random.default_rng(1)
for chi in (1.0, 1.5, 2.2):
    lp = 1.0 - chi * np.log10(f) + 0.6 * np.exp(-(f - 10) ** 2 / (2 * 1.5 ** 2)) + 0.3 * np.exp(-(f - 21) ** 2 / (2 * 2.0 ** 2))
    P = 10 ** (lp + 0.02 * rng.standard_normal(f.size))
    r = fr.sp_fit(f, P, "fixed", (2.0, 40.0))
    k = fr.sp_fit(f, 10 ** (np.log10(P) - np.log10(1 + (3.0 / f) ** 0)), "knee", (1.0, 40.0))
    m = (f >= 2) & (f <= 80)
    rm, npk = fr.peak_removed_log10(f[m], P[m], (2.0, 80.0))
    le_raw = fr.local_exponents(f[m], np.log10(P[m]), np.exp(np.linspace(np.log(2), np.log(80), 9)))
    le_rm = fr.local_exponents(f[m], rm, np.exp(np.linspace(np.log(2), np.log(80), 9)))
    print(f"true {chi}: fixed {r['exponent']:.3f} r2 {r['r2']:.3f} peaks {len(r['peaks'])} | knee {k['exponent']:.3f} | "
          f"local raw {np.round(le_raw, 2)} | peak-removed {np.round(le_rm, 2)} ({npk} peaks)")
P2 = fr.interp_line(f, 10 ** (1 - 1.5 * np.log10(f)) * (1 + 50 * (np.abs(f - 50) < 1)), 50.0)
print("line interp exponent:", round(fr.sp_fit(f, P2, "fixed", (30.0, 70.0))["exponent"], 3))
