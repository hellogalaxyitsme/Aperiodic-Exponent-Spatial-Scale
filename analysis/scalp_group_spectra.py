"""Group-mean average-reference and Laplacian spectra, interior channels (Figure S3).

ds003478 eyes open and eyes closed, ANPHY wake and N2; all participants, no selection. IRASA fractal spectra and Welch
spectra, each centred on its mean log power over 2-20 Hz. Needs the scalp unit cache written by scalp_derivations.py.
Usage: python analysis/scalp_group_spectra.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import pairs  # noqa: E402
from apex.paths import CACHE  # noqa: E402

SCALP = Path(f"{CACHE}/final_round/scalp")
_, CEN = pairs.log_bands(2.0, 20.0, 30)
UNITS = {"ds003478_EO": "ds003478_*_EO.npz", "ds003478_EC": "ds003478_*_EC.npz",
         "anphy_W": "anphy_*_W.npz", "anphy_N2": "anphy_*_N2.npz"}


def centred(logp, m):
    return logp - logp[..., m].mean(-1, keepdims=True)


def main():
    out = {"cen": CEN}
    for key, pat in UNITS.items():
        I, M, n = {"a": [], "l": []}, {"a": [], "l": []}, 0
        for p in sorted(SCALP.glob(pat)):
            if ".tmp" in p.name:
                continue
            z = np.load(p, allow_pickle=True)
            inner = z["interior"]
            f = z["f"]
            fm = (f >= 2.0) & (f <= 30.0)
            for d, ki, km in (("a", "Ia", "Ma"), ("l", "Il", "Ml")):
                I[d].append(centred(np.log10(z[ki][inner].astype(float)).mean(0), slice(None)))
                lw = np.log10(z[km][inner][:, fm].astype(float)).mean(0)
                M[d].append(lw - lw[f[fm] <= 20.0].mean())
            out["f_welch"] = f[fm]
            n += 1
        for d in ("a", "l"):
            out[f"{key}_irasa_{d}"] = np.array(I[d])
            out[f"{key}_welch_{d}"] = np.array(M[d])
        out[f"{key}_n"] = n
        print(key, n)
    (ROOT / "results" / "revision").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "results" / "revision" / "spectra_example.npz", **out)


if __name__ == "__main__":
    main()
