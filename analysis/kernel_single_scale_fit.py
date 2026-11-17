"""Single-scale coherence kernel from the ds004752 subdural grid (Results 3.5, S3).

Tangential pair slope; per band, the normalised structure function fitted by A (1 - exp(-(d/lam)^p)); frequency-scaling
exponent of the coherence length; frozen parameters (results/grid7a/frozen_params.json) used by kernel_predict_scalp.py.
Usage: python analysis/kernel_single_scale_fit.py --out results/grid7a
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import io, pairs, spectra, stats  # noqa: E402

HSET = np.round(np.arange(1.1, 1.95, 0.05), 4)


def fit_band(dist, y, p=None):
    def res(th):
        A, ll, pp = th[0], th[1], (th[2] if p is None else p)
        return A * (1 - np.exp(-(dist / np.exp(ll)) ** pp)) - y
    x0 = [1.0, np.log(15.0)] + ([1.0] if p is None else [])
    lo = [0.2, np.log(1.0)] + ([0.3] if p is None else [])
    hi = [3.0, np.log(300.0)] + ([2.5] if p is None else [])
    r = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grid7a")
    ap.add_argument("--far_mm", type=float, default=50.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    edges, centers = pairs.log_bands(2.0, 80.0, 48)
    Dall, xyz, names = [], None, None
    for s in io.sessions("sub-10"):
        el = s.electrodes()
        g = el[el["kind"] == "grid"].sort_values(["row", "idx"])
        names = g["name"].tolist()
        xyz = g[["x", "y", "z"]].to_numpy(float)
        X, ch, fs, ev = s.load("ieeg", drop_artifacts=True, picks=names)
        X = X[:, [ch.index(n) for n in names]]
        Xd = spectra.decimate_to(X, fs, 500.0)
        Dall.append(pairs.irasa_allpairs(Xd, 500.0, centers, hset=HSET, nperseg=1000))
    D = np.concatenate(Dall, 0).astype(float)                         # [trials, pairs, nb]
    T = D.shape[0]
    ia, ib = pairs.pair_index(len(names))
    dist = np.linalg.norm(xyz[ia] - xyz[ib], axis=1)
    nn = float(np.median(np.sort(np.linalg.norm(xyz[:, None] - xyz[None], axis=-1), axis=1)[:, 1]))
    Dm = D.mean(0)
    res = dict(n_contacts=len(names), n_pairs=len(ia), n_trials=T, nn_spacing_mm=nn, dist_range_mm=[float(dist.min()), float(dist.max())])
    x = np.log(dist) - np.log(dist).mean()
    rng = np.random.default_rng(1)
    W = rng.multinomial(T, np.ones(T) / T, size=300) / T
    for band, tag in (((2.0, 40.0), "2_40"), ((2.0, 20.0), "2_20"), ((20.0, 40.0), "20_40"), ((40.0, 80.0), "40_80")):
        chi = pairs.band_exponent(centers, Dm, band)
        res[f"tangential_slope_{tag}"] = float((chi - chi.mean()) @ x / (x @ x))
        cb = pairs.band_exponent(centers, np.einsum("bt,tpk->bpk", W, D), band)
        bs = (cb - cb.mean(1, keepdims=True)) @ x / (x @ x)
        res[f"tangential_slope_{tag}_ci"] = list(stats.bootstrap_ci(bs))
    # binned exponent curve (for figure)
    bins = np.array([5, 10, 15, 20, 30, 40, 55, 75, 100])
    bi = np.digitize(dist, bins) - 1
    chi240 = pairs.band_exponent(centers, Dm, (2.0, 40.0))
    res["binned_chi_2_40"] = [[float(np.exp(np.log(dist[bi == k]).mean())), float(np.median(chi240[bi == k])), int((bi == k).sum())]
                              for k in range(len(bins) - 1) if (bi == k).sum() >= 5]
    # normalised structure function per band
    far = dist >= args.far_mm
    res["n_far_pairs"] = int(far.sum())
    lam = {"free": [], "exp": [], "gauss": []}
    pfree, rms = [], {"free": [], "exp": [], "gauss": []}
    Y = Dm / np.median(Dm[far], axis=0, keepdims=True)
    for k in range(len(centers)):
        for kind, p in (("free", None), ("exp", 1.0), ("gauss", 2.0)):
            th, e = fit_band(dist, Y[:, k], p)
            lam[kind].append(float(np.exp(th[1])))
            rms[kind].append(e)
            if kind == "free":
                pfree.append(float(th[2]))
    lam = {k: np.array(v) for k, v in lam.items()}
    res["kernel_shape_p_median_2_40"] = float(np.median(np.array(pfree)[centers <= 40]))
    res["rms_by_kernel_2_40"] = {k: float(np.mean(np.array(v)[centers <= 40])) for k, v in rms.items()}
    kern = "exp" if res["rms_by_kernel_2_40"]["exp"] <= res["rms_by_kernel_2_40"]["gauss"] else "gauss"
    res["kernel_selected"] = kern
    m = (centers >= 2) & (centers <= 40)
    lf = np.log(centers[m])
    L = lam[kern][m]
    res["lam_eff_mm_at"] = {f"{f:.0f}Hz": float(np.exp(np.interp(np.log(f), np.log(centers), np.log(lam[kern])))) for f in (2, 5, 10, 20, 40, 80)}
    res["gamma_naive"] = float(-np.polyfit(lf, np.log(L), 1)[0])
    # blur-corrected (gauge-fixed): lam_eff^2 = lam0^2 f^-2g + c, c in [0, min lam^2)
    best = None
    for c in np.linspace(0, 0.98 * L.min() ** 2, 2000):
        yy = np.log(L ** 2 - c)
        p = np.polyfit(lf, yy, 1)
        r = np.sum((yy - np.polyval(p, lf)) ** 2)
        if best is None or r < best[0]:
            best = (r, c, p)
    _, c, p = best
    res["gamma_blur_corrected"] = float(-p[0] / 2)
    res["blur_c_mm2"] = float(c)
    res["lam10_source_mm"] = float(np.sqrt(np.exp(np.polyval(p, np.log(10.0)))))
    # collapse test (A4): y vs d/lam(f) should collapse across frequencies if kappa(d / lam(f)) holds
    def dispersion(xs):
        grid = np.exp(np.linspace(np.log(np.percentile(xs, 10)), np.log(np.percentile(xs, 90)), 12))
        ys = []
        for k in np.where(m)[0]:
            o = np.argsort(xs[:, k])
            ys.append(np.interp(np.log(grid), np.log(xs[o, k]), Y[o, k]))
        ys = np.array(ys)
        return float(np.mean(np.std(ys, axis=0)))
    Xd_ = np.repeat(dist[:, None], len(centers), 1)
    Xr = Xd_ / lam[kern][None]
    res["collapse_dispersion_raw_distance"] = dispersion(Xd_)
    res["collapse_dispersion_rescaled"] = dispersion(Xr)
    res["collapse_ratio"] = res["collapse_dispersion_rescaled"] / res["collapse_dispersion_raw_distance"]
    np.savez_compressed(out / "grid_curves.npz", centers=centers, lam_free=lam["free"], lam_exp=lam["exp"], lam_gauss=lam["gauss"],
                        p_free=np.array(pfree), dist=dist, chi_2_40=chi240, Y=Y)
    frozen = dict(source="grid_7a sub-10", gamma=res["gamma_blur_corrected"], gamma_naive=res["gamma_naive"],
                  lam10_source_mm=res["lam10_source_mm"], kernel=kern)
    blob = json.dumps(frozen, sort_keys=True).encode()
    frozen["sha256_12"] = hashlib.sha256(blob).hexdigest()[:12]
    json.dump(frozen, open(out / "frozen_params.json", "w"), indent=2)
    json.dump(res, open(out / "grid7a.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))
    print("FROZEN:", json.dumps(frozen))


if __name__ == "__main__":
    main()
