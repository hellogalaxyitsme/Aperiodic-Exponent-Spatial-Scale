"""Depth analyses with spectral parameterisation and peak removal (Results 3.1; Figure 1c,d).

C1 with specparam (fixed and knee), C2 without frontal channels, adjacent minus far spectra, peak-removed band-resolved
C1 (eight sub-bands of 2-80 Hz), pooled non-adjacent analysis and adjacent-versus-non-adjacent comparison.
Usage: python analysis/depth_estimator_peak_tests.py --jobs 16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import fr, io, stats  # noqa: E402
from apex.paths import CACHE  # noqa: E402

SEEG = Path(f"{CACHE}/final_round/seeg")
ROBUST = Path(f"{CACHE}/robust")
RES = Path("results/final_round")
DEV = {f"sub-{i:02d}" for i in range(1, 6)}
EDGES8 = np.exp(np.linspace(np.log(2.0), np.log(80.0), 9))
FRONT = {"F3", "F4", "F7", "F8", "Fz"}


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    b = np.random.default_rng(seed).choice(x, (n, x.size)).mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def test(v, alt="greater"):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return dict(stats.signflip_test(v, alt), ci95=boot_ci(v)) if len(v) >= 2 else dict(n=len(v))


def seeg_unit(path):
    z = np.load(path, allow_pickle=True)
    cohort, sub = Path(path).stem.split("_", 1)
    f, P = z["f"], fr.interp_line(z["f"], z["P"][z["prim"]].astype(float), float(z["line"]))
    d, nb = z["dist"][z["prim"]], z["n_between"][z["prim"]]
    ld = np.log(d)
    r = dict(cohort=cohort, sub=sub, split="dev" if sub in DEV else ("confirm" if cohort == "ds004752" else "hup"), n_pairs=len(d))
    for mode, rng in (("fixed", (2.0, 40.0)), ("knee", (1.0, 40.0))):
        chi = fr.sp_exponents(f, P, mode, rng)
        r[f"C1_sp{mode}"] = fr.slope(chi, ld)
        r[f"sp{mode}_valid"] = float(np.isfinite(chi).mean())
    # band-resolved C1, raw and peak-removed (fixed fit over 2-80 Hz)
    m = (f >= 2.0) & (f <= 80.0)
    raw = np.log10(P[:, m])
    pr = [fr.peak_removed_log10(f[m], P[i, m], (2.0, 80.0)) for i in range(len(P))]
    rm = np.stack([p[0] for p in pr])
    r["median_peaks"] = float(np.median([p[1] for p in pr]))
    for tag, L in (("raw", raw), ("rm", rm)):
        le = fr.local_exponents(f[m], L, EDGES8)
        for k in range(8):
            r[f"band{k}_{tag}"] = fr.slope(le[:, k], ld)
    # near vs far normalised spectra (2-80 Hz)
    norm = raw - raw[:, (f[m] <= 40)].mean(1, keepdims=True)
    near, far = nb == 0, d >= 20
    spec = dict(near=norm[near].mean(0) if near.sum() else None, far=norm[far].mean(0) if far.sum() else None)
    return r, spec, f[m]


def c2_unit(sub):
    """C2 pair slope (IRASA robust cache, trial mean) with/without frontal channels; specparam on mixed scalp spectra."""
    Z = [np.load(p, allow_pickle=True) for p in sorted(ROBUST.glob(f"{sub}_ses-*.npz"))]
    names = [str(x) for x in Z[0]["scalp_names"]]
    pos = Z[0]["scalp_pos"]
    cs = Z[0]["centers_scalp"]
    De = np.concatenate([z["De_orig"] for z in Z], 0).mean(0)
    a, b = np.triu_indices(len(names), 1)
    ld = np.log(np.linalg.norm(pos[a] - pos[b], axis=1))
    nof = np.array([names[i] not in FRONT and names[j] not in FRONT for i, j in zip(a, b)])
    n_nonfront = sum(n not in FRONT for n in names)
    from apex import pairs
    r = dict(sub=sub, split="dev" if sub in DEV else "confirm", n_scalp=len(names), n_nonfront=n_nonfront)
    for band, t in (((2.0, 20.0), "2_20"), ((5.0, 20.0), "5_20")):
        chi = pairs.band_exponent(cs, De, band)
        r[f"C2_all_{t}"] = fr.slope(chi, ld)
        r[f"C2_nofront_{t}"] = fr.slope(chi[nof], ld[nof]) if n_nonfront >= 4 else np.nan
    zf = SEEG / f"ds004752_{sub}.npz"
    if zf.exists():
        z = np.load(zf, allow_pickle=True)
        Pe = fr.interp_line(z["fe"], z["Pe"].astype(float), 50.0)
        chi = fr.sp_exponents(z["fe"], Pe, "fixed", (2.0, 20.0))
        r["C2_spfixed_all"] = fr.slope(chi, ld)
        r["C2_spfixed_nofront"] = fr.slope(chi[nof], ld[nof]) if n_nonfront >= 4 else np.nan
    return r


def pooled_nonadjacent():
    P = pd.read_csv("results/straddle/straddle_pairs.csv")
    P["coh"] = (P.cohort == "HUP").astype(int)
    out = {}
    na = P[P.n_between >= 1]
    for name, S in (("pooled", na), ("ds004752", na[na.cohort == "ds004752"]), ("HUP", na[na.cohort == "HUP"])):
        sl = [fr.slope(g.chi.to_numpy(), g.ld.to_numpy()) for _, g in S.groupby("sub") if len(g) >= 3]
        out[f"nonadjacent_slope_{name}"] = test(sl)
    try:
        import statsmodels.formula.api as smf
        fit = smf.mixedlm("chi ~ ld + coh", na, groups=na["sub"], re_formula="~ld").fit(method="lbfgs", reml=True)
        out["nonadjacent_mixed"] = dict(ld=float(fit.params["ld"]), se=float(fit.bse["ld"]), p_two_sided=float(fit.pvalues["ld"]),
                                        n_pairs=len(na), n_patients=int(na["sub"].nunique()), converged=bool(fit.converged))
    except Exception as e:
        out["nonadjacent_mixed"] = dict(error=str(e))
    pen = []
    for (coh, sub), g in P.groupby(["cohort", "sub"]):
        if (g.n_between == 0).sum() >= 2 and (g.n_between >= 1).sum() >= 2:
            pen.append(dict(cohort=coh, sub=sub, penalty=float(g.chi[g.n_between >= 1].mean() - g.chi[g.n_between == 0].mean()),
                            adj_mm=float(np.exp(g.ld[g.n_between == 0]).median())))
    pen = pd.DataFrame(pen)
    pen.to_csv(RES / "adjacent_penalty.csv", index=False)
    for name, g in (("pooled", pen), ("ds004752", pen[pen.cohort == "ds004752"]), ("HUP", pen[pen.cohort == "HUP"])):
        out[f"adjacent_penalty_{name}"] = test(g.penalty.to_numpy())
    # distance profile for the figure (patient-centred)
    P["d"] = np.exp(P.ld)
    P["chi_c"] = P.chi - P.groupby(["cohort", "sub"]).chi.transform("mean")
    P["bin"] = pd.cut(P.d, [3, 5, 7, 10, 14, 20, 30, 80])
    prof = P.groupby(["cohort", "bin", "sub"], observed=True).chi_c.mean().reset_index()
    prof["bin"] = prof["bin"].astype(str)
    prof.to_csv(RES / "distance_profile.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=args.jobs)(delayed(seeg_unit)(str(p)) for p in sorted(SEEG.glob("*.npz")))
    R = pd.DataFrame([r for r, _, _ in res])
    R.to_csv(RES / "seeg_units.csv", index=False)
    f = res[0][2]
    rows = []
    for (r, spec, _) in res:
        for k in ("near", "far"):
            if spec[k] is not None:
                rows.append(dict(cohort=r["cohort"], sub=r["sub"], kind=k, **{f"{x:.1f}": float(v) for x, v in zip(f, spec[k])}))
    pd.DataFrame(rows).to_csv(RES / "near_far_spectra.csv", index=False)
    J = {}
    for name, g in (("ds004752_confirm", R[R.split == "confirm"]), ("ds004752_all", R[R.cohort == "ds004752"]),
                    ("HUP", R[R.cohort == "HUP"]), ("pooled", R)):
        J[name] = {c: test(g[c].to_numpy()) for c in R.columns if c.startswith("C1_") or c.startswith("band")}
    # near-far difference: straight-line R2 over 2-40 Hz (log-log), group mean
    S = pd.DataFrame(rows)
    fc = np.array([float(c) for c in S.columns[3:]])
    for coh in ("ds004752", "HUP"):
        g = S[S.cohort == coh]
        nf = g[g.kind == "near"].set_index("sub").iloc[:, 2:].sub(g[g.kind == "far"].set_index("sub").iloc[:, 2:]).dropna()
        dm = nf.mean(0).to_numpy()
        m = fc <= 40
        x = np.log10(fc[m]); y = dm[m]
        res_ = y - np.polyval(np.polyfit(x, y, 1), x)
        J[f"near_minus_far_linear_R2_{coh}"] = float(1 - res_.var() / y.var())
        J[f"near_minus_far_n_{coh}"] = int(len(nf))
    C2 = pd.DataFrame([c2_unit(s) for s in io.subjects()])
    C2.to_csv(RES / "c2_ocular.csv", index=False)
    for name, g in (("confirm", C2[C2.split == "confirm"]), ("all", C2)):
        J[f"C2_{name}"] = {c: test(g[c].to_numpy()) for c in C2.columns if c.startswith("C2_")}
    J.update(pooled_nonadjacent())
    json.dump(J, open(RES / "seeg_group.json", "w"), indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
