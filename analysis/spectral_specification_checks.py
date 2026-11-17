"""Spectral specification and additional control analyses (Results 3.3-3.4; S4, S5; Figure 5).

models : five specifications per spectrum (fixed fit over the band and over a wider range; knee fit over the wider
         range and over the band, each summarised by the band slope of the fitted aperiodic curve; asymptotic knee
         exponent) for scalp P-2, the eye-state interaction and depth C1; fit retention and knee location.
measure: ds003478 eyes-open and eyes-closed units after EOG regression, and probable-interpolation flags per channel.
eog    : eye-state x derivation interaction after EOG regression.
interp : eye-state results without probably interpolated channels (neighbour R^2 >= 0.99 or 0.98).
grid   : zero-sum check of the retained graph modes.
common : per-channel specification values and paired contrasts on spectra retained under both specifications.
Usage: python analysis/spectral_specification_checks.py models --jobs 20   (then measure, eog, interp, grid, common)
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from apex import fr, pairs, scalp, stats  # noqa: E402
from apex.paths import CACHE, DATA  # noqa: E402

SCALP = Path(f"{CACHE}/final_round/scalp")
SEEG = Path(f"{CACHE}/final_round/seeg")
FLAGS = Path(f"{CACHE}/final_round/interp_flags")
GRID = Path(f"{CACHE}/grid_v2")
RES = Path("results/revision")
DEV = {f"sub-{i:02d}" for i in range(1, 6)}
R2_INTERP = (0.99, 0.98)


def band_slope(spec, lo, hi):
    """Least-squares log-log slope (as an exponent) of the fitted aperiodic component over [lo, hi], 30 log points."""
    if not np.isfinite(spec["exponent"]):
        return np.nan
    f = np.geomspace(lo, hi, 30)
    k = spec["knee"] if np.isfinite(spec["knee"]) else 0.0
    y = -np.log10(np.maximum(k + f ** spec["exponent"], 1e-300))
    x = np.log10(f) - np.log10(f).mean()
    return float(-((y - y.mean()) @ x) / (x @ x))


def knee_freq(spec):
    k, e = spec["knee"], spec["exponent"]
    if not (np.isfinite(k) and np.isfinite(e) and k > 0 and e > 0):
        return np.nan
    lf = np.log10(k) / e
    return float(10.0 ** lf) if lf < 6 else np.inf


def spec_set(f, p, band, wide):
    """Four fits; returns dict of quantities. band = slope band (and narrow fit range), wide = knee fit range."""
    out = {}
    fits = {"F_band": fr.sp_fit(f, p, "fixed", band), "F_wide": fr.sp_fit(f, p, "fixed", wide),
            "K_wide": fr.sp_fit(f, p, "knee", wide), "K_band": fr.sp_fit(f, p, "knee", band)}
    for name, s in fits.items():
        ok = np.isfinite(s["r2"]) and s["r2"] >= fr.R2_MIN
        out[f"{name}_ok"] = ok
        out[f"{name}_slope"] = band_slope(s, *band) if ok else np.nan
        if name.startswith("K"):
            fk = knee_freq(s)
            rng = wide if name == "K_wide" else band
            out[f"{name}_kneefreq"] = fk
            out[f"{name}_knee_in_range"] = bool(np.isfinite(fk) and rng[0] <= fk <= rng[1])
    out["K_wide_asym"] = fits["K_wide"]["exponent"] if out["K_wide_ok"] else np.nan
    return out


SPECS = ("F_band", "F_wide", "K_wide", "K_band", "K_wide_asym")


def _val(d, s):
    return d["K_wide_asym"] if s == "K_wide_asym" else d[f"{s}_slope"]


def scalp_unit(path):
    z = np.load(path, allow_pickle=True)
    tag = Path(path).stem
    ds, rest = tag.split("_", 1)
    sub, cond = rest.rsplit("_", 1)
    f, inner = z["f"], np.nonzero(z["interior"])[0]
    A = [spec_set(f, z["Ma"][i], (2.0, 20.0), (1.0, 30.0)) for i in inner]
    L = [spec_set(f, z["Ml"][i], (2.0, 20.0), (1.0, 30.0)) for i in inner]
    r = dict(dataset=ds, sub=sub, cond=cond, n_interior=len(inner))
    for s in SPECS:
        a = np.array([_val(d, s) for d in A]); l_ = np.array([_val(d, s) for d in L])
        ok = np.isfinite(a) & np.isfinite(l_)
        r[f"{s}_P2"] = float((a - l_)[ok].mean()) if ok.sum() >= 5 else np.nan
        r[f"{s}_avg"] = float(a[ok].mean()) if ok.sum() >= 5 else np.nan
        r[f"{s}_lap"] = float(l_[ok].mean()) if ok.sum() >= 5 else np.nan
    for der, D in (("avg", A), ("lap", L)):
        for s in ("F_band", "F_wide", "K_wide", "K_band"):
            r[f"retain_{s}_{der}"] = float(np.mean([d[f"{s}_ok"] for d in D]))
        for s in ("K_wide", "K_band"):
            r[f"kneein_{s}_{der}"] = float(np.mean([d[f"{s}_knee_in_range"] for d in D if d[f"{s}_ok"]] or [np.nan]))
            r[f"kneefreq_{s}_{der}"] = float(np.nanmedian([d[f"{s}_kneefreq"] for d in D if d[f"{s}_ok"]] or [np.nan]))
    return r


def seeg_unit(path):
    z = np.load(path, allow_pickle=True)
    cohort, sub = Path(path).stem.split("_", 1)
    f, P = z["f"], fr.interp_line(z["f"], z["P"][z["prim"]].astype(float), float(z["line"]))
    ld = np.log(z["dist"][z["prim"]])
    D = [spec_set(f, p, (2.0, 40.0), (1.0, 40.0)) for p in P]
    r = dict(cohort=cohort, sub=sub, split="dev" if sub in DEV else ("confirm" if cohort == "ds004752" else "hup"), n_pairs=len(ld))
    for s in SPECS:
        r[f"{s}_C1"] = fr.slope(np.array([_val(d, s) for d in D]), ld)
    for s in ("F_band", "F_wide", "K_wide", "K_band"):
        r[f"retain_{s}"] = float(np.mean([d[f"{s}_ok"] for d in D]))
    for s in ("K_wide", "K_band"):
        r[f"kneein_{s}"] = float(np.mean([d[f"{s}_knee_in_range"] for d in D if d[f"{s}_ok"]] or [np.nan]))
    return r


def test(v, alt="greater"):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(n=int(len(v)))
    b = np.random.default_rng(0).choice(v, (10000, len(v))).mean(1)
    return dict(stats.signflip_test(v, alt), ci95=[float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))])


def models(jobs):
    files = [p for p in sorted(glob.glob(str(SCALP / "*.npz"))) if ".tmp" not in p and
             (Path(p).stem.startswith("anphy_") or Path(p).stem.endswith(("_EO", "_EC")) and Path(p).stem.startswith("ds003478_")
              or Path(p).stem.endswith("_full") and Path(p).stem.startswith("modma_"))]
    S = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(scalp_unit)(p) for p in files))
    S.to_csv(RES / "models_scalp_units.csv", index=False)
    C = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(seeg_unit)(str(p)) for p in sorted(SEEG.glob("*.npz"))))
    C.to_csv(RES / "models_seeg_units.csv", index=False)
    J = {}
    for (ds, cond), g in S.groupby(["dataset", "cond"]):
        J[f"P2_{ds}_{cond}"] = {s: test(g[f"{s}_P2"]) for s in SPECS}
        J[f"retention_{ds}_{cond}"] = {c: float(g[c].mean()) for c in g.columns if c.startswith(("retain_", "kneein_", "kneefreq_"))}
    P = S[S.dataset == "ds003478"].pivot(index="sub", columns="cond")
    for s in SPECS:
        a = P[f"{s}_avg"]["EO"] - P[f"{s}_avg"]["EC"]; l_ = P[f"{s}_lap"]["EO"] - P[f"{s}_lap"]["EC"]
        J[f"EOEC_{s}"] = dict(avg=test(a, "two-sided"), lap=test(l_, "two-sided"), avg_minus_lap=test(a - l_, "greater"))
    for name, g in (("confirm", C[C.split == "confirm"]), ("ds004752_all", C[C.cohort == "ds004752"]), ("HUP", C[C.cohort == "HUP"])):
        J[f"C1_{name}"] = {s: test(g[f"{s}_C1"]) for s in SPECS}
        J[f"retention_C1_{name}"] = {c: float(g[c].mean()) for c in g.columns if c.startswith(("retain_", "kneein_"))}
    json.dump(J, open(RES / "models_group.json", "w"), indent=1, default=float)


def neighbours(pos, k=6):
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return np.argsort(D, axis=1)[:, 1:k + 1]


def interp_r2(S, pos):
    """S [seg, ch, t] -> R^2 of each channel regressed on its 6 nearest neighbours (all segments concatenated)."""
    X = np.concatenate(list(S), axis=-1)
    X = X - X.mean(1, keepdims=True)
    nb = neighbours(pos)
    r2 = np.empty(len(pos))
    for i in range(len(pos)):
        A = X[nb[i]].T
        b, *_ = np.linalg.lstsq(A, X[i], rcond=None)
        r2[i] = 1.0 - np.var(X[i] - A @ b) / np.var(X[i])
    return r2


def ds3478_reg_one(sub, files):
    """Same channels, segments and EO/EC selection as fr_scalp.ds3478_one, applied to the EOG-regressed signal."""
    import scalp_ds003478_eye_state_ocular as F
    import scalp_derivations as FS
    try:
        runs = [F.load_run(f) for f in files]
        names = [c for c in runs[0][0] if all(c in r[0] for r in runs)]
        pos = runs[0][1][[runs[0][0].index(c) for c in names]]
        keep_ch = np.ones(len(names), bool)
        for nm, _, Xa, _, _ in runs:
            S = F.segs(Xa[[nm.index(c) for c in names]], Xa.shape[1] // F.L)
            med = np.median(np.ptp(S, axis=-1), axis=0)
            keep_ch &= (med < 5 * np.median(med)) & (med > 0.1 * np.median(med))
        kn = [c for c, k in zip(names, keep_ch) if k]
        P = pos[keep_ch]
        base, reg, lab, nreg = [], [], [], 0
        for nm, _, Xa, eog, ev in runs:
            ix = [nm.index(c) for c in kn]
            nn = Xa.shape[1] // F.L
            S = F.segs(Xa[ix], nn)
            ptp = np.ptp(S, axis=-1)
            ks = np.all(ptp <= 5 * np.median(ptp, axis=0, keepdims=True), axis=1)
            base.append(S[ks]); lab.append(F.labels(ev, nn)[ks])
            if eog.shape[0] == 2 and np.all(np.std(eog, axis=1) > 0):
                Xr, _ = F.eog_regress(Xa[ix], eog, ks, nn)
                reg.append(F.segs(Xr, nn)[ks]); nreg += 1
        base, lab = np.concatenate(base), np.concatenate(lab)
        FLAGS.mkdir(parents=True, exist_ok=True)
        np.savez(FLAGS / f"{sub}.npz", names=np.array(kn), pos=P, r2=interp_r2(base, P))
        out = [f"{sub}: flags written"]
        k = min(int((lab == "EO").sum()), int((lab == "EC").sum()))
        if nreg == len(runs) and k >= F.MIN_SEG:
            reg = np.concatenate(reg)
            rng = np.random.default_rng(0)
            for c in ("EO", "EC"):
                ii = np.sort(rng.choice(np.nonzero(lab == c)[0], k, replace=False))
                out.append(FS.compute_unit(f"ds003478_{sub}_{c}reg", reg[ii], F.FS, P, irasa_pairs=False))
        return out
    except Exception as e:
        return [f"{sub} ERROR {type(e).__name__}: {e}"]


def measure(jobs):
    by = {}
    for f in sorted(glob.glob(f"{DATA}/ds003478/sub-*/eeg/*task-Rest*_eeg.set")):
        by.setdefault(Path(f).name.split("_")[0], []).append(f)
    msgs = Parallel(n_jobs=jobs)(delayed(ds3478_reg_one)(s, f) for s, f in sorted(by.items()))
    print("\n".join(m for mm in msgs for m in mm))


def chi_int(z, key, keep=None):
    """Interior-mean IRASA exponent (2-20 Hz) for avg ('Ia') or Laplacian ('Il'), optionally restricted."""
    import scalp_derivations as FS
    chi = pairs.band_exponent(FS.CEN, z[key].astype(float), (2.0, 20.0))
    m = z["interior"].astype(bool) if keep is None else (z["interior"].astype(bool) & keep)
    return float(chi[m].mean()) if m.sum() >= 5 else np.nan


def sp_int(z, key):
    chi = fr.sp_exponents(z["f"], z[key].astype(float), "fixed", (2.0, 20.0))
    m = z["interior"].astype(bool) & np.isfinite(chi)
    return float(chi[m].mean()) if m.sum() >= 5 else np.nan


def eog_one(sub):
    r = dict(sub=sub)
    for c in ("EO", "EC", "EOreg", "ECreg"):
        z = np.load(SCALP / f"ds003478_{sub}_{c}.npz", allow_pickle=True)
        r[f"irasa_avg_{c}"], r[f"irasa_lap_{c}"] = chi_int(z, "Ia"), chi_int(z, "Il")
        r[f"sp_avg_{c}"], r[f"sp_lap_{c}"] = sp_int(z, "Ma"), sp_int(z, "Ml")
    return r


def eog(jobs):
    subs = sorted({Path(p).stem.split("_")[1] for p in glob.glob(str(SCALP / "ds003478_*_EOreg.npz"))})
    subs = [s for s in subs if (SCALP / f"ds003478_{s}_ECreg.npz").exists()]
    D = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(eog_one)(s) for s in subs))
    D.to_csv(RES / "eog_eyestate_units.csv", index=False)
    J = {"n": int(len(D))}
    for est in ("irasa", "sp"):
        for v, (o, c) in (("unregressed", ("EO", "EC")), ("eog_regressed", ("EOreg", "ECreg"))):
            a = D[f"{est}_avg_{o}"] - D[f"{est}_avg_{c}"]; l_ = D[f"{est}_lap_{o}"] - D[f"{est}_lap_{c}"]
            J[f"{est}_{v}"] = dict(avg=test(a, "two-sided"), lap=test(l_, "two-sided"), avg_minus_lap=test(a - l_, "greater"))
            D[f"{est}_{v}_int"] = a - l_
        J[f"{est}_retained_fraction"] = float(D[f"{est}_eog_regressed_int"].mean() / D[f"{est}_unregressed_int"].mean())
    json.dump(J, open(RES / "eog_eyestate_group.json", "w"), indent=1, default=float)


def interp_one(sub):
    fl = np.load(FLAGS / f"{sub}.npz", allow_pickle=True)
    r = dict(sub=sub, n_ch=len(fl["r2"]), r2_median=float(np.median(fl["r2"])))
    zs = {c: np.load(SCALP / f"ds003478_{sub}_{c}.npz", allow_pickle=True) for c in ("EO", "EC")}
    assert np.allclose(zs["EO"]["pos"], fl["pos"])
    nb = neighbours(fl["pos"])
    for thr in R2_INTERP:
        flag = fl["r2"] >= thr
        keep = ~flag & ~flag[nb].any(1)
        r[f"n_flag_{thr}"] = int(flag.sum())
        r[f"n_interior_kept_{thr}"] = int((zs["EO"]["interior"].astype(bool) & keep).sum())
        for c, z in zs.items():
            r[f"avg_{c}_{thr}"], r[f"lap_{c}_{thr}"] = chi_int(z, "Ia", keep), chi_int(z, "Il", keep)
    for c, z in zs.items():
        r[f"avg_{c}_all"], r[f"lap_{c}_all"] = chi_int(z, "Ia"), chi_int(z, "Il")
    return r


def interp(jobs):
    subs = sorted({Path(p).stem.split("_")[1] for p in glob.glob(str(SCALP / "ds003478_*_EC.npz"))})
    subs = [s for s in subs if (FLAGS / f"{s}.npz").exists() and (SCALP / f"ds003478_{s}_EO.npz").exists()]
    D = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(interp_one)(s) for s in subs))
    D.to_csv(RES / "interp_units.csv", index=False)
    J = {"n": int(len(D)), "r2_median_of_medians": float(D.r2_median.median())}
    for v in ("all",) + R2_INTERP:
        a = D[f"avg_EO_{v}"] - D[f"avg_EC_{v}"]; l_ = D[f"lap_EO_{v}"] - D[f"lap_EC_{v}"]
        J[f"set_{v}"] = dict(P2_EO=test(D[f"avg_EO_{v}"] - D[f"lap_EO_{v}"]), P2_EC=test(D[f"avg_EC_{v}"] - D[f"lap_EC_{v}"]),
                            avg_minus_lap=test(a - l_))
        if v != "all":
            J[f"set_{v}"]["subjects_with_flags"] = int((D[f"n_flag_{v}"] > 0).sum())
            J[f"set_{v}"]["flagged_channels_total"] = int(D[f"n_flag_{v}"].sum())
    json.dump(J, open(RES / "interp_group.json", "w"), indent=1, default=float)


def grid():
    tags = pd.read_csv("results/grid_v2/arrays.csv").dropna(subset=["n_contacts"]).tag
    rows = []
    for t in sorted(tags):
        z = np.load(GRID / f"{t}.npz", allow_pickle=True)
        U = z["U"]
        rows.append(dict(tag=t, n_contacts=U.shape[0], n_modes=U.shape[1], max_abs_colsum=float(np.abs(U.sum(0)).max()),
                         min_eig_retained=float(z["lam"].min())))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "grid_mode_check.csv", index=False)
    json.dump(dict(n_grids=len(D), max_abs_colsum=float(D.max_abs_colsum.max()), min_eig_retained=float(D.min_eig_retained.min())),
              open(RES / "grid_mode_check.json", "w"), indent=1)


CONTRASTS = {"range": ("F_wide", "F_band"), "model": ("K_wide", "F_wide"), "slope_definition": ("K_wide_asym", "K_wide")}


def scalp_channels(path):
    """Per-interior-channel values of the five specifications (NaN where not retained), both derivations."""
    z = np.load(path, allow_pickle=True)
    tag = Path(path).stem
    ds, rest = tag.split("_", 1)
    sub, cond = rest.rsplit("_", 1)
    f, inner = z["f"], np.nonzero(z["interior"])[0]
    out = []
    for der, key in (("avg", "Ma"), ("lap", "Ml")):
        for ch in inner:
            d = spec_set(f, z[key][ch], (2.0, 20.0), (1.0, 30.0))
            out.append(dict(dataset=ds, sub=sub, cond=cond, der=der, ch=int(ch), **{s: _val(d, s) for s in SPECS}))
    return out


def seeg_pairs(path):
    z = np.load(path, allow_pickle=True)
    cohort, sub = Path(path).stem.split("_", 1)
    f, P = z["f"], fr.interp_line(z["f"], z["P"][z["prim"]].astype(float), float(z["line"]))
    ld = np.log(z["dist"][z["prim"]])
    split = "dev" if sub in DEV else ("confirm" if cohort == "ds004752" else "hup")
    return [dict(cohort=cohort, sub=sub, split=split, pair=i, logd=float(ld[i]),
                 **{s: _val(d, s) for s in SPECS}) for i, d in enumerate(spec_set(f, p, (2.0, 40.0), (1.0, 40.0)) for p in P)]


def boot(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    b = np.random.default_rng(0).choice(v, (10000, len(v))).mean(1)
    return dict(mean=float(v.mean()), ci95=[float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], n=int(len(v)))


def common_stats(Sc, Dp):
    """Paired contrasts on spectra retained under both specifications of each contrast."""
    J = {}
    for c, (a, b) in CONTRASTS.items():
        ok = Sc[a].notna() & Sc[b].notna()
        W = Sc[ok].pivot_table(index=["dataset", "sub", "cond", "ch"], columns="der", values=[a, b])
        W = W.dropna()                                           # retained under both specs for both derivations
        p2 = (W[(a, "avg")] - W[(a, "lap")]) - (W[(b, "avg")] - W[(b, "lap")])
        U = p2.groupby(level=[0, 1, 2]).agg(["mean", "size"])
        U = U[U["size"] >= 5]
        for (ds, cond), g in U.groupby(level=[0, 2]):
            J.setdefault(f"P2_{ds}_{cond}", {})[c] = dict(boot(g["mean"]), channels_median=float(g["size"].median()))
        E = p2.loc["ds003478"]
        E = E.reset_index().pivot_table(index=["sub", "ch"], columns="cond", values=0).dropna()   # channel in both EO and EC
        per = (E["EO"] - E["EC"]).groupby(level=0).agg(["mean", "size"])
        per = per[per["size"] >= 5]
        J.setdefault("EOEC_interaction", {})[c] = dict(boot(per["mean"]), channels_median=float(per["size"].median()))
        for name, g in (("confirm", Dp[Dp.split == "confirm"]), ("ds004752_all", Dp[Dp.cohort == "ds004752"]), ("HUP", Dp[Dp.cohort == "HUP"])):
            v, npairs = [], []
            for _, h in g.groupby("sub"):
                k = h[a].notna() & h[b].notna()
                if k.sum() >= 3:
                    v.append(fr.slope(h[a][k].to_numpy(), h.logd[k].to_numpy()) - fr.slope(h[b][k].to_numpy(), h.logd[k].to_numpy()))
                    npairs.append(int(k.sum()))
            J.setdefault(f"C1_{name}", {})[c] = dict(boot(v), pairs_median=float(np.median(npairs)))
    return J


def common(jobs):
    files = [p for p in sorted(glob.glob(str(SCALP / "*.npz"))) if ".tmp" not in p and
             (Path(p).stem.startswith("anphy_") or Path(p).stem.endswith(("_EO", "_EC")) and Path(p).stem.startswith("ds003478_")
              or Path(p).stem.endswith("_full") and Path(p).stem.startswith("modma_"))]
    Sc = pd.DataFrame([r for rr in Parallel(n_jobs=jobs)(delayed(scalp_channels)(p) for p in files) for r in rr])
    Sc.to_csv(RES / "common_scalp_channels.csv.gz", index=False)
    Dp = pd.DataFrame([r for rr in Parallel(n_jobs=jobs)(delayed(seeg_pairs)(str(p)) for p in sorted(SEEG.glob("*.npz"))) for r in rr])
    Dp.to_csv(RES / "common_seeg_pairs.csv.gz", index=False)
    json.dump(common_stats(Sc, Dp), open(RES / "common_paired_specs.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["models", "measure", "eog", "interp", "grid", "common"])
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    {"models": lambda: models(a.jobs), "measure": lambda: measure(a.jobs), "eog": lambda: eog(a.jobs),
     "interp": lambda: interp(a.jobs), "grid": grid, "common": lambda: common(a.jobs)}[a.step]()
    print("done", a.step)


if __name__ == "__main__":
    main()
