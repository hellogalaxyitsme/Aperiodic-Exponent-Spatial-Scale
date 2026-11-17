"""Subdural grids (Results 3.2) and the two-component mixture model (Results 3.5, S3).

measure: per array (>= 8 good contacts; seizure-onset and resected contacts excluded), 8-s epochs, IRASA fractal spectra
  of every within-array pair and of every non-constant graph-Laplacian mode; tangential pair slope (5-80 mm) and mode
  slope (exponent of six log-eigenvalue bins on ln sqrt(eigenvalue)).
fit: two-component model C(f) = a_l(f) K(l_l) + a_g(f) K(l_g) + a_0(f) I fitted on half of the grids; group-median
  parameters frozen, then evaluated on the other half and propagated to the scalp montages through the New York Head.
Usage: python analysis/grid_modes_and_mixture_model.py measure --jobs 10
       python analysis/grid_modes_and_mixture_model.py fit
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from apex import io, pairs, spectra, stats  # noqa: E402
from apex.paths import CACHE, DATA  # noqa: E402

HUP = Path(f"{DATA}/HUP_ds004100")
CACHE = Path(f"{CACHE}/grid_v2")
OUT = Path("results/grid_v2")
HSET = np.round(np.arange(1.1, 1.95, 0.05), 4)
_, CEN = pairs.log_bands(2.0, 40.0, 40)
LL = np.array([2, 3, 4, 6, 8, 11, 15, 20, 30], float)
LG = np.array([30, 45, 60, 90, 135, 200, 300], float)
MAX_EPOCHS = 80


def slope(y, x):
    x = x - x.mean()
    return float((y - y.mean()) @ x / (x @ x)) if len(x) >= 3 and (x @ x) > 0 else np.nan


def graph_modes(xyz):
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    sig = np.median(np.sort(D, axis=1)[:, 1])
    W = np.exp(-D ** 2 / (2 * sig ** 2))
    np.fill_diagonal(W, 0)
    lam, U = np.linalg.eigh(np.diag(W.sum(1)) - W)
    return lam[1:], U[:, 1:], D, sig


def irasa_at(Y, fs):
    """Y [ep, k, t] -> fractal PSD at CEN, averaged over epochs (linear), [k, nb]."""
    out = []
    for s in range(0, Y.shape[1], 128):
        f, frac, _ = spectra.irasa(Y[:, s:s + 128], fs, hset=HSET, nperseg=int(2 * fs))
        out.append(frac.mean(0))
    P = np.concatenate(out, 0)
    m = f > 0
    return np.exp(np.apply_along_axis(lambda y: np.interp(np.log(CEN), np.log(f[m]), y), -1, np.log(np.maximum(P[:, m], 1e-300))))


def measure_array(tag, S, fs, xyz, names):
    """S [ep, ch, t] referential. Saves pair/mode/referential spectra and returns measured statistics."""
    lam, U, D, sig = graph_modes(xyz)
    ia, ib = np.triu_indices(len(names), 1)
    Pp = irasa_at(S[:, ia] - S[:, ib], fs)
    Pm = irasa_at(np.einsum("ck,ect->ekt", U, S), fs)
    Pr = irasa_at(S, fs)
    np.savez_compressed(CACHE / f"{tag}.npz", Pp=Pp, Pm=Pm, Pr=Pr, lam=lam, U=U, xyz=xyz, names=np.array(names), n_ep=S.shape[0])
    return dict(tag=tag, **stats_from(Pp, Pm, Pr, lam, D[ia, ib]), n_contacts=len(names), n_epochs=int(S.shape[0]), nn_mm=float(sig))


def stats_from(Pp, Pm, Pr, lam, d):
    r = {}
    ok = (d >= 5) & (d <= 80)
    for band, t in (((2.0, 40.0), "2_40"), ((2.0, 20.0), "2_20")):
        chi_p = pairs.band_exponent(CEN, Pp, band)
        r[f"pair_slope_{t}"] = slope(chi_p[ok], np.log(d[ok]))
        e = np.exp(np.linspace(np.log(lam.min()), np.log(lam.max()) + 1e-9, 7))
        bins = np.clip(np.searchsorted(e, lam, side="right") - 1, 0, 5)
        use = [k for k in range(6) if np.any(bins == k)]
        chi_b = pairs.band_exponent(CEN, np.stack([Pm[bins == k].sum(0) for k in use]), band)
        r[f"mode_slope_{t}"] = slope(chi_b, np.log(np.sqrt(np.array([lam[bins == k].mean() for k in use]))))
        r[f"mode_chi_low_{t}"], r[f"mode_chi_high_{t}"] = float(chi_b[0]), float(chi_b[-1])
        r[f"ref_minus_pair_chi_{t}"] = float(pairs.band_exponent(CEN, Pr, band).mean() - chi_p.mean())
    return r


def hup_arrays(sub):
    import mne
    d = HUP / sub / "ses-presurgery" / "ieeg"
    el = pd.read_csv(sorted(d.glob("*acq-ecog_space-*_electrodes.tsv"))[0], sep="\t")
    el = el[pd.to_numeric(el["x"], errors="coerce").notna()].copy()
    el[["x", "y", "z"]] = el[["x", "y", "z"]].astype(float)
    if np.nanmax(np.abs(el[["x", "y", "z"]].to_numpy())) < 1.0:
        el[["x", "y", "z"]] *= 1000.0
    runs = sorted(d.glob("*task-interictal_acq-ecog_run-*_ieeg.edf"))
    good, segs, fs = None, [], None
    for r in runs:
        ch = pd.read_csv(str(r).replace("_ieeg.edf", "_channels.tsv"), sep="\t")
        sd = ch["status_description"].fillna("").astype(str).str.lower()
        g = set(ch.loc[(ch["status"] == "good") & (ch["type"].isin(["ECOG", "SEEG"])) & ~sd.str.contains("soz") & ~sd.str.contains("resect"), "name"])
        good = g if good is None else good & g
    names = [n for n in el["name"] if n in good]
    for r in runs:
        raw = mne.io.read_raw_edf(r, preload=True, verbose="ERROR")
        X = raw.get_data(picks=names)
        fs0 = raw.info["sfreq"]
        if fs0 > 520:
            X = spectra.decimate_to(X, fs0, 500.0); fs0 = 500.0
        fs = fs0
        L = int(8 * fs)
        segs += [X[:, i * L:(i + 1) * L] for i in range(X.shape[1] // L)]
    S = np.stack(segs)
    ptp = np.ptp(S, axis=-1)
    S = S[np.all(ptp <= 5 * np.median(ptp, axis=0, keepdims=True), axis=1)][:MAX_EPOCHS]
    pos = el.set_index("name").loc[names, ["x", "y", "z"]].to_numpy()
    # Amendment made before any grid result: names look like "EEG LG 01-Ref"; array = letters after "EEG ".
    # HUP types every contact ECOG (depth shafts included), so only subdural GRIDS (>= 16 contacts) are analysed.
    pref = pd.Series(names).str.extract(r"^(?:EEG\s+)?([A-Za-z]+)\s*0*\d+")[0].fillna("?").to_numpy()
    out = []
    for p in np.unique(pref):
        m = pref == p
        if m.sum() >= 16 and p not in {"EKG", "ECG", "?"}:
            out.append((f"{sub}_{p}", S[:, m], fs, pos[m], list(np.array(names)[m])))
    return out


def sub10_arrays():
    Sall, xyz, names, fs = [], None, None, 500.0
    for s in io.sessions("sub-10"):
        el = s.electrodes()
        g = el[el["kind"] == "grid"].sort_values(["row", "idx"])
        names = g["name"].tolist()
        xyz = g[["x", "y", "z"]].to_numpy(float)
        X, ch, f0, ev = s.load("ieeg", drop_artifacts=True, picks=names)
        Sall.append(spectra.decimate_to(X[:, [ch.index(n) for n in names]], f0, 500.0))
    return [("sub-10_grid", np.concatenate(Sall, 0)[:MAX_EPOCHS], fs, xyz, names)]


def measure_subject(sub):
    try:
        arrs = sub10_arrays() if sub == "sub-10" else hup_arrays(sub)
        return [measure_array(*a) for a in arrs] or [dict(tag=sub, error="no array with >= 8 good contacts")]
    except Exception as e:
        return [dict(tag=sub, error=f"{type(e).__name__}: {e}")]


def ecog_subjects():
    return sorted(p.name for p in HUP.glob("sub-*") if list((p / "ses-presurgery" / "ieeg").glob("*task-interictal_acq-ecog*_ieeg.edf")))


# ---------------------------------------------------------------- two-component model
def design(lam_l, lam_g, U, D, ia, ib):
    Kl, Kg = np.exp(-D / lam_l), np.exp(-D / lam_g)
    Ap = np.stack([2 * (1 - Kl[ia, ib]), 2 * (1 - Kg[ia, ib]), 2 * np.ones(len(ia))], 1)
    Am = np.stack([np.einsum("ck,cd,dk->k", U, Kl, U), np.einsum("ck,cd,dk->k", U, Kg, U), np.ones(U.shape[1])], 1)
    return Ap, Am


def fit_array(z):
    xyz, U, Pp, Pm = z["xyz"], z["U"], z["Pp"], z["Pm"]
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    ia, ib = np.triu_indices(len(xyz), 1)
    wp, wm = 1 / np.sqrt(len(ia)), 1 / np.sqrt(U.shape[1])
    best = None
    for l1 in LL:
        for l2 in LG:
            Ap, Am = design(l1, l2, U, D, ia, ib)
            amps, cost = [], 0.0
            for k in range(len(CEN)):
                A = np.vstack([Ap / Pp[:, k:k + 1] * wp, Am / Pm[:, k:k + 1] * wm])
                y = np.concatenate([np.full(len(ia), wp), np.full(U.shape[1], wm)])
                a, rn = nnls(A, y)
                amps.append(a); cost += rn ** 2
            if best is None or cost < best[0]:
                best = (cost, l1, l2, np.array(amps))
    cost, l1, l2, amps = best
    return dict(cost=cost, lam_l=l1, lam_g=l2, amps=amps)       # amps [nb, 3]


def predict_array(z, lam_l, lam_g, amps):
    xyz, U = z["xyz"], z["U"]
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    ia, ib = np.triu_indices(len(xyz), 1)
    Ap, Am = design(lam_l, lam_g, U, D, ia, ib)
    return Ap @ amps.T, Am @ amps.T                            # [npairs, nb], [nmodes, nb]


def chi_of(a):
    return float(pairs.band_exponent(CEN, np.maximum(a, 1e-30)[None], (2.0, 40.0))[0]), float(pairs.band_exponent(CEN, np.maximum(a, 1e-30)[None], (2.0, 20.0))[0])


def scalp_predictions(fr):
    """NY Head predictions for the two montages, derivations as in apex.hdeeg.metrics_v2."""
    import mne
    import torch
    from apex import hdeeg, scalp
    from kernel_predict_scalp import procrustes
    from kernel_head_model import G_lambda, load_nyhead
    V, vc, area, labs, locs = load_nyhead()
    up = [l.upper() for l in labs]
    seg = np.load(sorted(p for p in Path(f"{CACHE}/anphy_segments").glob("EPCTL*.npz") if ".tmp" not in p.name)[0], allow_pickle=True)
    mont = {"anphy83": ([str(c) for c in seg["chans"]], seg["pos"])}
    # ds003478 .set files carry NaN positions; the BIDS electrodes.tsv has coordinates in mm (metadata only)
    elf = sorted(Path(f"{DATA}/ds003478").glob("sub-*/eeg/*task-Rest*_electrodes.tsv"))[0]
    E = pd.read_csv(elf, sep="\t")
    E = E[~E["name"].str.upper().isin({"HEOG", "VEOG", "CB1", "CB2"}) & np.isfinite(E[["x", "y", "z"]].astype(float)).all(1)]
    mont["ds003478_64"] = (E["name"].tolist(), E[["x", "y", "z"]].to_numpy(float))
    lams = np.exp(np.linspace(np.log(1.0), np.log(400.0), 48))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    preds = {}
    for mname, (chans, pos) in mont.items():
        matched = [(i, up.index(c.upper())) for i, c in enumerate(chans) if c.upper() in up]
        T = procrustes(pos[[i for i, _ in matched]], locs[[j for _, j in matched]])
        nyidx = np.array([int(np.argmin(np.linalg.norm(locs - p, axis=1))) for p in T(pos)])
        n = len(chans)
        A = scalp.avgref_matrix(n)
        Lap = scalp.hjorth_matrix(pos, k=6)
        inner = hdeeg.interior_mask(pos)
        Y, deg = scalp.sph_basis(pos, 8)
        Yr, dg = Y[:, deg >= 1], deg[deg >= 1]
        coef = np.linalg.solve(Yr.T @ Yr + 1e-3 * np.trace(Yr.T @ Yr) / Yr.shape[1] * np.eye(Yr.shape[1]), Yr.T) @ A
        rng = np.random.default_rng(0)
        ia, ib = np.triu_indices(n, 1)
        pick = rng.choice(len(ia), size=min(600, len(ia)), replace=False)
        Pm = np.zeros((len(pick), n)); Pm[np.arange(len(pick)), ia[pick]] = 1; Pm[np.arange(len(pick)), ib[pick]] = -1
        rows = np.vstack([A, Lap, Pm, coef[dg <= 5]])
        kinds = np.array(["avg"] * n + ["lap"] * n + ["pair"] * len(pick) + ["sh"] * int((dg <= 5).sum()))
        logG = np.log(np.maximum(G_lambda(V[:, nyidx], vc, area, rows, lams, "exp", dev), 1e-300))
        pdist = np.linalg.norm(pos[ia[pick]] - pos[ib[pick]], axis=1)
        x = np.log(pdist) - np.log(pdist).mean()
        out = {}
        for l0 in (1.0, 2.0, 3.0):
            g = lambda L: np.exp(np.stack([np.interp(np.log(L), np.log(lams), row) for row in logG]))   # noqa: E731
            amps = np.array(fr["amps_median"])                                           # [nb, 3] -> l, g, 0
            Pw = amps[:, 0][None] * g(fr["lam_l_mm"])[:, None] + amps[:, 1][None] * g(fr["lam_g_mm"])[:, None] + amps[:, 2][None] * g(l0)[:, None]
            chi = pairs.band_exponent(CEN, Pw, (2.0, 20.0))
            ca, cl, cpair = chi[kinds == "avg"], chi[kinds == "lap"], chi[kinds == "pair"]
            dsel = dg[dg <= 5]
            Psh = Pw[kinds == "sh"]
            chi_deg = pairs.band_exponent(CEN, np.stack([Psh[dsel == k].sum(0) for k in range(1, 6)]), (2.0, 20.0))
            nn = np.arange(1, 6) * np.arange(2, 7)
            out[f"l0_{l0:g}mm"] = dict(P2_gap_interior=float((ca - cl)[inner].mean()), pair_slope=float((cpair - cpair.mean()) @ x / (x @ x)),
                                       P3_slope_n5=float(np.polyfit(nn, chi_deg, 1)[0]), chi_avg=float(ca.mean()))
        preds[mname] = dict(n_channels=n, n_matched=len(matched), n_interior=int(inner.sum()), **out)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "fit"])
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    subs = ["sub-10"] + ecog_subjects()
    if args.stage == "measure":
        R = pd.DataFrame([r for rr in Parallel(n_jobs=args.jobs)(delayed(measure_subject)(s) for s in subs) for r in rr])
        R.to_csv(OUT / "arrays.csv", index=False)
        V = R[R["error"].isna()] if "error" in R else R
        V = V.assign(sub=V["tag"].str.rsplit("_", n=1).str[0])
        Pt = V.groupby("sub")[[c for c in V.columns if c.startswith(("pair_slope", "mode_slope", "ref_minus"))]].median()
        G = dict(n_subjects=len(subs), n_arrays=len(V), n_patients_with_arrays=len(Pt),
                 errors=R.loc[R["error"].notna(), ["tag", "error"]].to_dict("records") if "error" in R else [])
        for c in Pt.columns:
            G[c] = stats.signflip_test(Pt[c].to_numpy(), "two-sided") | dict(median=float(Pt[c].median()))
        json.dump(G, open(OUT / "measure_group.json", "w"), indent=2, default=float)
        print(json.dumps(G, indent=1, default=float))
        return
    # ---- fit stage
    eco = ecog_subjects()
    fit_subs = {"sub-10"} | {s for i, s in enumerate(eco) if i % 2 == 0}          # odd positions (1st, 3rd, ...)
    # only arrays produced by the current measure run (files left by an interrupted run are ignored)
    valid = set(pd.read_csv(OUT / "arrays.csv").dropna(subset=["n_contacts"])["tag"])
    files = sorted(f for f in CACHE.glob("*.npz") if f.stem in valid)
    rows, fits = [], {}
    for fpath in files:
        tag = fpath.stem
        sub = tag.rsplit("_", 1)[0]
        z = dict(np.load(fpath, allow_pickle=True))
        ft = fit_array(z)
        fits[tag] = ft
        ca = [chi_of(ft["amps"][:, k]) for k in range(3)]
        rows.append(dict(tag=tag, sub=sub, half="fit" if sub in fit_subs else "test", lam_l=ft["lam_l"], lam_g=ft["lam_g"],
                         chi_local_2_40=ca[0][0], chi_global_2_40=ca[1][0], chi_incoh_2_40=ca[2][0],
                         global_share_10Hz=float(ft["amps"][np.argmin(np.abs(CEN - 10)), 1] / ft["amps"][np.argmin(np.abs(CEN - 10))].sum())))
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "two_component_fits.csv", index=False)
    Ff = F[F.half == "fit"]
    # group-median amplitude spectra (each array normalised to unit total power at 10 Hz)
    k10 = int(np.argmin(np.abs(CEN - 10)))
    An = np.stack([fits[t]["amps"] / fits[t]["amps"][k10].sum() for t in Ff.tag])
    fr = dict(lam_l_mm=float(Ff.lam_l.median()), lam_g_mm=float(Ff.lam_g.median()), amps_median=np.median(An, 0).tolist(),
              n_fit_arrays=len(Ff), fit_subjects=sorted(fit_subs & set(F["sub"])), centers=CEN.tolist(),
              chi_local_median=float(Ff.chi_local_2_40.median()), chi_global_median=float(Ff.chi_global_2_40.median()),
              chi_incoh_median=float(Ff.chi_incoh_2_40.median()), global_share_10Hz_median=float(Ff.global_share_10Hz.median()))
    # (i) test half: predicted vs measured slopes with the frozen group parameters
    test = []
    for t in F[F.half == "test"].tag:
        z = dict(np.load(CACHE / f"{t}.npz", allow_pickle=True))
        Ppred, Mpred = predict_array(z, fr["lam_l_mm"], fr["lam_g_mm"], np.array(fr["amps_median"]))
        D = np.linalg.norm(z["xyz"][:, None] - z["xyz"][None], axis=-1)
        ia, ib = np.triu_indices(len(z["xyz"]), 1)
        meas = stats_from(z["Pp"], z["Pm"], z["Pr"], z["lam"], D[ia, ib])
        pred = stats_from(Ppred, Mpred, Mpred[:1].repeat(len(z["xyz"]), 0), z["lam"], D[ia, ib])
        test.append(dict(tag=t, sub=t.rsplit("_", 1)[0], **{f"meas_{k}": meas[k] for k in ("pair_slope_2_40", "mode_slope_2_40")},
                         **{f"pred_{k}": pred[k] for k in ("pair_slope_2_40", "mode_slope_2_40")}))
    T = pd.DataFrame(test)
    T.to_csv(OUT / "test_half.csv", index=False)
    fr["scalp"] = scalp_predictions(fr)
    blob = json.dumps(fr, sort_keys=True, default=float).encode()
    fr["sha256_12"] = hashlib.sha256(blob).hexdigest()[:12]
    json.dump(fr, open(OUT / "two_component_frozen.json", "w"), indent=2, default=float)
    Tp = T.groupby("sub").median(numeric_only=True)
    res = dict(sha256_12=fr["sha256_12"], fit_summary={k: v for k, v in fr.items() if k not in ("amps_median", "centers", "scalp")},
               scalp=fr["scalp"], test_half_n_patients=len(Tp))
    for k in ("pair_slope_2_40", "mode_slope_2_40"):
        diff = (Tp[f"meas_{k}"] - Tp[f"pred_{k}"]).to_numpy()
        res[f"test_{k}"] = dict(measured_mean=float(Tp[f"meas_{k}"].mean()), predicted_mean=float(Tp[f"pred_{k}"].mean()),
                                diff=stats.signflip_test(diff, "two-sided"),
                                diff_ci95=stats.bootstrap_ci(np.random.default_rng(0).choice(diff[np.isfinite(diff)], (5000, np.isfinite(diff).sum())).mean(1)))
    json.dump(res, open(OUT / "fit_result.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
