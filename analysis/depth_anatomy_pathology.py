"""Anatomical and pathology controls for C1 (Results 3.1, Table S1).

Contiguous pair: every contact between the two contacts on the shaft is lateral temporal grey matter. HUP pathology
control: pairs where neither contact is labelled seizure-onset or resected in any run.
Usage: python analysis/depth_anatomy_pathology.py --out results/straddle --jobs 12
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
sys.path.insert(0, str(ROOT / "analysis"))
from apex import io, pairs, spectra, stats  # noqa: E402
from depth_robustness_tests import DEV, load_subject  # noqa: E402
import depth_replication_hup as H  # noqa: E402

B40, B20 = (2.0, 40.0), (2.0, 20.0)


def slope(chi, ld):
    x = ld - ld.mean()
    return float((chi - chi.mean()) @ x / (x @ x)) if len(x) >= 3 and (x @ x) > 0 else np.nan


def ds4752_pairs():
    rows = []
    for sub in io.subjects():
        d = load_subject(sub)
        el = io.Session(sub, "ses-01").electrodes()
        lab = {(r.shaft, int(r.idx)): r.sclass for r in el.itertuples() if r.kind == "depth"}
        names, xyz, cls = d["depth_names"], d["depth_xyz"], d["depth_class"]
        a, b = d["pair_a"], d["pair_b"]
        dist = np.linalg.norm(xyz[a] - xyz[b], axis=1)
        prim = (cls[a] == "lateral_temporal") & (cls[b] == "lateral_temporal") & (dist >= 3) & (dist <= 80)
        Dm = d["Dd_orig"].mean(0)[prim]
        chi = pairs.band_exponent(d["centers_depth"], Dm, B40)
        for k, (ia, ib) in enumerate(zip(a[prim], b[prim])):
            pa, pb = io.parse_contact(str(names[ia])), io.parse_contact(str(names[ib]))
            lo, hi = sorted((pa["idx"], pb["idx"]))
            between = [lab.get((pa["shaft"], j), "missing") for j in range(lo + 1, hi)]
            rows.append(dict(cohort="ds004752", sub=sub, split="dev" if sub in DEV else "confirm", shaft=pa["shaft"],
                             ld=float(np.log(dist[prim][k])), chi=float(chi[k]), n_between=len(between),
                             crossing=int(any(c != "lateral_temporal" for c in between)), patho=0))
    return pd.DataFrame(rows)


def hup_one(sub, vc, inho, labs):
    import mne
    from scipy.spatial import cKDTree
    try:
        d = H.HUP / sub / "ses-presurgery" / "ieeg"
        el = pd.read_csv(sorted(d.glob("*acq-seeg_space-*_electrodes.tsv"))[0], sep="\t")
        el = el[pd.to_numeric(el["x"], errors="coerce").notna()].copy()
        el[["x", "y", "z"]] = el[["x", "y", "z"]].astype(float)
        if np.nanmax(np.abs(el[["x", "y", "z"]].to_numpy())) < 1.0:
            el[["x", "y", "z"]] *= 1000.0
        dist, vi = cKDTree(vc).query(el[["x", "y", "z"]].to_numpy())
        lab = [labs[inho[v] - 1] if inho[v] > 0 else "none" for v in vi]
        el["lt"] = (dist <= 4.0) & np.array([bool(H.LT_PAT.search(l)) for l in lab])
        m = el["name"].str.extract(r"^([A-Za-z]+)(\d+)$")
        el["shaft"], el["idx"] = m[0], pd.to_numeric(m[1], errors="coerce")
        ltmap = {(r.shaft, int(r.idx)): bool(r.lt) for r in el.dropna(subset=["shaft", "idx"]).itertuples()}
        lt = el[el["lt"] & el["shaft"].notna()]
        runs = sorted(d.glob("*task-interictal_acq-seeg_run-*_ieeg.edf"))
        segs, names, patho = [], None, set()
        for r in runs:
            ch = pd.read_csv(str(r).replace("_ieeg.edf", "_channels.tsv"), sep="\t")
            sd = ch["status_description"].fillna("").astype(str).str.lower()
            patho |= set(ch.loc[sd.str.contains("soz") | sd.str.contains("resect"), "name"])
            good = set(ch.loc[(ch["status"] == "good") & (ch["type"] == "SEEG"), "name"])
            names = [n for n in lt["name"] if n in good] if names is None else [n for n in names if n in good]
            raw = mne.io.read_raw_edf(r, preload=True, verbose="ERROR")
            X = raw.get_data(picks=names)
            fs = raw.info["sfreq"]
            L = int(8 * fs)
            segs += [X[:, i * L:(i + 1) * L] for i in range(X.shape[1] // L)]
        if not names or len(segs) < 5:
            return []
        S = np.stack(segs)
        ptp = np.ptp(S, axis=-1)
        S = S[np.all(ptp <= 5 * np.median(ptp, axis=0, keepdims=True), axis=1)]
        sub_el = lt.set_index("name").loc[names].reset_index()
        ia, ib = [], []
        for _, g in sub_el.groupby("shaft"):
            ix = g.index.to_numpy()
            ia += [ix[p] for p in range(len(ix)) for q in range(p + 1, len(ix))]
            ib += [ix[q] for p in range(len(ix)) for q in range(p + 1, len(ix))]
        ia, ib = np.array(ia, int), np.array(ib, int)
        xyz = sub_el[["x", "y", "z"]].to_numpy()
        dd = np.linalg.norm(xyz[ia] - xyz[ib], axis=1)
        ok = (dd >= 3) & (dd <= 80)
        ia, ib, dd = ia[ok], ib[ok], dd[ok]
        if len(ia) < 3:
            return []
        _, cen = pairs.log_bands(2.0, 40.0, 40)
        f, frac, _ = spectra.irasa(S[:, ia] - S[:, ib], fs, hset=H.HSET, nperseg=int(2 * fs))
        fm = f > 0
        Dm = np.exp(np.apply_along_axis(lambda y: np.interp(np.log(cen), np.log(f[fm]), y), -1, np.log(frac[..., fm].mean(0))))
        chi = pairs.band_exponent(cen, Dm, B40)
        out = []
        for k in range(len(ia)):
            ra, rb = sub_el.iloc[ia[k]], sub_el.iloc[ib[k]]
            lo, hi = sorted((int(ra["idx"]), int(rb["idx"])))
            between = [ltmap.get((ra["shaft"], j), False) for j in range(lo + 1, hi)]
            out.append(dict(cohort="HUP", sub=sub, split="hup", shaft=ra["shaft"], ld=float(np.log(dd[k])), chi=float(chi[k]),
                            n_between=len(between), crossing=int(not all(between)),
                            patho=int(ra["name"] in patho or rb["name"] in patho)))
        return out
    except Exception as e:
        print(sub, "ERROR", type(e).__name__, e, flush=True)
        return []


def per_patient(P, mask, col="chi"):
    v = []
    for _, g in P[mask].groupby("sub"):
        v.append(slope(g[col].to_numpy(), g["ld"].to_numpy()))
    v = np.array([x for x in v if np.isfinite(x)])
    return stats.signflip_test(v, "greater") if len(v) >= 2 else dict(n=len(v))


def mixed(P):
    import statsmodels.formula.api as smf
    try:
        fit = smf.mixedlm("chi ~ ld + crossing", P, groups=P["sub"], re_formula="~ld").fit(method="lbfgs", reml=True)
        return {k: dict(coef=float(fit.params[k]), se=float(fit.bse[k]), p_two_sided=float(fit.pvalues[k])) for k in ("ld", "crossing")} | dict(n_pairs=len(P), converged=bool(fit.converged))
    except Exception as e:
        return dict(error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/straddle")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    A = ds4752_pairs()
    vc, inho, labs = H.ny_labels()
    subs = sorted(p.name for p in H.HUP.glob("sub-*") if list((p / "ses-presurgery" / "ieeg").glob("*task-interictal_acq-seeg*_ieeg.edf")))
    B = pd.DataFrame([r for rr in Parallel(n_jobs=args.jobs)(delayed(hup_one)(s, vc, inho, labs) for s in subs) for r in rr])
    P = pd.concat([A, B], ignore_index=True)
    P.to_csv(out / "straddle_pairs.csv", index=False)
    G = {}
    for name, S in (("ds4752_confirm", P[P.split == "confirm"]), ("ds4752_all", P[P.cohort == "ds004752"]), ("HUP", P[P.cohort == "HUP"])):
        G[name] = dict(n_pairs=len(S), frac_crossing=float(S.crossing.mean()),
                       all_pairs=per_patient(S, np.ones(len(S), bool)),
                       contiguous_only=per_patient(S, S.crossing.to_numpy() == 0),
                       crossing_only=per_patient(S, S.crossing.to_numpy() == 1),
                       mixed_ld_plus_crossing=mixed(S))
    H_ = P[P.cohort == "HUP"]
    G["HUP_pathology"] = dict(frac_pairs_patho=float(H_.patho.mean()),
                              non_patho_pairs=per_patient(H_, H_.patho.to_numpy() == 0),
                              non_patho_contiguous=per_patient(H_, (H_.patho.to_numpy() == 0) & (H_.crossing.to_numpy() == 0)))
    json.dump(G, open(out / "group.json", "w"), indent=2, default=float)
    print(json.dumps(G, indent=1, default=float))


if __name__ == "__main__":
    main()
