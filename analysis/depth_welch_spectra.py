"""Welch spectra of the within-shaft depth pairs (ds004752 and HUP) and of the ds004752 scalp pairs.

Same sessions, trials, pairs and rejection rules as the IRASA analyses; used for spectral parameterisation, peak
removal and the near/far comparison. Writes <cache>/final_round/seeg/<cohort>_<participant>.npz.
Usage: python analysis/depth_welch_spectra.py --jobs 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from apex import fr, io, pairs, spectra  # noqa: E402
from apex.paths import CACHE  # noqa: E402

OUT = Path(f"{CACHE}/final_round/seeg")
ROBUST = Path(f"{CACHE}/robust")


def ds4752_one(sub):
    try:
        out = OUT / f"ds004752_{sub}.npz"
        if out.exists():
            return f"{sub} cached"
        el = io.Session(sub, "ses-01").electrodes()
        lab = {(r.shaft, int(r.idx)): r.sclass for r in el.itertuples() if r.kind == "depth"}
        sumD, sumE, n = None, None, 0
        for zf in sorted(ROBUST.glob(f"{sub}_ses-*.npz")):
            z = np.load(zf, allow_pickle=True)
            sess = io.Session(sub, str(z["ses"]))
            names = [str(x) for x in z["depth_names"]]
            sc = [str(x) for x in z["scalp_names"]]
            trials = z["trials"]
            Xi, ch_i, fs_i, ev_i = sess.load("ieeg", drop_artifacts=False, picks=names)
            Xe, ch_e, fs_e, ev_e = sess.load("eeg", drop_artifacts=False, picks=sc)
            Xi = Xi[[int(np.where(ev_i["nTrial"].to_numpy() == t)[0][0]) for t in trials]][:, [ch_i.index(c) for c in names]]
            Xe = Xe[[int(np.where(ev_e["nTrial"].to_numpy() == t)[0][0]) for t in trials]][:, [ch_e.index(c) for c in sc]]
            Xd = spectra.decimate_to(Xi, fs_i, 500.0)
            a, b = z["pair_a"], z["pair_b"]
            f, Pd = fr.welch_mixed(Xd[:, a] - Xd[:, b], 500.0, fmax=100.0)
            sa, sb = np.triu_indices(len(sc), 1)
            fe, Pe = fr.welch_mixed(Xe[:, sa] - Xe[:, sb], fs_e, fmax=45.0)
            w = len(trials)
            sumD = Pd * w if sumD is None else sumD + Pd * w
            sumE = Pe * w if sumE is None else sumE + Pe * w
            n += w
            meta = z
        xyz, cls, a, b = meta["depth_xyz"], meta["depth_class"], meta["pair_a"], meta["pair_b"]
        dist = np.linalg.norm(xyz[a] - xyz[b], axis=1)
        nb, cr = [], []
        for ia, ib in zip(a, b):
            pa, pb = io.parse_contact(str(meta["depth_names"][ia])), io.parse_contact(str(meta["depth_names"][ib]))
            lo, hi = sorted((pa["idx"], pb["idx"]))
            between = [lab.get((pa["shaft"], j), "missing") for j in range(lo + 1, hi)]
            nb.append(len(between))
            cr.append(int(any(c != "lateral_temporal" for c in between)))
        prim = (cls[a] == "lateral_temporal") & (cls[b] == "lateral_temporal") & (dist >= 3) & (dist <= 80)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, f=f, P=(sumD / n).astype(np.float32), fe=fe, Pe=(sumE / n).astype(np.float32),
                            dist=dist, prim=prim, n_between=np.array(nb), crossing=np.array(cr),
                            shaft=meta["depth_shaft"][a], scalp_names=meta["scalp_names"], scalp_pos=meta["scalp_pos"],
                            line=50.0, n_trials=n)
        return f"{sub}: {int(prim.sum())} primary pairs, {n} trials"
    except Exception as e:
        return f"{sub} ERROR {type(e).__name__}: {e}"


def hup_one(sub, vc, inho, labs):
    import mne
    from scipy.spatial import cKDTree
    import depth_replication_hup as H
    try:
        out = OUT / f"HUP_{sub}.npz"
        if out.exists():
            return f"{sub} cached"
        d = H.HUP / sub / "ses-presurgery" / "ieeg"
        el = pd.read_csv(sorted(d.glob("*acq-seeg_space-*_electrodes.tsv"))[0], sep="\t")
        el = el[pd.to_numeric(el["x"], errors="coerce").notna()].copy()
        el[["x", "y", "z"]] = el[["x", "y", "z"]].astype(float)
        if np.nanmax(np.abs(el[["x", "y", "z"]].to_numpy())) < 1.0:
            el[["x", "y", "z"]] *= 1000.0
        dd_, vi = cKDTree(vc).query(el[["x", "y", "z"]].to_numpy())
        lab = [labs[inho[v] - 1] if inho[v] > 0 else "none" for v in vi]
        el["lt"] = (dd_ <= 4.0) & np.array([bool(H.LT_PAT.search(l)) for l in lab])
        m = el["name"].str.extract(r"^([A-Za-z]+)(\d+)$")
        el["shaft"], el["idx"] = m[0], pd.to_numeric(m[1], errors="coerce")
        ltmap = {(r.shaft, int(r.idx)): bool(r.lt) for r in el.dropna(subset=["shaft", "idx"]).itertuples()}
        lt = el[el["lt"] & el["shaft"].notna()]
        runs = sorted(d.glob("*task-interictal_acq-seeg_run-*_ieeg.edf"))
        segs, names, patho, fs = [], None, set(), None
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
            return f"{sub}: no usable pairs"
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
            return f"{sub}: < 3 pairs"
        _, cen = pairs.log_bands(2.0, 40.0, 40)
        f, frac, mixed = spectra.irasa(S[:, ia] - S[:, ib], fs, hset=H.HSET, nperseg=int(2 * fs))
        fm = f > 0
        Dm = np.exp(np.apply_along_axis(lambda y: np.interp(np.log(cen), np.log(f[fm]), y), -1, np.log(frac[..., fm].mean(0))))
        chi = pairs.band_exponent(cen, Dm, (2.0, 40.0))
        keep = (f > 0) & (f <= 100)
        nb, cr, pa_ = [], [], []
        for k in range(len(ia)):
            ra, rb = sub_el.iloc[ia[k]], sub_el.iloc[ib[k]]
            lo, hi = sorted((int(ra["idx"]), int(rb["idx"])))
            between = [ltmap.get((ra["shaft"], j), False) for j in range(lo + 1, hi)]
            nb.append(len(between)); cr.append(int(not all(between)))
            pa_.append(int(ra["name"] in patho or rb["name"] in patho))
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, f=f[keep], P=mixed[..., keep].mean(0).astype(np.float32), chi_irasa=chi, dist=dd,
                            prim=np.ones(len(ia), bool), n_between=np.array(nb), crossing=np.array(cr), patho=np.array(pa_),
                            shaft=sub_el["shaft"].to_numpy()[ia], line=60.0, n_trials=S.shape[0])
        return f"{sub}: {len(ia)} pairs, {S.shape[0]} segments"
    except Exception as e:
        return f"{sub} ERROR {type(e).__name__}: {e}"


def main():
    import depth_replication_hup as H
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", default="both", choices=["both", "ds004752", "HUP"])
    args = ap.parse_args()
    lim = args.limit or None
    msgs = []
    if args.cohort in ("both", "ds004752"):
        msgs += Parallel(n_jobs=args.jobs)(delayed(ds4752_one)(s) for s in io.subjects()[:lim])
    if args.cohort in ("both", "HUP"):
        vc, inho, labs = H.ny_labels()
        subs = sorted(p.name for p in H.HUP.glob("sub-*") if list((p / "ses-presurgery" / "ieeg").glob("*task-interictal_acq-seeg*_ieeg.edf")))
        msgs += Parallel(n_jobs=args.jobs)(delayed(hup_one)(s, vc, inho, labs) for s in subs[:lim])
    print("\n".join(msgs))


if __name__ == "__main__":
    main()
