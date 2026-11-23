"""Scalp derivation analyses (Results 3.3; Figures 3, S3; Table S3).

measure (per participant x state or condition): IRASA fractal spectra (30 log centres, 2-20 Hz) and Welch spectra
(0.5-45 Hz) for every average-referenced channel, every six-neighbour Hjorth Laplacian channel and 600 random channel
pairs (seed 0).
  anphy    : W, N2, N3, R (participants with the 83-channel montage)
  ds003478 : all data, EOG-regressed, eyes open, eyes closed (segments from scalp_ds003478_eye_state_ocular.py)
  modma    : full montage and the 62 contacts nearest the ds003478 10-10 positions
analyze: P-2 (interior channels) and pair slope by IRASA, specparam fixed (2-20 Hz) and knee (1-30 Hz), five sub-bands
of 2-20 Hz, and eye-state change by derivation.
Usage: python analysis/scalp_derivations.py measure --dataset anphy --jobs 12
       python analysis/scalp_derivations.py analyze --jobs 16
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from apex import fr, hdeeg, pairs, scalp, stats  # noqa: E402
from apex.paths import CACHE, DATA  # noqa: E402

OUT = Path(f"{CACHE}/final_round/scalp")
RES = Path("results/final_round")
_, CEN = pairs.log_bands(2.0, 20.0, 30)
HSET = np.round(np.arange(1.1, 1.95, 0.05), 3)
SUB5 = np.exp(np.linspace(np.log(2.0), np.log(20.0), 6))


def compute_unit(tag, S, fs, pos, extra=None, irasa_pairs=True, irasa_channels=True):
    out = OUT / f"{tag}.npz"
    if out.exists():
        return f"{tag} cached"
    S = S - S.mean(1, keepdims=True)
    n = S.shape[1]
    nper = int(2 * fs)
    Lap = scalp.hjorth_matrix(pos, k=6)
    SL = np.einsum("oc,sct->sot", Lap, S)
    rng = np.random.default_rng(0)
    ia, ib = np.triu_indices(n, 1)
    pick = rng.choice(len(ia), size=min(600, len(ia)), replace=False)
    res = dict(pos=pos, interior=hdeeg.interior_mask(pos), pa=ia[pick], pb=ib[pick],
               dist=np.linalg.norm(pos[ia[pick]] - pos[ib[pick]], axis=1), n_seg=S.shape[0], fs=fs)
    if irasa_channels:
        res["Ia"] = pairs.irasa_channels(S, fs, CEN, HSET, nper).mean(0)
        res["Il"] = pairs.irasa_channels(SL, fs, CEN, HSET, nper).mean(0)
    else:                                                        # mixed spectra only
        res["Ia"] = np.full((n, len(CEN)), np.nan); res["Il"] = np.full((n, len(CEN)), np.nan)
    f, res["Ma"] = fr.welch_mixed(S, fs, fmax=45.0)
    _, res["Ml"] = fr.welch_mixed(SL, fs, fmax=45.0)
    Ip, Mp = [], []
    for s0 in range(0, len(pick), 100):
        pk = pick[s0:s0 + 100]
        D = S[:, ia[pk]] - S[:, ib[pk]]
        Ip.append(pairs.irasa_channels(D, fs, CEN, HSET, nper).mean(0) if irasa_pairs else np.full((len(pk), len(CEN)), np.nan))
        Mp.append(fr.welch_mixed(D, fs, fmax=45.0)[1])
    res["Ip"], res["Mp"], res["f"] = np.concatenate(Ip), np.concatenate(Mp), f
    for k in ("Ia", "Il", "Ip", "Ma", "Ml", "Mp"):
        res[k] = res[k].astype(np.float32)
    res.update(extra or {})
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{tag}.tmp.npz"                                  # atomic write
    np.savez_compressed(tmp, **res)
    tmp.rename(out)
    return f"{tag}: {n} ch, {S.shape[0]} seg"


# ---------------------------------------------------------------- datasets
def anphy_one(npz):
    try:
        z = np.load(npz, allow_pickle=True)
        if len(z["chans"]) < 20:
            return [f"{Path(npz).stem} skipped (11 channels)"]
        out = []
        for st in ("W", "N2", "N3", "R"):
            if f"{st}_eeg" in z:
                out.append(compute_unit(f"anphy_{Path(npz).stem}_{st}", z[f"{st}_eeg"].astype(float), 250.0, z["pos"]))
        return out
    except Exception as e:
        return [f"{npz} ERROR {type(e).__name__}: {e}"]


def ds3478_one(sub, files):
    import scalp_ds003478_eye_state_ocular as F
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
        out = [compute_unit(f"ds003478_{sub}_base", base, F.FS, P, irasa_pairs=False, irasa_channels=False)]
        if nreg == len(runs):
            out.append(compute_unit(f"ds003478_{sub}_eogreg", np.concatenate(reg), F.FS, P, irasa_pairs=False, irasa_channels=False))
        k = min(int((lab == "EO").sum()), int((lab == "EC").sum()))
        if k >= F.MIN_SEG:
            rng = np.random.default_rng(0)
            for c in ("EO", "EC"):
                ii = np.sort(rng.choice(np.nonzero(lab == c)[0], k, replace=False))
                out.append(compute_unit(f"ds003478_{sub}_{c}", base[ii], F.FS, P, irasa_pairs=False))
        return out
    except Exception as e:
        return [f"{sub} ERROR {type(e).__name__}: {e}"]


def egi_subset_names():
    """EGI contact names nearest (one-to-one) to the ds003478 10-10 scalp names, in MNE standard head coordinates."""
    import mne
    from scipy.optimize import linear_sum_assignment
    E = pd.read_csv(sorted(glob.glob(f"{DATA}/ds003478/sub-001/eeg/*_electrodes.tsv"))[0], sep="\t")
    want = [n for n in E.name if n.upper() not in {"CB1", "CB2", "HEOG", "VEOG"}]

    def head_pos(montage_name):
        m = mne.channels.make_standard_montage(montage_name)
        info = mne.create_info(m.ch_names, 100.0, "eeg")
        info.set_montage(m)
        return {c["ch_name"]: c["loc"][:3].copy() for c in info["chs"]}
    p10 = {k.upper(): v for k, v in head_pos("standard_1005").items()}
    pe = head_pos("GSN-HydroCel-129")
    tens = [n for n in want if n.upper() in p10]
    egi = [k for k in pe if k.startswith("E")]
    C = np.linalg.norm(np.stack([p10[n.upper()] for n in tens])[:, None] - np.stack([pe[e] for e in egi])[None], axis=-1)
    r, c = linear_sum_assignment(C)
    return {tens[i]: egi[j] for i, j in zip(r, c)}, float(np.median(C[r, c]) * 1000), len(want)


def modma_one(mat, mapping):
    import mne
    from scipy.io import loadmat
    import scalp_preprocessing_variants as A
    try:
        sid = Path(mat).name[:8]
        clean = sorted(A.MODMA_CLEAN.glob(f"{sid}*.fif"))[0]
        rc = mne.io.read_raw_fif(clean, preload=False, verbose="ERROR").pick("eeg")
        posc = np.array([c["loc"][:3] for c in rc.info["chs"]]) * 1000.0
        okc = np.linalg.norm(posc, axis=1) > 1
        M = loadmat(mat)
        key = [k for k in M if not k.startswith("__") and isinstance(M[k], np.ndarray) and M[k].ndim == 2 and M[k].shape[0] >= 128][0]
        fs = float(np.ravel(M["samplingRate"])[0])
        X = M[key][:len(posc)] * 1e-6
        names = np.array(rc.ch_names)[:X.shape[0]][okc[:X.shape[0]]]
        X, pos = X[okc[:X.shape[0]]], posc[okc][:X.shape[0]]
        Xa = X - X.mean(0, keepdims=True)
        Sr, kc = A.segment(Xa, fs)
        nk = names[kc]
        sub = np.array([n in set(mapping.values()) for n in nk])
        return [compute_unit(f"modma_{sid}_full", Sr, fs, pos[kc], irasa_pairs=False),
                compute_unit(f"modma_{sid}_sub64", Sr[:, sub], fs, pos[kc][sub], extra=dict(n_sub=int(sub.sum())), irasa_pairs=False)]
    except Exception as e:
        return [f"{mat} ERROR {type(e).__name__}: {e}"]


# ---------------------------------------------------------------- analysis
def unit_stats(path):
    z = np.load(path, allow_pickle=True)
    tag = Path(path).stem
    ds, rest = tag.split("_", 1)
    subj, cond = rest.rsplit("_", 1)
    inner = z["interior"]
    x = np.log(z["dist"])
    r = dict(dataset=ds, sub=subj, cond=cond, n_ch=len(z["pos"]), n_seg=int(z["n_seg"]))
    for b, (lo, hi) in [("all", (2.0, 20.0))] + [(f"b{k}", (SUB5[k], SUB5[k + 1])) for k in range(5)]:
        ca = pairs.band_exponent(CEN, z["Ia"], (lo, hi))
        cl = pairs.band_exponent(CEN, z["Il"], (lo, hi))
        cp = pairs.band_exponent(CEN, z["Ip"], (lo, hi))
        r[f"irasa_{b}_P2"] = float((ca - cl)[inner].mean())
        r[f"irasa_{b}_pair"] = fr.slope(cp, x)
        if b == "all":
            r["irasa_chi_avg_int"], r["irasa_chi_lap_int"] = float(ca[inner].mean()), float(cl[inner].mean())
    f = z["f"]
    for mode, rng in (("fixed", (2.0, 20.0)), ("knee", (1.0, 30.0))):
        ca = fr.sp_exponents(f, z["Ma"], mode, rng)
        cl = fr.sp_exponents(f, z["Ml"], mode, rng)
        cp = fr.sp_exponents(f, z["Mp"], mode, rng)
        okc = inner & np.isfinite(ca) & np.isfinite(cl)
        r[f"sp{mode}_P2"] = float((ca - cl)[okc].mean()) if okc.sum() >= 5 else np.nan
        r[f"sp{mode}_pair"] = fr.slope(cp, x)
        r[f"sp{mode}_chi_avg_int"], r[f"sp{mode}_chi_lap_int"] = float(np.nanmean(ca[okc])), float(np.nanmean(cl[okc]))
        r[f"sp{mode}_valid"] = float(np.mean(np.isfinite(np.concatenate([ca, cl, cp]))))
    return r


def unit_stats_cached(path):
    """Compute-only cache of unit_stats per unit; identical output."""
    c = OUT.parent / "scalp_stats" / (Path(path).stem + ".json")
    if c.exists():
        return json.load(open(c))
    r = unit_stats(path)
    c.parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(c.with_suffix(".tmp"), "w"), default=float)
    os.replace(c.with_suffix(".tmp"), c)
    return r


def summarize(R):
    J = {}
    for (ds, cond), g in R.groupby(["dataset", "cond"]):
        J[f"{ds}_{cond}"] = {c: stats.signflip_test(g[c].dropna().to_numpy(), "greater")
                             for c in g.columns if c.endswith("_P2") or c.endswith("_pair")}
        J[f"{ds}_{cond}"]["n"] = int(len(g))
    # (h) eyes open vs closed by derivation (ds003478)
    D = R[R.dataset == "ds003478"].pivot(index="sub", columns="cond")
    for est in ("irasa", "spfixed", "spknee"):
        a = (D[f"{est}_chi_avg_int"]["EO"] - D[f"{est}_chi_avg_int"]["EC"]).dropna()
        l = (D[f"{est}_chi_lap_int"]["EO"] - D[f"{est}_chi_lap_int"]["EC"]).dropna()
        J[f"EOminusEC_{est}"] = dict(avg=stats.signflip_test(a.to_numpy(), "two-sided"), lap=stats.signflip_test(l.to_numpy(), "two-sided"),
                                      avg_minus_lap=stats.signflip_test((a - l).dropna().to_numpy(), "greater"))
    # (g) MODMA paired subset - full
    Mo = R[R.dataset == "modma"].pivot(index="sub", columns="cond")
    for c in ("irasa_all_P2", "irasa_all_pair", "spfixed_P2", "spfixed_pair"):
        J[f"modma_sub64_minus_full_{c}"] = stats.signflip_test((Mo[c]["sub64"] - Mo[c]["full"]).dropna().to_numpy(), "two-sided")
        J[f"modma_sub64_{c}_less_than_0"] = stats.signflip_test(Mo[c]["sub64"].dropna().to_numpy(), "less")
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["measure", "analyze", "mapping"])
    ap.add_argument("--dataset", choices=["anphy", "ds003478", "modma"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    lim = args.limit or None
    if args.step == "mapping":
        m, med, n = egi_subset_names()
        print(f"{len(m)} of {n} names mapped; median distance {med:.1f} mm"); print(m)
        return
    if args.step == "measure":
        if args.dataset == "anphy":
            items = [p for p in sorted(glob.glob(f"{CACHE}/anphy_segments/EPCTL*.npz")) if ".tmp" not in p][:lim]
            msgs = Parallel(n_jobs=args.jobs)(delayed(anphy_one)(p) for p in items)
        elif args.dataset == "ds003478":
            by = {}
            for f in sorted(glob.glob(f"{DATA}/ds003478/sub-*/eeg/*task-Rest*_eeg.set")):
                by.setdefault(Path(f).name.split("_")[0], []).append(f)
            msgs = Parallel(n_jobs=args.jobs)(delayed(ds3478_one)(s, f) for s, f in sorted(by.items())[:lim])
        else:
            import scalp_preprocessing_variants as A
            mapping, med, n = egi_subset_names()
            print(f"MODMA subset: {len(mapping)} of {n} names, median distance {med:.1f} mm", flush=True)
            msgs = Parallel(n_jobs=args.jobs)(delayed(modma_one)(m, mapping) for m in sorted(glob.glob(str(A.MODMA_RAW / "*.mat")))[:lim])
        print("\n".join(m for mm in msgs for m in mm))
        return
    files = [p for p in sorted(glob.glob(str(OUT / "*.npz"))) if ".tmp" not in p][:lim]
    R = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(unit_stats_cached)(p) for p in files))
    RES.mkdir(parents=True, exist_ok=True)
    R.to_csv(RES / "scalp_units.csv", index=False)
    J = summarize(R)
    json.dump(J, open(RES / "scalp_group.json", "w"), indent=1, default=float)
    print("done", len(R), "units")


if __name__ == "__main__":
    main()
