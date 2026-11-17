"""Spectra for the depth robustness analyses (ds004752; Results 3.1, Table S1).

Within-shaft grey-matter depth pairs and scalp pairs in three variants: as analysed (8-s trials), evoked response
removed, and stimulus-free maintenance window (3-6 s). Depth bands 2-80 Hz (48 log bands), scalp 2-40 Hz; per-pair
white-noise floor from 600-900 Hz.
Usage: python analysis/depth_robustness_spectra.py --jobs 20
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import io, pairs, scalp, spectra  # noqa: E402
from apex.paths import CACHE  # noqa: E402

CACHE = Path(f"{CACHE}/robust")
HSET = np.round(np.arange(1.1, 1.95, 0.05), 4)


def irasa_pairs(X, fs, ia, ib, centers, nper):
    out = np.empty((X.shape[0], len(ia), len(centers)), np.float32)
    for s in range(0, len(ia), 48):
        sl = slice(s, s + 48)
        f, frac, _ = spectra.irasa(X[:, ia[sl]] - X[:, ib[sl]], fs, hset=HSET, nperseg=nper)
        m = f > 0
        lp = np.log(np.maximum(frac[..., m], 1e-300))
        out[:, sl] = np.exp(np.apply_along_axis(lambda y: np.interp(np.log(centers), np.log(f[m]), y), -1, lp))
    return out


def process(sess, force=False):
    out = CACHE / f"{sess.sub}_{sess.ses}.npz"
    if out.exists() and not force:
        return f"{sess.sub} {sess.ses}: cached"
    t0 = time.time()
    el = sess.electrodes()
    grey = el[(el["kind"] == "depth") & (el["sclass"] != io.WHITE)].sort_values(["shaft", "idx"]).reset_index(drop=True)
    names = grey["name"].tolist()
    ia, ib = [], []
    for _, g in grey.groupby("shaft"):
        ix = g.index.to_numpy()
        for p in range(len(ix)):
            for q in range(p + 1, len(ix)):
                ia.append(ix[p]); ib.append(ix[q])
    ia, ib = np.array(ia, int), np.array(ib, int)
    avail = set(pd.read_csv(sess.path("eeg", "channels.tsv"), sep="\t")["name"])
    sc = [c for c in scalp.DS4752_SCALP if c in avail]
    Xi, ch_i, fs_i, ev_i = sess.load("ieeg", drop_artifacts=False, picks=names)
    Xe, ch_e, fs_e, ev_e = sess.load("eeg", drop_artifacts=False, picks=sc)
    gi = set(ev_i.loc[ev_i["Artifact"] == 0, "nTrial"]); ge = set(ev_e.loc[ev_e["Artifact"] == 0, "nTrial"])
    keep = sorted(gi & ge & set(ev_i["nTrial"]) & set(ev_e["nTrial"]))
    Xi = Xi[[int(np.where(ev_i["nTrial"].to_numpy() == t)[0][0]) for t in keep]][:, [ch_i.index(n) for n in names]]
    Xe = Xe[[int(np.where(ev_e["nTrial"].to_numpy() == t)[0][0]) for t in keep]][:, [ch_e.index(n) for n in sc]]
    # floor per depth pair (full rate, orig)
    floor = np.empty((len(keep), len(ia)), np.float32)
    for k in range(len(ia)):
        f, P = spectra.multitaper_psd(Xi[:, ia[k]] - Xi[:, ib[k]], fs_i, NW=4.0)
        floor[:, k] = spectra.noise_floor(f, P, (600.0, 900.0))
    Xd = spectra.decimate_to(Xi, fs_i, 500.0)
    _, cd = pairs.log_bands(2.0, 80.0, 48)
    _, cs = pairs.log_bands(2.0, 40.0, 40)
    sa, sb = np.triu_indices(len(sc), 1)
    res = dict(sub=sess.sub, ses=sess.ses, trials=np.array(keep), depth_names=np.array(names), depth_xyz=grey[["x", "y", "z"]].to_numpy(float),
               depth_class=grey["sclass"].to_numpy(), depth_shaft=grey["shaft"].to_numpy(), depth_idx=grey["idx"].to_numpy(int),
               pair_a=ia, pair_b=ib, centers_depth=cd, centers_scalp=cs, floor=floor, scalp_names=np.array(sc),
               scalp_pos=scalp.std1020_positions(sc))
    for var in ("orig", "evk", "maint"):
        D, E = Xd, Xe
        if var != "orig":
            D = Xd - Xd.mean(0, keepdims=True)
            E = Xe - Xe.mean(0, keepdims=True)
        if var == "maint":
            D = D[..., int(3 * 500):int(6 * 500)]
            E = E[..., int(3 * fs_e):int(6 * fs_e)]
        nper_d = int((1.5 if var == "maint" else 2.0) * 500)
        nper_e = int((1.5 if var == "maint" else 2.0) * fs_e)
        res[f"Dd_{var}"] = irasa_pairs(D, 500.0, ia, ib, cd, nper_d)
        res[f"De_{var}"] = irasa_pairs(E, fs_e, sa, sb, cs, nper_e)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    return f"{sess.sub} {sess.ses}: {len(ia)} depth pairs, {len(sa)} scalp pairs, {len(keep)} trials, {time.time() - t0:.0f} s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--subjects", nargs="*")
    args = ap.parse_args()
    subs = args.subjects or io.subjects()
    tasks = [s for sub in subs for s in io.sessions(sub)]
    print("\n".join(Parallel(n_jobs=args.jobs)(delayed(process)(s) for s in tasks)))
