"""Grid robustness (Results 3.2; Figure 2; Table S2).

measure: per-contact white-noise floor and Welch spectra of pairs, graph modes and referential contacts.
analyze: reproduction from the IRASA cache, noise-floor subtraction, eight sub-bands of 2-40 Hz, specparam fixed
(2-40 Hz) and knee (1-40 Hz).
Usage: python analysis/grid_robustness.py measure --jobs 8
       python analysis/grid_robustness.py analyze --jobs 16
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
from apex import fr, io, pairs, spectra, stats  # noqa: E402
from apex.paths import CACHE  # noqa: E402
import grid_modes_and_mixture_model as G  # noqa: E402

OUT = Path(f"{CACHE}/final_round/grid")
RES = Path("results/final_round")
SUB_EDGES = np.exp(np.linspace(np.log(2.0), np.log(40.0), 9))


def floor_of(S, fs):
    band = (600.0, 900.0) if fs >= 2000 else (150.0, 200.0)
    f, P = fr.welch_mixed(S, fs, fmax=fs / 2)
    m = (f >= band[0]) & (f <= band[1])
    return np.median(P[:, m], axis=-1)


def store(tag, S, fs, xyz, names, floor):
    lam, U, D, sig = G.graph_modes(xyz)
    ia, ib = np.triu_indices(len(names), 1)
    f, Pp = fr.welch_mixed(S[:, ia] - S[:, ib], fs, fmax=100.0)
    _, Pm = fr.welch_mixed(np.einsum("ck,ect->ekt", U, S), fs, fmax=100.0)
    _, Pr = fr.welch_mixed(S, fs, fmax=100.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{tag}.npz", f=f, Pp=Pp.astype(np.float32), Pm=Pm.astype(np.float32), Pr=Pr.astype(np.float32),
                        floor=floor, names=np.array(names), n_ep=S.shape[0], fs=fs)
    return tag


def measure_subject(sub):
    try:
        if sub == "sub-10":
            Sall, Snat, fsn = [], [], None
            for s in io.sessions("sub-10"):
                el = s.electrodes()
                g = el[el["kind"] == "grid"].sort_values(["row", "idx"])
                names = g["name"].tolist()
                xyz = g[["x", "y", "z"]].to_numpy(float)
                X, ch, f0, ev = s.load("ieeg", drop_artifacts=True, picks=names)
                X = X[:, [ch.index(n) for n in names]]
                Snat.append(X); fsn = f0
                Sall.append(spectra.decimate_to(X, f0, 500.0))
            S = np.concatenate(Sall, 0)[:G.MAX_EPOCHS]
            fl = floor_of(np.concatenate(Snat, 0)[:G.MAX_EPOCHS], fsn)     # PSD (V^2/Hz) of white noise is rate-independent
            return [store("sub-10_grid", S, 500.0, xyz, names, fl)]
        out = []
        for tag, S, fs, xyz, names in G.hup_arrays(sub):
            out.append(store(tag, S, fs, xyz, names, floor_of(S, fs)))
        return out
    except Exception as e:
        return [f"{sub} ERROR {type(e).__name__}: {e}"]


# ---------------------------------------------------------------- analysis
def mode_bins(lam):
    e = np.exp(np.linspace(np.log(lam.min()), np.log(lam.max()) + 1e-9, 7))
    bins = np.clip(np.searchsorted(e, lam, side="right") - 1, 0, 5)
    use = [k for k in range(6) if np.any(bins == k)]
    return bins, use, np.log(np.sqrt(np.array([lam[bins == k].mean() for k in use])))


def stats_irasa(Pp, Pm, lam, d, band=(2.0, 40.0), cen=None):
    ok = (d >= 5) & (d <= 80)
    chi_p = pairs.band_exponent(cen, Pp, band)
    good = ok & np.isfinite(chi_p)
    bins, use, x = mode_bins(lam)
    Mb = np.stack([np.nansum(Pm[bins == k], 0) for k in use])
    chi_b = pairs.band_exponent(cen, Mb, band)
    return fr.slope(chi_p[good], np.log(d[good])), fr.slope(chi_b, x)


def analyze_one(tag):
    zi = np.load(G.CACHE / f"{tag}.npz", allow_pickle=True)
    zm = np.load(OUT / f"{tag}.npz", allow_pickle=True)
    assert list(zi["names"]) == list(zm["names"]), tag
    lam, U, xyz = zi["lam"], zi["U"], zi["xyz"]
    ia, ib = np.triu_indices(len(xyz), 1)
    d = np.linalg.norm(xyz[ia] - xyz[ib], axis=1)
    cen = G.CEN
    r = dict(tag=tag)
    r["irasa_pair"], r["irasa_mode"] = stats_irasa(zi["Pp"], zi["Pm"], lam, d, cen=cen)
    # floor subtraction (units: V^2/Hz for both)
    fl = zm["floor"]
    Pp = zi["Pp"] - (fl[ia] + fl[ib])[:, None]
    Pm = zi["Pm"] - ((U ** 2).T @ fl)[:, None]
    r["floor_frac_pairs_excluded"] = float(np.mean(np.any(Pp <= 0, axis=1)))
    Pp[np.any(Pp <= 0, axis=1)] = np.nan
    Pm[np.any(Pm <= 0, axis=1)] = np.nan
    r["floor_frac_modes_excluded"] = float(np.mean(np.isnan(Pm[:, 0])))
    r["floor_rel_10hz_median"] = float(np.median(((fl[ia] + fl[ib])[:, None] / zi["Pp"])[:, np.argmin(np.abs(cen - 10))]))
    r["floor_pair"], r["floor_mode"] = stats_irasa(Pp, Pm, lam, d, cen=cen)
    # band-resolved (IRASA)
    for k, (lo, hi) in enumerate(zip(SUB_EDGES[:-1], SUB_EDGES[1:])):
        r[f"band{k}_pair"], r[f"band{k}_mode"] = stats_irasa(zi["Pp"], zi["Pm"], lam, d, band=(lo, hi), cen=cen)
    # specparam on mixed spectra
    f = zm["f"]
    bins, use, x = mode_bins(lam)
    ok = (d >= 5) & (d <= 80)
    for mode, rng in (("fixed", (2.0, 40.0)), ("knee", (1.0, 40.0))):
        Pp_m = fr.interp_line(f, zm["Pp"], 60.0)
        Pm_m = fr.interp_line(f, zm["Pm"], 60.0)
        chi_p = fr.sp_exponents(f, Pp_m[ok], mode, rng)
        chi_b = fr.sp_exponents(f, np.stack([Pm_m[bins == k].sum(0) for k in use]), mode, rng)
        r[f"sp{mode}_pair"] = fr.slope(chi_p, np.log(d[ok]))
        r[f"sp{mode}_mode"] = fr.slope(chi_b, x)
        r[f"sp{mode}_pair_valid"] = float(np.isfinite(chi_p).mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["measure", "analyze"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.step == "measure":
        tags = set(pd.read_csv(G.OUT / "arrays.csv").dropna(subset=["n_contacts"]).tag)
        subs = sorted({t.split("_")[0] for t in tags})[: args.limit or None]
        msgs = Parallel(n_jobs=args.jobs)(delayed(measure_subject)(s) for s in subs)
        print("\n".join(m for mm in msgs for m in mm))
        return
    tags = sorted(pd.read_csv(G.OUT / "arrays.csv").dropna(subset=["n_contacts"]).tag)
    tags = [t for t in tags if (OUT / f"{t}.npz").exists()][: args.limit or None]
    R = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(analyze_one)(t) for t in tags))
    RES.mkdir(parents=True, exist_ok=True)
    R.to_csv(RES / "grid_robustness.csv", index=False)
    J = {"n_arrays": len(R)}
    for c in [c for c in R.columns if c.endswith("_pair") or c.endswith("_mode")]:
        J[c] = stats.signflip_test(R[c].dropna().to_numpy(), "greater" if c.endswith("_pair") else "less")
    for c in ("floor_frac_pairs_excluded", "floor_frac_modes_excluded", "floor_rel_10hz_median", "spfixed_pair_valid", "spknee_pair_valid"):
        J[c] = float(R[c].median())
    json.dump(J, open(RES / "grid_robustness.json", "w"), indent=1, default=float)
    print(json.dumps(J, indent=1, default=float))


if __name__ == "__main__":
    main()
