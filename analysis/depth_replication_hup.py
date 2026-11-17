"""Independent replication of C1 in the HUP stereo-EEG cohort (OpenNeuro ds004100).

Protocol, fixed before any HUP exponent was computed:
  data      every acq-seeg task-interictal run; 500 Hz; non-overlapping 8-s epochs; an epoch is dropped if any
            analysed channel has peak-to-peak > 5x that channel's median epoch peak-to-peak
  channels  type SEEG, status good, finite coordinates; shaft = leading letters of the name, idx = trailing digits
  labels    fsaverage coordinates (m -> mm) mapped to the nearest New York Head cortex75K vertex; grey matter if
            distance <= 4 mm; lateral temporal if that vertex's Harvard-Oxford label is a superior, middle or
            inferior temporal gyrus division (temporal pole, fusiform, parahippocampal excluded)
  pairs     same shaft, both contacts lateral temporal grey matter, 3-80 mm apart
  statistic chi_D = IRASA fractal exponent of the pair difference on 40 log bands 2-40 Hz (identical to C1);
            per patient OLS slope of chi_D on ln(distance) if >= 3 pairs
  test      one-sided sign-flip across patients (H: slope > 0), alpha 0.05; also 2-20 Hz and shaft-centred slopes
Usage: python analysis/depth_replication_hup.py --labels-only     (metadata check)
       python analysis/depth_replication_hup.py --out results/hup_c1 --jobs 12
"""
import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import pairs, stats  # noqa: E402
from apex.paths import DATA, NYHEAD  # noqa: E402

HUP = Path(f"{DATA}/HUP_ds004100")
NYH = str(NYHEAD)
HSET = np.round(np.arange(1.1, 1.95, 0.05), 4)
LT_PAT = re.compile(r"(Superior|Middle|Inferior) Temporal Gyrus", re.I)


def ny_labels():
    f = h5py.File(NYH, "r")
    s = f["sa"]
    labs = ["".join(chr(x) for x in np.array(f[r]).ravel()) for r in np.array(s["HO_labels"]).ravel()]
    c = s["cortex75K"]
    vc = np.array(c["vc"]).T
    inho = np.array(c["in_HO"]).ravel().astype(int)            # 1-based index into HO_labels, 0 = none
    return vc, inho, labs


def slope(chi, ld):
    x = ld - ld.mean()
    return float((chi - chi.mean()) @ x / (x @ x)) if len(x) >= 3 and x @ x > 0 else np.nan


def process(sub, vc, inho, labs):
    import mne
    from scipy.spatial import cKDTree
    try:
        d = HUP / sub / "ses-presurgery" / "ieeg"
        el = pd.read_csv(sorted(d.glob("*acq-seeg_space-*_electrodes.tsv"))[0], sep="\t")
        el = el[pd.to_numeric(el["x"], errors="coerce").notna()].copy()
        el[["x", "y", "z"]] = el[["x", "y", "z"]].astype(float)
        # Protocol amendment: coordsystem.json declares metres but values are millimetres; units detected from magnitude.
        if np.nanmax(np.abs(el[["x", "y", "z"]].to_numpy())) < 1.0:
            el[["x", "y", "z"]] *= 1000.0
        tree = cKDTree(vc)
        dist, vi = tree.query(el[["x", "y", "z"]].to_numpy())
        lab = [labs[inho[v] - 1] if inho[v] > 0 else "none" for v in vi]
        el["grey"] = dist <= 4.0
        el["lt"] = el["grey"] & np.array([bool(LT_PAT.search(l)) for l in lab])
        m = el["name"].str.extract(r"^([A-Za-z]+)(\d+)$")
        el["shaft"], el["idx"] = m[0], pd.to_numeric(m[1], errors="coerce")
        lt = el[el["lt"] & el["shaft"].notna()]
        runs = sorted(d.glob("*task-interictal_acq-seeg_run-*_ieeg.edf"))
        segs, names = [], None
        for r in runs:
            ch = pd.read_csv(str(r).replace("_ieeg.edf", "_channels.tsv"), sep="\t")
            good = set(ch.loc[(ch["status"] == "good") & (ch["type"] == "SEEG"), "name"])
            names = [n for n in lt["name"] if n in good] if names is None else [n for n in names if n in good]
            raw = mne.io.read_raw_edf(r, preload=True, verbose="ERROR")
            X = raw.get_data(picks=names)
            fs = raw.info["sfreq"]
            L = int(8 * fs)
            segs += [X[:, i * L:(i + 1) * L] for i in range(X.shape[1] // L)]
        if not names or len(segs) < 5:
            return dict(sub=sub, n_lt=len(lt), error="no usable channels/epochs")
        S = np.stack(segs)
        ptp = np.ptp(S, axis=-1)
        keep = np.all(ptp <= 5 * np.median(ptp, axis=0, keepdims=True), axis=1)
        S = S[keep]
        sub_el = lt.set_index("name").loc[names]
        ia, ib = [], []
        for _, g in sub_el.reset_index().groupby("shaft"):
            ix = g.index.to_numpy()
            ia += [ix[p] for p in range(len(ix)) for q in range(p + 1, len(ix))]
            ib += [ix[q] for p in range(len(ix)) for q in range(p + 1, len(ix))]
        ia, ib = np.array(ia, int), np.array(ib, int)
        xyz = sub_el[["x", "y", "z"]].to_numpy()
        dd = np.linalg.norm(xyz[ia] - xyz[ib], axis=1)
        ok = (dd >= 3) & (dd <= 80)
        ia, ib, dd = ia[ok], ib[ok], dd[ok]
        if len(ia) < 3:
            return dict(sub=sub, n_lt=len(names), n_pairs=len(ia), error="fewer than 3 pairs")
        _, cen = pairs.log_bands(2.0, 40.0, 40)
        from apex import spectra
        f, frac, _ = spectra.irasa(S[:, ia] - S[:, ib], fs, hset=HSET, nperseg=int(2 * fs))
        fm = f > 0
        Dm = np.exp(np.apply_along_axis(lambda y: np.interp(np.log(cen), np.log(f[fm]), y), -1, np.log(frac[..., fm].mean(0))))
        ld = np.log(dd)
        chi = pairs.band_exponent(cen, Dm, (2.0, 40.0))
        chi20 = pairs.band_exponent(cen, Dm, (2.0, 20.0))
        shafts = sub_el["shaft"].to_numpy()[ia]
        c, x = chi.copy(), ld.copy()
        for s_ in np.unique(shafts):
            mm = shafts == s_
            c[mm] -= c[mm].mean(); x[mm] -= x[mm].mean()
        return dict(sub=sub, fs=fs, n_lt=len(names), n_pairs=len(ia), n_shafts=int(len(np.unique(shafts))), n_epochs=int(S.shape[0]),
                    slope_2_40=slope(chi, ld), slope_2_20=slope(chi20, ld), slope_shaft_centred=float(c @ x / (x @ x)) if x @ x > 0 else np.nan,
                    median_spacing_mm=float(np.median(dd[np.abs(sub_el["idx"].to_numpy()[ib] - sub_el["idx"].to_numpy()[ia]) == 1])) if np.any(np.abs(sub_el["idx"].to_numpy()[ib] - sub_el["idx"].to_numpy()[ia]) == 1) else np.nan)
    except Exception as e:
        return dict(sub=sub, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hup_c1")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args()
    vc, inho, labs = ny_labels()
    if args.labels_only:
        print(len(labs), "HO labels; lateral temporal matches:", [l for l in labs if LT_PAT.search(l)])
        print("vertex coordinate range (mm):", vc.min(0).round(1), vc.max(0).round(1))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subs = sorted(p.name for p in HUP.glob("sub-*") if list((p / "ses-presurgery" / "ieeg").glob("*task-interictal_acq-seeg*_ieeg.edf")))
    rows = Parallel(n_jobs=args.jobs)(delayed(process)(s, vc, inho, labs) for s in subs)
    D = pd.DataFrame(rows)
    D.to_csv(out / "hup_c1.csv", index=False)
    V = D[D.get("error", pd.Series(index=D.index, dtype=object)).isna()] if "error" in D else D
    G = dict(n_subjects_available=len(subs), n_analysed=len(V), errors=D.loc[D["error"].notna(), ["sub", "error"]].to_dict("records") if "error" in D else [])
    for c in ("slope_2_40", "slope_2_20", "slope_shaft_centred"):
        G[c] = stats.signflip_test(V[c].to_numpy(), "greater", n_perm=100000)
    json.dump(G, open(out / "group.json", "w"), indent=2, default=float)
    print(json.dumps(G, indent=1, default=float))


if __name__ == "__main__":
    main()
