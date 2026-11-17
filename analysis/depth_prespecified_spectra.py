"""Per-session spectra for the prespecified ds004752 analysis (Methods 2.3-2.4; Table 2).

On the same artefact-free trials: IRASA fractal spectra of every within-shaft grey-matter bipolar pair and of every scalp
channel pair, and scalp spectra under average reference and Laplacian. All parameters are read from the configuration
file (configs/stageB_dev.yaml for participants 1-5; configs/stageBC_confirm.yaml, frozen, for participants 6-15).
Usage: python analysis/depth_prespecified_spectra.py --config configs/stageBC_confirm.yaml --jobs 12
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import io, pairs, scalp, spectra  # noqa: E402
from apex.paths import CACHE  # noqa: E402


def process(sess, cfg, out, force):
    if out.exists() and not force:
        return f"{sess.sub} {sess.ses}: cached"
    t0 = time.time()
    sp = cfg["spectra"]
    a, b, s = sp["irasa_h"]
    hset = np.round(np.arange(a, b + s / 2, s), 4)
    edges, centers = pairs.log_bands(*cfg["allpairs"]["band_range"], cfg["allpairs"]["n_bands"])
    el = sess.electrodes()
    grey = el[(el["kind"] == "depth") & (el["sclass"] != io.WHITE)].sort_values(["shaft", "idx"])
    names = grey["name"].tolist()
    import pandas as pd
    avail = set(pd.read_csv(sess.path("eeg", "channels.tsv"), sep="\t")["name"])
    sc_names = [c for c in scalp.DS4752_SCALP if c in avail]        # Fp1/Fp2 (ocular), A1/A2, T1/T2 excluded everywhere
    Xi, ch_i, fs_i, ev_i = sess.load("ieeg", drop_artifacts=False, picks=names)
    Xe, ch_e, fs_e, ev_e = sess.load("eeg", drop_artifacts=False, picks=sc_names)
    # common artifact-free trials (both modalities)
    good_i = set(ev_i.loc[ev_i["Artifact"] == 0, "nTrial"]) if "Artifact" in ev_i else set(ev_i["nTrial"])
    good_e = set(ev_e.loc[ev_e["Artifact"] == 0, "nTrial"]) if "Artifact" in ev_e else set(ev_e["nTrial"])
    keep = sorted(good_i & good_e & set(ev_i["nTrial"]) & set(ev_e["nTrial"]))
    ii = [int(np.where(ev_i["nTrial"].to_numpy() == t)[0][0]) for t in keep]
    ie = [int(np.where(ev_e["nTrial"].to_numpy() == t)[0][0]) for t in keep]
    Xi = Xi[ii][:, [ch_i.index(n) for n in names]]
    Xe = Xe[ie][:, [ch_e.index(n) for n in sc_names]]
    # depth: decimate to irasa_fs, all pairs
    fsd = sp["irasa_fs"]
    Xd = spectra.decimate_to(Xi, fs_i, fsd)
    nper = int(sp["irasa_nperseg_s"] * fsd)
    Pd_mt, Dd_mt = pairs.mt_allpairs(Xd, fsd, edges, sp["multitaper_NW"])
    Dd_ir = pairs.irasa_allpairs(Xd, fsd, centers, hset=hset, nperseg=nper)
    # scalp: all pairs + derivations (vertex-referenced as recorded; avg-ref and Hjorth rows sum to zero)
    pos = scalp.std1020_positions(sc_names)
    nperse = int(sp["irasa_nperseg_s"] * fs_e)
    Pe_mt, De_mt = pairs.mt_allpairs(Xe, fs_e, edges, sp["multitaper_NW"])
    De_ir = pairs.irasa_allpairs(Xe, fs_e, centers, hset=hset, nperseg=nperse)
    der = {}
    for name, M in (("vertex", np.eye(len(pos))), ("avg", scalp.avgref_matrix(len(pos))), ("lap", scalp.hjorth_matrix(pos))):
        Y = np.einsum("oc,tcn->ton", M, Xe)
        der[f"scalp_{name}_ir"] = pairs.irasa_channels(Y, fs_e, centers, hset=hset, nperseg=nperse).astype(np.float32)
        der[f"scalp_{name}_mt"] = pairs.mt_allpairs(Y, fs_e, edges, sp["multitaper_NW"])[0].astype(np.float32)
    # high-frequency scalp index (EMG / floor): mean PSD 55-95 Hz relative to 35-40 Hz, avg reference
    f, P = spectra.multitaper_psd(np.einsum("oc,tcn->ton", scalp.avgref_matrix(len(pos)), Xe), fs_e, NW=sp["multitaper_NW"])
    hf = P[..., (f >= 55) & (f <= 95) & (np.abs(f - 50) > 3)].mean(-1) / P[..., (f >= 35) & (f <= 40)].mean(-1)
    np.savez_compressed(
        out, sub=sess.sub, ses=sess.ses, trials=np.array(keep), centers=centers, edges=edges,
        depth_names=np.array(names), depth_xyz=grey[["x", "y", "z"]].to_numpy(float), depth_class=grey["sclass"].to_numpy(),
        depth_shaft=grey["shaft"].to_numpy(), depth_idx=grey["idx"].to_numpy(int), depth_gyrus=grey["gyrus"].to_numpy(),
        Pd_mt=Pd_mt.astype(np.float32), Dd_mt=Dd_mt.astype(np.float32), Dd_ir=Dd_ir.astype(np.float32),
        scalp_names=np.array(sc_names), scalp_pos=pos, Pe_mt=Pe_mt.astype(np.float32), De_mt=De_mt.astype(np.float32),
        De_ir=De_ir.astype(np.float32), scalp_hf_ratio=hf.astype(np.float32), **der)
    return f"{sess.sub} {sess.ses}: {len(names)} depth ({len(names) * (len(names) - 1) // 2} pairs), {len(keep)} trials, {time.time() - t0:.0f} s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default=f"{CACHE}/allpairs")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--subjects", nargs="*", help="override subject list (cache only; analysis split is enforced downstream)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    subs = args.subjects or cfg["subjects"]
    tasks = [(s, cache / f"{s.sub}_{s.ses}.npz") for sub in subs for s in io.sessions(sub)]
    print("\n".join(Parallel(n_jobs=args.jobs)(delayed(process)(s, cfg, o, args.force) for s, o in tasks)))
