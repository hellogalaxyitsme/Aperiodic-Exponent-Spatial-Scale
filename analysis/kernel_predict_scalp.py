"""Prediction of the ANPHY scalp contrasts from the frozen single-scale kernel (Results 3.5, S3).

Inputs: results/grid7a/frozen_params.json and the New York Head lead field; no ANPHY measurement is read. Derivations as
in the scalp analysis (interior Hjorth Laplacian vs average reference; the same 600 random pairs).
Output: results/predict_anphy/predictions.json.
Usage: python analysis/kernel_predict_scalp.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from apex import hdeeg, scalp  # noqa: E402
from apex.paths import CACHE  # noqa: E402
from kernel_head_model import G_lambda, load_nyhead  # noqa: E402

f = np.exp(np.linspace(np.log(2), np.log(20), 30))
lf = np.log(f)


def procrustes(src, dst):
    """Similarity transform (scale, rotation, translation) mapping src -> dst (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    A, Bm = src - ms, dst - md
    U, S, Vt = np.linalg.svd(A.T @ Bm)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = S.sum() / (A ** 2).sum()
    return lambda X: s * (X - ms) @ R + md


def shift(logG, lams, lam10, gam):
    lam_f = np.clip(lam10 * (f / 10) ** (-gam), lams[0], lams[-1])
    lg = np.stack([np.interp(np.log(lam_f), np.log(lams), row) for row in logG])
    x = lf - lf.mean()
    return -(lg - lg.mean(-1, keepdims=True)) @ x / (x @ x)


def main():
    out = Path("results/predict_anphy")
    out.mkdir(parents=True, exist_ok=True)
    fr = json.load(open("results/grid7a/frozen_params.json"))
    segfiles = [p for p in sorted(Path(f"{CACHE}/anphy_segments").glob("EPCTL*.npz")) if ".tmp" not in p.name]
    seg = np.load(segfiles[0], allow_pickle=True)
    chans, pos = [str(c) for c in seg["chans"]], seg["pos"]
    V, vc, area, labs, locs = load_nyhead()
    up = [l.upper() for l in labs]
    matched = [(i, up.index(c.upper())) for i, c in enumerate(chans) if c.upper() in up]
    T = procrustes(pos[[i for i, _ in matched]], locs[[j for _, j in matched]])
    P_ny = T(pos)
    nyidx = np.array([int(np.argmin(np.linalg.norm(locs - p, axis=1))) for p in P_ny])
    resid = float(np.median([np.linalg.norm(P_ny[i] - locs[j]) for i, j in matched]))
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
    Pm = np.zeros((len(pick), n))
    Pm[np.arange(len(pick)), ia[pick]] = 1
    Pm[np.arange(len(pick)), ib[pick]] = -1
    rows = np.vstack([A, Lap, Pm, coef[dg <= 5]])
    kinds = np.array(["avg"] * n + ["lap"] * n + ["pair"] * len(pick) + ["sh"] * int((dg <= 5).sum()))
    lams = np.exp(np.linspace(np.log(2.0), np.log(200.0), 40))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = G_lambda(V[:, nyidx], vc, area, rows, lams, fr["kernel"], dev)
    logG = np.log(np.maximum(G, 1e-300))
    pdist = np.linalg.norm(pos[ia[pick]] - pos[ib[pick]], axis=1)
    preds = {}
    for tag, gam in (("blur_corrected", fr["gamma"]), ("naive_gamma", fr["gamma_naive"])):
        s = shift(logG, lams, fr["lam10_source_mm"], gam)
        sa, sl, sp = s[kinds == "avg"], s[kinds == "lap"], s[kinds == "pair"]
        x = np.log(pdist) - np.log(pdist).mean()
        # degree power: sum of G over coefficients of each degree, then shift
        dsel = dg[dg <= 5]
        Gdeg = np.stack([G[kinds == "sh"][dsel == k].sum(0) for k in range(1, 6)])
        sdeg = shift(np.log(Gdeg), lams, fr["lam10_source_mm"], gam)
        nn = np.arange(1, 6) * np.arange(2, 7)
        preds[tag] = dict(P2_gap_interior=float((sa - sl)[inner].mean()), P2_gap_all=float((sa - sl).mean()),
                          pair_slope=float((sp - sp.mean()) @ x / (x @ x)), P3_slope_n5=float(np.polyfit(nn, sdeg, 1)[0]),
                          scalp_avg_bias=float(sa.mean()), gamma=float(gam))
    res = dict(frozen_params=fr, n_channels=n, n_matched_by_name=len(matched), procrustes_median_residual_mm=resid,
               n_interior=int(inner.sum()), predictions=preds)
    blob = json.dumps(res, sort_keys=True, default=float).encode()
    res["sha256_12"] = hashlib.sha256(blob).hexdigest()[:12]
    json.dump(res, open(out / "predictions.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
