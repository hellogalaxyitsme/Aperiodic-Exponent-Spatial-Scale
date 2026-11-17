"""New York Head forward model utilities: lead field, derivation gains G_e(lam) for scalp derivations.

G_e(lam) = sum_ij w_ei a_i kappa(|r_i - r_j| / lam) a_j w_ej, with w_e the derivation applied to the lead field;
computed blockwise with torch. Requires the New York Head file (sa_nyhead.mat; path set by APEX_NYHEAD).
Usage: python analysis/kernel_head_model.py --out results/stageD
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import scalp  # noqa: E402
from apex.paths import NYHEAD  # noqa: E402

NYH = str(NYHEAD)
NAME_MAP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def load_nyhead():
    f = h5py.File(NYH, "r")
    s = f["sa"]
    c = s["cortex75K"]
    V = np.array(c["V_fem_normal"]).astype(np.float64)          # [74382, 231]
    vc = np.array(c["vc"]).T                                      # [74382, 3] mm
    tri = np.array(c["tri"]).T.astype(int) - 1                    # [ntri, 3]
    in_cort = np.array(c["in_cort"]).ravel().astype(int) - 1
    labs = ["".join(chr(x) for x in np.array(f[r]).ravel()) for r in np.array(s["clab_electrodes"]).ravel()]
    locs = np.array(s["locs_3D"])[:3].T                           # [231, 3]
    # vertex areas (1/3 of adjacent triangle areas)
    p = vc[tri]
    ar = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    area = np.zeros(len(vc))
    for k in range(3):
        np.add.at(area, tri[:, k], ar / 3)
    return V[in_cort], vc[in_cort], area[in_cort], labs, locs


def derivations(labs, locs, names):
    idx = [labs.index(NAME_MAP.get(n, n)) for n in names]
    pos = locs[idx]
    n = len(names)
    rows, tags = [], []
    A = scalp.avgref_matrix(n)
    Lp = scalp.hjorth_matrix(pos)
    for i in range(n):
        rows.append(("avg", names[i], A[i]))
        rows.append(("lap", names[i], Lp[i]))
    a, b = np.triu_indices(n, 1)
    for i, j in zip(a, b):
        r = np.zeros(n)
        r[i], r[j] = 1, -1
        rows.append(("pair", f"{names[i]}-{names[j]}", r))
    M = np.stack([r[2] for r in rows])
    dist = np.array([np.linalg.norm(pos[int(np.nonzero(r[2] == 1)[0][0])] - pos[int(np.nonzero(r[2] == -1)[0][0])]) if r[0] == "pair" else np.nan
                     for r in rows])
    return idx, M, [(r[0], r[1]) for r in rows], dist


def G_lambda(Vsel, vc, area, M, lams, kernel, dev, block=2048):
    """Vsel: [N, nch] lead field restricted to montage; M: [nder, nch]. Returns G [nder, nlam]."""
    W = torch.tensor((Vsel @ M.T) * area[:, None], dtype=torch.float32, device=dev)    # [N, nder], area-weighted
    X = torch.tensor(vc, dtype=torch.float32, device=dev)
    N = X.shape[0]
    G = torch.zeros(W.shape[1], len(lams), dtype=torch.float64, device=dev)
    for s in range(0, N, block):
        D = torch.cdist(X[s:s + block], X)                                               # [b, N]
        for li, lam in enumerate(lams):
            K = torch.exp(-D / lam) if kernel == "exp" else torch.exp(-(D / lam) ** 2 / 2)
            G[:, li] += ((K @ W) * W[s:s + block]).sum(0).double()
    return G.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stageD")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V, vc, area, labs, locs = load_nyhead()
    print(f"NY Head: {V.shape[0]} cortical vertices, total area {area.sum() / 100:.0f} cm^2, median spacing ~{np.sqrt(np.median(area)):.2f} mm; device {dev}")
    lams = np.exp(np.linspace(np.log(2.0), np.log(150.0), 36))
    montages = {"full17": scalp.DS4752_SCALP, "six": ["F3", "F4", "C3", "C4", "O1", "O2"]}
    for mname, names in montages.items():
        idx, M, tags, dist = derivations(labs, locs, names)
        res = {"lams": lams, "dist": dist, "kind": np.array([t[0] for t in tags]), "label": np.array([t[1] for t in tags])}
        for kern in ("exp", "gauss"):
            G = G_lambda(V[:, idx], vc, area, M, lams, kern, dev)
            res[f"G_{kern}"] = G
            res[f"eta_{kern}"] = np.gradient(np.log(G), np.log(lams), axis=1)
            e = res[f"eta_{kern}"]
            k = res["kind"]
            print(f"{mname} {kern}: eta at lam=5,10,20,40 mm | avg mean "
                  + ", ".join(f"{np.interp(np.log(L), np.log(lams), e[k == 'avg'].mean(0)):.2f}" for L in (5, 10, 20, 40))
                  + " | lap mean " + ", ".join(f"{np.interp(np.log(L), np.log(lams), e[k == 'lap'].mean(0)):.2f}" for L in (5, 10, 20, 40))
                  + " | pairs mean " + ", ".join(f"{np.interp(np.log(L), np.log(lams), e[k == 'pair'].mean(0)):.2f}" for L in (5, 10, 20, 40)), flush=True)
        np.savez_compressed(out / f"eta_{mname}.npz", **res)


if __name__ == "__main__":
    main()
