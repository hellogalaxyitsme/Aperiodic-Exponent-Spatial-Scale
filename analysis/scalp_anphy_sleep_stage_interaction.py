"""Sleep stage x derivation interaction in ANPHY.

For each participant and stage S (N2, N3, REM): stage effect under average reference and under Laplacian
(exponent in S minus exponent in wakefulness, interior channels) and their difference, which equals the change in the
derivation contrast between S and wakefulness. IRASA (tested) and specparam fixed mode (descriptive); two-sided
sign-flip tests and bootstrap 95% CIs. Reads the scalp units written by scalp_derivations.py (analyze).
Usage: python analysis/scalp_anphy_sleep_stage_interaction.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import stats  # noqa: E402

OUT = ROOT / "results" / "revision"


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(n, x.size))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    R = pd.read_csv(ROOT / "results" / "final_round" / "scalp_units.csv")
    A = R[R.dataset == "anphy"].pivot(index="sub", columns="cond")
    J, rows = {}, []
    for est in ("irasa", "spfixed"):
        for S in ("N2", "N3", "R"):
            a = (A[f"{est}_chi_avg_int"][S] - A[f"{est}_chi_avg_int"]["W"])
            l = (A[f"{est}_chi_lap_int"][S] - A[f"{est}_chi_lap_int"]["W"])
            ok = a.notna() & l.notna()
            a, l = a[ok], l[ok]
            entry = {}
            for name, v in (("avg", a), ("lap", l), ("avg_minus_lap", a - l)):
                t = stats.signflip_test(v.to_numpy(), "two-sided")
                t["ci95"] = boot(v.to_numpy())
                entry[name] = t
            J[f"{est}_{S}minusW"] = entry
            for sub in a.index:
                rows.append(dict(estimator=est, stage=S, sub=sub, avg=a[sub], lap=l[sub], interaction=a[sub] - l[sub]))
    (OUT / "sleep_derivation.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "sleep_derivation_units.csv", index=False)
    for k, e in J.items():
        print(k, {n: (round(v["mean"], 3), [round(c, 3) for c in v["ci95"]], v["n_positive"], v["n"], v["p"]) for n, v in e.items()})


if __name__ == "__main__":
    main()
