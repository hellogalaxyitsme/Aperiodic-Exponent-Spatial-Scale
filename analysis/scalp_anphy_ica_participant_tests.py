"""ANPHY ICA sensitivity tests with the participant as the unit (S4; Data S1).

For each participant, the ICA minus raw difference (interior P-2, pair slope, near-neighbour slope) is averaged over
that participant's available states; the 26 participant values are tested with a two-sided sign-flip test
(B = 100,000) and summarised with a percentile bootstrap 95% CI. These tests replace the per-state ICA comparisons in
scalp_anphy_segments_ica.py, which treated the states of one participant as independent.
Reads results/anphy_v2/anphy_v2.csv; writes results/revision/anphy_ica_participant.json.
Usage: python analysis/scalp_anphy_ica_participant_tests.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex import stats  # noqa: E402

MEASURES = ("P2_gap_interior", "pair_slope", "P3_slope_n5")


def bootstrap_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    D = pd.read_csv(ROOT / "results/anphy_v2/anphy_v2.csv")
    if "error" in D:
        D = D[D["error"].isna()]
    piv = D.pivot_table(index=["subject", "state"], columns="variant", values=list(MEASURES))
    out = {"unit": "participant (mean of ICA minus raw over available states)"}
    for c in MEASURES:
        d = (piv[c]["ica"] - piv[c]["raw"]).dropna()
        per_subject = d.groupby(level="subject").mean()
        x = per_subject.to_numpy()
        res = stats.signflip_test(x, "two-sided", n_perm=100000)
        res["ci95"] = bootstrap_ci(x)
        res["n_states_total"] = int(len(d))
        res["per_state_mean"] = {s: float(v) for s, v in d.groupby(level="state").mean().items()}
        out[f"ica_minus_raw_{c}"] = res
    dst = ROOT / "results/revision/anphy_ica_participant.json"
    dst.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
