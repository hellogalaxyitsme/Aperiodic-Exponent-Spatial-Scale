"""Inventory of the exploratory family: every participant-level sign-flip test with its source and
Benjamini-Hochberg q (Supporting Information Data S1). Monte Carlo p-values are (b + 1)/(B + 1); outputs that stored
b/B are converted exactly. A test reported by several analyses is counted once (listed in also_exported_as); opposite
one-sided tests of the same quantity remain separate. The ANPHY ICA comparisons are taken from
scalp_anphy_ica_participant_tests.py (participant-level). Writes results/revision/statistical_inventory.csv.
Usage: python analysis/statistical_inventory.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


R = Path("results")
FAMILY = ["robust/group.json", "anphy_v2/group.json", "straddle/group.json", "grid_v2/measure_group.json",
          "ds3478_followup/group.json", "final_round/seeg_group.json", "final_round/grid_robustness.json",
          "final_round/scalp_group.json"]
CHECKS = ["revision/models_group.json", "revision/eog_eyestate_group.json", "revision/interp_group.json", "revision/ecg_group.json",
        "revision/sleep_derivation.json"]


def walk(o, path, rows, src, n_group=None, scalp=False):
    if isinstance(o, dict):
        if "p" in o and isinstance(o["p"], (int, float)):
            if scalp:
                key = path.split("/")[0]
                if not np.isfinite(o.get("mean", np.nan)) or (n_group is not None and o.get("n", n_group) < n_group):
                    return
                if key.startswith("modma_") and "irasa_all_pair" in key:
                    return
            rows.append(dict(source=src, test=path, n=o.get("n"), estimate=o.get("mean"), n_positive=o.get("n_positive"),
                             p=float(o["p"]), p_method=o.get("p_method")))
            return
        ng = o["n"] if scalp and isinstance(o.get("n"), int) and "p" not in o else n_group
        for k, v in o.items():
            walk(v, f"{path}/{k}" if path else k, rows, src, ng, scalp)


rows = []
for f in FAMILY + [e for e in CHECKS if (R / e).exists()]:
    walk(json.load(open(R / f)), "", rows, f, scalp=f.endswith("scalp_group.json"))
D = pd.DataFrame(rows)
D = D[~((D.source == "revision/sleep_derivation.json") & D.test.str.startswith("spfixed"))].reset_index(drop=True)
# E036: the pooled ICA tests in anphy_v2 sign-flipped participant-states independently; they are replaced by the
# participant-level tests (mean over states per participant, n = 26).
ICA = "revision/anphy_ica_participant.json"
D = D[~((D.source == "anphy_v2/group.json") & D.test.str.startswith("ica_minus_raw_"))]
ica_rows = []
walk({k: v for k, v in json.load(open(R / ICA)).items() if k.startswith("ica_minus_raw_")}, "", ica_rows, ICA)
D = pd.concat([D, pd.DataFrame(ica_rows)], ignore_index=True)

# Valid Monte Carlo p-values (Phipson & Smyth 2010): p = (b + 1) / (B + 1) for randomly sampled sign patterns (n > 16).
# Older outputs stored b / B; b is recovered exactly from p and the number of permutations B used by each analysis.
# Exact enumeration (n <= 16) already includes the observed sign pattern and is kept unchanged.
D["n_perm"] = np.where((D.source == "anphy_v2/group.json") & ~D.test.str.contains(r"^[A-Z0-9]+_minus_[A-Z]+_"), 20000, 100000)
mc = (D.n > 16) & ~D.get("p_method", pd.Series(index=D.index, dtype=object)).eq("mc_plus1")
b = np.round(D.p * D.n_perm)
assert np.allclose(D.p[mc] * D.n_perm[mc], b[mc], atol=1e-6), "p is not b/B for some Monte Carlo rows"
D.loc[mc, "p"] = (b[mc] + 1) / (D.n_perm[mc] + 1)
D["permutation"] = np.where(D.n > 16, D.n_perm.astype(str) + " random", "exact")

# One row per test: the same test exported by several analyses (same n and estimate, p on the same side of 0.5) is kept
# once, preferring the main-analysis output; opposite one-sided tests of the same quantity remain separate tests.
PRIORITY = ["final_round/", "robust/", "straddle/", "grid_v2/", "ds3478_followup/", "anphy_v2/", "revision/"]
D["_rank"] = [next(i for i, s in enumerate(PRIORITY) if src.startswith(s)) for src in D.source]
D["_key"] = (D.n.astype(int).astype(str) + "|" + D.estimate.round(6).astype(str) + "|" + np.where(D.p > 0.5, "hi", "lo"))
D = D.sort_values(["_rank"], kind="stable")
dups = D[D.duplicated("_key")].groupby("_key").apply(lambda g: "; ".join(g.source + ":" + g.test))
D = D.drop_duplicates("_key").copy()
D["also_exported_as"] = D._key.map(dups).fillna("")
D = D.sort_index().drop(columns=["_rank", "_key", "n_perm"]).reset_index(drop=True)
m = len(D)
order = np.argsort(D.p.to_numpy())
q = np.empty(m)
q[order] = np.minimum.accumulate((D.p.to_numpy()[order] * m / np.arange(1, m + 1))[::-1])[::-1]
D["q_bh"] = np.minimum(q, 1.0)
D["survives_bh_5pct"] = D.q_bh <= 0.05
out = R / "revision" / "statistical_inventory.csv"
out.parent.mkdir(parents=True, exist_ok=True)
D.to_csv(out, index=False)
print(f"family {m}; p<.05 {(D.p < .05).sum()}; BH survivors {D.survives_bh_5pct.sum()}; sources:", D.source.value_counts().to_dict())
