"""C1 restricted to classifiable pairs: at least one intervening contact, all grey matter (Results 3.1).
Usage: python analysis/depth_anatomy_classifiable.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from apex import stats  # noqa: E402

S = pd.read_csv("results/straddle/straddle_pairs.csv")
for coh in ("ds004752", "HUP"):
    g = S[S.cohort == coh]
    for lab, m in (("contiguous, non-adjacent", (g.crossing == 0) & (g.n_between >= 1)),
                   ("crossing", g.crossing == 1),
                   ("adjacent only (unclassifiable)", g.n_between == 0)):
        sl = [np.polyfit(h.ld, h.chi, 1)[0] for _, h in g[m].groupby("sub") if len(h) >= 3 and h.ld.std() > 0]
        print(coh, lab, "pairs", int(m.sum()), stats.signflip_test(np.array(sl), "greater") if len(sl) >= 2 else len(sl))
    # within-distance-matched comparison: crossing vs contiguous non-adjacent at same n_between
    sub = g[g.n_between >= 1]
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        continue
    if sub.crossing.sum() > 10:
        fit = smf.mixedlm("chi ~ ld + crossing", sub, groups=sub["sub"]).fit(method="lbfgs")
        print(coh, "non-adjacent pairs mixed model: ld", round(fit.params["ld"], 3), "p", round(fit.pvalues["ld"], 5),
              "| crossing", round(fit.params["crossing"], 3), "p", round(fit.pvalues["crossing"], 3), "n", len(sub))
