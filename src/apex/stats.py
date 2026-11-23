"""Patient-level statistics: paired sign-flip permutation, bootstrap CI, Holm adjustment."""
from __future__ import annotations

import itertools

import numpy as np


def signflip_test(x, alternative="greater", n_perm=100000, seed=0):
    """Sign-flip permutation test of mean(x) against 0.

    n <= 16: exact enumeration of all 2^n sign patterns (the observed pattern is included).
    n > 16: n_perm random sign patterns; p = (b + 1) / (n_perm + 1), where b is the number of patterns whose mean is at
    least as extreme as the observed mean (Phipson & Smyth, 2010), so p is never zero.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    obs = x.mean()
    exact = n <= 16
    if exact:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([1, -1], size=(n_perm, n))
    null = (signs * x[None]).mean(1)
    if alternative == "greater":
        hits = null >= obs - 1e-15
    elif alternative == "less":
        hits = null <= obs + 1e-15
    else:
        hits = np.abs(null) >= abs(obs) - 1e-15
    p = hits.mean() if exact else (hits.sum() + 1) / (n_perm + 1)
    return dict(n=n, mean=float(obs), p=float(p), n_positive=int(np.sum(x > 0)),
                p_method="exact" if exact else "mc_plus1")


def bootstrap_ci(samples, level=0.95):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size < 10:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
