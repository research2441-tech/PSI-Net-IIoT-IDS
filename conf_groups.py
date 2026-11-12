"""Recover calibration/test connection ids for a saved run (indices are deterministic)."""
import numpy as np, psi_lib as L, json
_cache = {}


def ids(data, fold, holdout=None):
    if data not in _cache:
        Xn, Xc, y, g, m = L.load(data); _cache[data] = (y, g, L.folds(y, g), m)
    y, g, F, m = _cache[data]
    tr, te = F[fold]; fi, va, ca = L.inner_split(tr, y, g, 0.1, seed=fold)
    if holdout:
        h = m['classes'].index(holdout); ca = ca[y[ca] != h]
    return g[ca], g[te], y[ca], y[te]
