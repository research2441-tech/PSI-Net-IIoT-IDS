"""Shared utilities: loading, group-disjoint splitting, preprocessing, metrics."""
import json, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score)
from scipy.stats import spearmanr

import os
OUT = os.environ.get('PSI_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))


def load(name, out=OUT):
    d = np.load(f'{out}/{name}.npz'); m = json.load(open(f'{out}/{name}.json'))
    return d['Xnum'], d['Xcat'], d['y'].astype(np.int64), d['groups'], m


def folds(y, groups, k=5, seed=0):
    """Connection/session-disjoint stratified folds."""
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, groups))


def random_folds(y, k=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def inner_split(idx, y, groups, frac=0.1, seed=0, by_group=True):
    """Split training indices into fit / validation / calibration (group-disjoint)."""
    rng = np.random.default_rng(seed)
    if by_group:
        g = np.unique(groups[idx]); rng.shuffle(g)
        nv = int(frac * len(g)); gv, gc = set(g[:nv]), set(g[nv:2 * nv])
        gi = groups[idx]
        va = idx[np.isin(gi, list(gv))]; ca = idx[np.isin(gi, list(gc))]
        fi = idx[~np.isin(gi, list(gv | gc))]
    else:
        p = rng.permutation(idx); nv = int(frac * len(idx))
        va, ca, fi = p[:nv], p[nv:2 * nv], p[2 * nv:]
    return fi, va, ca


class Prep:
    """Signed-log + z-score for numeric fields, integer codes for categorical fields."""
    def fit(self, Xnum, Xcat):
        z = np.sign(Xnum) * np.log1p(np.abs(Xnum))
        self.mu = z.mean(0); self.sd = z.std(0) + 1e-6
        self.ncat = [int(Xcat[:, j].max()) + 2 for j in range(Xcat.shape[1])]
        return self

    def fit_ple(self, Xnum, T=16):
        z = self.num(Xnum).astype(np.float64)
        qs = np.linspace(0, 1, T + 1)
        self.edges = np.stack([np.quantile(z[:, j], qs) for j in range(z.shape[1])])  # (d, T+1)
        self.T = T
        return self

    def ple(self, Xnum):
        """Quantile piecewise-linear encoding of every numeric field: (N, d, T)."""
        z = self.num(Xnum)[:, :, None]
        lo, hi = self.edges[None, :, :-1], self.edges[None, :, 1:]
        w = np.maximum(hi - lo, 1e-6)
        return np.clip((z - lo) / w, 0, 1).astype(np.float32)

    def num(self, Xnum):
        z = np.sign(Xnum) * np.log1p(np.abs(Xnum))
        return ((z - self.mu) / self.sd).astype(np.float32)

    def scalar_view(self, Xnum, Xcat):
        """One scalar per field (used by scalar-input CNN baselines)."""
        c = [(Xcat[:, j] / max(self.ncat[j] - 2, 1)).astype(np.float32) for j in range(Xcat.shape[1])]
        return np.concatenate([self.num(Xnum)] + ([np.stack(c, 1)] if c else []), 1)

    def onehot_view(self, Xnum, Xcat):
        cols = [self.num(Xnum)]
        for j in range(Xcat.shape[1]):
            oh = np.zeros((len(Xcat), self.ncat[j]), np.float32)
            oh[np.arange(len(Xcat)), np.clip(Xcat[:, j], 0, self.ncat[j] - 1)] = 1
            cols.append(oh)
        return np.concatenate(cols, 1)


def presence(Xnum, Xcat, seg_ids, n_seg, none_codes):
    """Binary indicator that a protocol segment carries any non-default value."""
    nz = np.concatenate([Xnum != 0] + ([Xcat != np.asarray(none_codes)[None]] if Xcat.shape[1] else []), 1)
    P = np.zeros((len(nz), n_seg), np.float32)
    for g in range(n_seg):
        P[:, g] = nz[:, seg_ids == g].any(1)
    return P


def seriation_order(Z, seg_ids, seed=0, n=20000):
    """Protocol-segmented ordering: segments kept contiguous, fields inside a segment
    ordered by the Fiedler vector of the |Spearman| affinity graph."""
    rng = np.random.default_rng(seed)
    S = Z[rng.choice(len(Z), min(n, len(Z)), replace=False)]
    order = []
    for g in np.unique(seg_ids):
        f = np.where(seg_ids == g)[0]
        if len(f) <= 2:
            order += list(f); continue
        sub = S[:, f]
        ok = sub.std(0) > 0
        rho = np.zeros((len(f), len(f)))
        if ok.sum() >= 2:
            r = spearmanr(sub[:, ok]).statistic
            r = np.atleast_2d(np.nan_to_num(r))
            idx = np.where(ok)[0]
            rho[np.ix_(idx, idx)] = np.abs(r)
        np.fill_diagonal(rho, 0)
        L = np.diag(rho.sum(1)) - rho
        w, v = np.linalg.eigh(L)
        fied = v[:, 1] if len(w) > 1 else np.zeros(len(f))
        order += list(f[np.argsort(fied + 1e-9 * np.arange(len(f)))])
    return np.array(order)


def ece(prob, y, bins=15):
    conf = prob.max(1); pred = prob.argmax(1); acc = (pred == y)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(prob, y, normal=0):
    pred = prob.argmax(1)
    pa = 1 - prob[:, normal]; ya = (y != normal).astype(int); pda = (pred != normal).astype(int)
    present = np.unique(y)
    out = dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average='macro', labels=present),
               weighted_f1=f1_score(y, pred, average='weighted'),
               bal_acc=balanced_accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred),
               det_tpr=float(pda[ya == 1].mean()), det_fpr=float(pda[ya == 0].mean()),
               det_auroc=float(roc_auc_score(ya, pa)) if 0 < ya.mean() < 1 else float('nan'),
               ece=ece(prob, y), nll=float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)))))
    out['per_class_f1'] = f1_score(y, pred, average=None, labels=np.arange(prob.shape[1])).tolist()
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in out.items()}


def row_keys(Xnum, Xcat):
    a = np.ascontiguousarray(np.concatenate([Xnum.astype(np.float64), Xcat.astype(np.float64)], 1))
    v = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, inv = np.unique(v, return_inverse=True)
    return inv


def conflict_ceiling(keys, y, n_cls):
    """Accuracy/macro-F1 of the oracle that predicts the majority label of each identical
    feature vector inside the evaluated set (an upper bound for any deterministic classifier)."""
    K = keys.max() + 1
    C = np.zeros((K, n_cls), np.int64); np.add.at(C, (keys, y), 1)
    maj = C.argmax(1)[keys]
    return float((maj == y).mean()), float(f1_score(y, maj, average='macro', labels=np.unique(y)))


def compress(keys, y, n_cls):
    """Collapse identical training vectors: representative index, multiplicity, soft label."""
    K = keys.max() + 1
    C = np.zeros((K, n_cls), np.float64); np.add.at(C, (keys, y), 1)
    first = np.full(K, -1); first[keys[::-1]] = np.arange(len(keys))[::-1]
    n = C.sum(1)
    return first, n, C / n[:, None]
