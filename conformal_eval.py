"""Unknown-attack flagging from saved embeddings: Mondrian conformal Mahalanobis test
versus softmax/energy/plain-Mahalanobis thresholds. Usage: python conformal_eval.py RESULT_DIR PATTERN"""
import sys, glob, json, numpy as np
from scipy.special import logsumexp, softmax


def maha_fit(Ef, yf, classes, beta=0.1):
    mu = {c: Ef[yf == c].mean(0) for c in classes}
    R = np.concatenate([Ef[yf == c] - mu[c] for c in classes])
    S = R.T @ R / len(R); D = S.shape[0]
    S = (1 - beta) * S + beta * np.trace(S) / D * np.eye(D)
    return mu, np.linalg.inv(S)


def maha(E, mu, P, classes):
    return np.stack([np.einsum('nd,de,ne->n', E - mu[c], P, E - mu[c]) for c in classes], 1)


def evaluate(f, eps=0.05, holdout=None):
    d = np.load(f)
    yc, yt, yf = d['yc'], d['yt'], d['yf']
    classes = [c for c in np.unique(yf) if (yc == c).sum() >= 5]
    mu, P = maha_fit(d['Ef'], yf, classes)
    Ac, At = maha(d['Ec'], mu, P, classes), maha(d['Et'], mu, P, classes)
    cidx = {c: i for i, c in enumerate(classes)}
    # Mondrian conformal p-values
    pv = np.zeros_like(At)
    for c in classes:
        cal = np.sort(Ac[yc == c, cidx[c]]); n = len(cal)
        pv[:, cidx[c]] = (1 + n - np.searchsorted(cal, At[:, cidx[c]], side='left')) / (n + 1)
    unk_conf = pv.max(1) < eps
    # plain thresholds calibrated to the same marginal false-unknown rate on calibration data
    known_c = np.isin(yc, classes)
    Lc, Lt = d['Lc'][:, classes], d['Lt'][:, classes]
    msp_c, msp_t = softmax(Lc, 1).max(1), softmax(Lt, 1).max(1)
    en_c, en_t = logsumexp(Lc, 1), logsumexp(Lt, 1)
    md_c, md_t = Ac.min(1), At.min(1)
    flags = {'conformal_mahalanobis': unk_conf,
             'msp': msp_t < np.quantile(msp_c[known_c], eps),
             'energy': en_t < np.quantile(en_c[known_c], eps),
             'mahalanobis_global': md_t > np.quantile(md_c[known_c], 1 - eps)}
    known_t = np.isin(yt, classes)
    pred_cls = np.array(classes)[Lt.argmax(1)]
    out = {}
    for k, fl in flags.items():
        r = dict(false_unknown_known=float(fl[known_t].mean()))
        # class-conditional false-unknown rates (worst class) on known test classes
        per = [fl[yt == c].mean() for c in classes if (yt == c).sum() >= 50]
        r['worst_class_false_unknown'] = float(np.max(per)); r['mean_class_false_unknown'] = float(np.mean(per))
        if holdout is not None:
            u = yt == holdout
            r['unknown_detect'] = float(fl[u].mean())
            r['unknown_alert'] = float((fl[u] | (pred_cls[u] != 0)).mean())   # flagged or sent to some attack class
            r['unknown_as_normal'] = float((~fl[u] & (pred_cls[u] == 0)).mean())
        out[k] = r
    # prediction-set coverage/size for known classes (conformal only)
    sets = pv > eps
    yk = yt[known_t]; cov = sets[known_t][np.arange(known_t.sum()), [cidx[c] for c in yk]]
    out['conformal_mahalanobis']['coverage'] = float(cov.mean())
    out['conformal_mahalanobis']['avg_set_size'] = float(sets[known_t].sum(1).mean())
    return out


if __name__ == '__main__':
    res = {}
    for f in sorted(glob.glob(f'{sys.argv[1]}/{sys.argv[2]}')):
        ho = None
        if '_ho-' in f:
            import psi_lib as L; meta = json.load(open(L.OUT + '/' + f.split('/')[-1].split('_group')[0].split('_random')[0] + '.json'))
            ho = meta['classes'].index(f.split('_ho-')[1].split('_f')[0])
        res[f.split('/')[-1]] = {e: evaluate(f, e, ho) for e in [0.01, 0.05, 0.1]}
    json.dump(res, open(sys.argv[3] if len(sys.argv) > 3 else 'conformal_results.json', 'w'), indent=1)
    print(len(res), 'files')


def cluster_weights(g):
    """Weight each record by 1/|connection| so every test connection counts once."""
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def evaluate_groups(f, gc, gt, eps=0.05, holdout=None, draws=20, seed=0):
    """Record-level vs connection-level Mondrian conformal tests plus threshold baselines.
    Connection-level calibration keeps one randomly drawn record per calibration connection."""
    d = np.load(f)
    yc, yt, yf = d['yc'], d['yt'], d['yf']
    assert len(gc) == len(yc) and len(gt) == len(yt)
    classes = [c for c in np.unique(yf) if (yc == c).sum() >= 1]
    mu, P = maha_fit(d['Ef'], yf, classes)
    Ac, At = maha(d['Ec'], mu, P, classes), maha(d['Et'], mu, P, classes)
    ci = {c: i for i, c in enumerate(classes)}
    wt = cluster_weights(gt)
    known_t = np.isin(yt, classes)

    def pvals(sel):
        pv = np.zeros_like(At)
        for c in classes:
            m = (yc == c) & sel
            cal = np.sort(Ac[m, ci[c]]); n = len(cal)
            pv[:, ci[c]] = (1 + n - np.searchsorted(cal, At[:, ci[c]], side='left')) / (n + 1)
        return pv

    rng = np.random.default_rng(seed)
    Lhat = d['Lt'][:, classes].argmax(1); flag_p = np.zeros(len(yt))
    # one record per calibration connection, repeated draws
    flag_g = np.zeros(len(yt)); cov_g = np.zeros(len(yt)); size_g = np.zeros(len(yt))
    ug, inv = np.unique(gc, return_inverse=True)
    for r in range(draws):
        order = rng.permutation(len(gc)); pick = np.zeros(len(gc), bool)
        _, first = np.unique(inv[order], return_index=True); pick[order[first]] = True
        pv = pvals(pick); s = pv > eps
        flag_g += ~s.any(1); size_g += s.sum(1)
        cov_g += np.array([s[i, ci[y]] if y in ci else False for i, y in enumerate(yt)])
        pc_hat = pv[np.arange(len(yt)), Lhat]
        flag_p += (pc_hat <= eps)
    flag_g /= draws; cov_g /= draws; size_g /= draws; flag_p /= draws
    pv_r = pvals(np.ones(len(yc), bool)); s_r = pv_r > eps
    flag_r = ~s_r.any(1).astype(float) if False else (~s_r.any(1)).astype(float)
    cov_r = np.array([s_r[i, ci[y]] if y in ci else False for i, y in enumerate(yt)], float)
    Lc, Lt = d['Lc'][:, classes], d['Lt'][:, classes]
    kc = np.isin(yc, classes)
    msp_c, msp_t = softmax(Lc, 1).max(1), softmax(Lt, 1).max(1)
    en_c, en_t = logsumexp(Lc, 1), logsumexp(Lt, 1)
    md_c, md_t = Ac.min(1), At.min(1)
    pc_r = pv_r[np.arange(len(yt)), Lhat]
    flags = {'conformal_connection': flag_g, 'conformal_record': flag_r, 'conformal_pred_conn': flag_p,
             'conformal_pred_record': (pc_r <= eps).astype(float),
             'msp': (msp_t < np.quantile(msp_c[kc], eps)).astype(float),
             'energy': (en_t < np.quantile(en_c[kc], eps)).astype(float),
             'mahalanobis_global': (md_t > np.quantile(md_c[kc], 1 - eps)).astype(float)}
    pred_cls = np.array(classes)[Lt.argmax(1)]
    ncal_groups = {int(c): int(len(np.unique(gc[yc == c]))) for c in classes}
    out = {'ncal_groups': ncal_groups}
    for k, fl in flags.items():
        r = {}
        per_r, per_w = [], []
        for c in classes:
            m = yt == c
            if m.sum() < 50: continue
            per_r.append(fl[m].mean()); per_w.append((fl[m] * wt[m]).sum() / wt[m].sum())
        r['fur_record'] = float(fl[known_t].mean())
        r['fur_conn'] = float((fl[known_t] * wt[known_t]).sum() / wt[known_t].sum())
        r['worst_fur_record'] = float(max(per_r)); r['worst_fur_conn'] = float(max(per_w))
        r['mean_class_fur_conn'] = float(np.mean(per_w))
        if holdout is not None:
            u = yt == holdout
            r['unknown_detect'] = float(fl[u].mean())
            r['unknown_detect_conn'] = float((fl[u] * wt[u]).sum() / wt[u].sum())
            r['unknown_alert'] = float(np.mean(np.maximum(fl[u], (pred_cls[u] != 0).astype(float))))
            r['unknown_as_normal'] = float(np.mean((1 - fl[u]) * (pred_cls[u] == 0)))
        out[k] = r
    out['conformal_connection']['coverage_conn'] = float((cov_g[known_t] * wt[known_t]).sum() / wt[known_t].sum())
    out['conformal_connection']['avg_set_size'] = float(size_g[known_t].mean())
    out['conformal_record']['coverage_conn'] = float((cov_r[known_t] * wt[known_t]).sum() / wt[known_t].sum())
    out['conformal_record']['avg_set_size'] = float(s_r[known_t].sum(1).mean())
    return out
