import json, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import psi_lib as L
R = L.OUT + '/results'


def block_matrix(models, datasets=('edge_audited', 'wustl'), key='macro_f1'):
    M = []
    for m in models:
        row = []
        for d in datasets:
            for k in range(5):
                row.append(json.load(open(f'{R}/{d}_group_{m}_f{k}.json'))[key])
        M.append(row)
    return np.array(M)   # models x blocks


def tests(models, ref='psi+ple'):
    M = block_matrix(models)
    ranks = np.array([rankdata(-M[:, b]) for b in range(M.shape[1])]).T
    fr = friedmanchisquare(*M)
    i = models.index(ref); out = {'friedman_chi2': float(fr.statistic), 'friedman_p': float(fr.pvalue),
                                 'mean_rank': {m: float(ranks[j].mean()) for j, m in enumerate(models)}, 'pairs': {}}
    k = len(models); n = M.shape[1]
    out['nemenyi_cd'] = float(3.219 * np.sqrt(k * (k + 1) / (6 * n)))   # Nemenyi q_0.05 for k=11 models
    ps = []
    for j, m in enumerate(models):
        if j == i: continue
        d = M[i] - M[j]
        p = wilcoxon(M[i], M[j], zero_method='zsplit').pvalue if np.any(d != 0) else 1.0
        ps.append((m, float(p), float(np.median(d)), int((d > 0).sum())))
    # Holm step-down
    order = sorted(range(len(ps)), key=lambda t: ps[t][1]); adj = [0] * len(ps); run = 0
    for r, t in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[t][1])); adj[t] = run
    for (m, p, md, w), a in zip(ps, adj):
        out['pairs'][m] = dict(p=p, p_holm=a, median_diff=md, wins=w, blocks=n)
    return out
