"""Pool out-of-fold predictions, compute metrics, group-bootstrap CIs and significance tests."""
import glob, json, re, sys, numpy as np, psi_lib as L
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from sklearn.metrics import f1_score
R = L.OUT + '/results'


def pooled(data, proto, model, tag=''):
    fs = sorted(glob.glob(f'{R}/{data}_{proto}_{model}{tag}_f[0-9]_prob.npz'))
    if len(fs) < 5: return None
    P, idx = [], []
    for f in fs:
        d = np.load(f); P.append(d['P']); idx.append(d['te'])
    return np.concatenate(P), np.concatenate(idx)


def fold_metrics(data, proto, model, key='macro_f1'):
    out = []
    for k in range(5):
        try: out.append(json.load(open(f'{R}/{data}_{proto}_{model}_f{k}.json'))[key])
        except FileNotFoundError: return None
    return np.array(out)


def group_boot_ci(y, pred, groups, B=500, seed=0, fn=None):
    """Cluster bootstrap: each replicate draws connections with replacement (Poisson-multinomial
    weights per connection) and recomputes macro-F1 from the weighted confusion matrix."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True); G = len(ug)
    K = int(max(y.max(), pred.max())) + 1
    present = np.unique(y); vals = []
    cell = y * K + pred
    for b in range(B):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[inv].astype(float)
        cm = np.bincount(cell, weights=w, minlength=K * K).reshape(K, K)
        tp = np.diag(cm); P = cm.sum(0); T = cm.sum(1)
        f1 = np.where(P + T > 0, 2 * tp / np.maximum(P + T, 1e-12), 0.0)
        pres = present[T[present] > 0]
        vals.append(f1[pres].mean())
    return np.percentile(vals, [2.5, 97.5])


def mf1(y, p): return f1_score(y, p, average='macro', labels=np.unique(y))


def summarize(data, proto, models, B=300):
    Xn, Xc, y, groups, meta = L.load(data)
    rows = {}
    for m in models:
        r = pooled(data, proto, m)
        if r is None: continue
        P, idx = r
        met = L.metrics(P, y[idx]); pr = P.argmax(1)
        lo, hi = group_boot_ci(y[idx], pr, groups[idx], B=B, fn=mf1)
        fj = [json.load(open(f)) for f in sorted(glob.glob(f'{R}/{data}_{proto}_{m}_f[0-9].json'))]
        met['mf1_ci'] = [float(lo), float(hi)]
        met['train_seconds'] = float(np.mean([j.get('train_seconds', np.nan) for j in fj]))
        met['test_seconds'] = float(np.mean([j.get('test_seconds', np.nan) for j in fj]))
        met['n_params'] = fj[0].get('n_params'); met['n_test'] = int(len(idx))
        met['fold_mf1'] = [j['macro_f1'] for j in fj]; met['fold_acc'] = [j['acc'] for j in fj]
        met['robust'] = [j.get('robust', {}) for j in fj]
        met['ceiling_fold'] = [j['ceiling_acc'] for j in fj]; met['ceiling_mf1_fold'] = [j['ceiling_macro_f1'] for j in fj]
        rows[m] = met
    return rows


if __name__ == '__main__':
    spec = json.loads(sys.argv[1])
    out = {f"{d}|{p}": summarize(d, p, ms) for d, p, ms in spec}
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
    for k, v in out.items():
        print(k)
        for m, r in v.items():
            print(f"  {m:14s} acc {r['acc']:.4f} mF1 {r['macro_f1']:.4f} [{r['mf1_ci'][0]:.3f},{r['mf1_ci'][1]:.3f}] "
                  f"mcc {r['mcc']:.4f} tpr {r['det_tpr']:.4f} fpr {r['det_fpr']:.4f} auc {r['det_auroc']:.4f} ece {r['ece']:.4f}")
