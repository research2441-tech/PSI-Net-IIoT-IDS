"""Aggregate every experiment into one JSON of numbers used by the manuscript and the figures."""
import json, glob, os, numpy as np, subprocess
import psi_lib as L, aggregate as A, stats_tests as S, conformal_eval as CE, fig_results as FR
R = L.OUT + '/results'
MODELS = ['lr', 'rf', 'xgb', 'lgbm', 'mlp', 'cnn1d', 'cnnbigru', 'inc2d', 'inc1d', 'ftt+ple', 'psi+ple']
V = {}


def fj(data, proto, m, k):
    p = f'{R}/{data}_{proto}_{m}_f{k}.json'
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------- main benchmark
agg = {}
for d in ['edge_audited', 'wustl']:
    agg[f'{d}|group'] = A.summarize(d, 'group', MODELS, B=300)
for key in agg:
    for m, r in agg[key].items():
        r.pop('robust', None)
V['main'] = agg
# robustness (mean over folds) for Edge-IIoTset
rob = {}
for m in ['lgbm', 'inc1d', 'ftt+ple', 'psi+ple']:
    rs = [fj('edge_audited', 'group', m, k) for k in range(5)]
    rs = [r['robust'] for r in rs if r and r.get('robust')]
    if rs: rob[m] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
V['robust_edge'] = rob
rob = {}
for m, tag in [('lgbm', 'lgbmR'), ('inc1d', 'inc1d'), ('ftt+ple', 'ftt+ple'), ('psi+ple', 'psi+ple')]:
    rs = [fj('wustl', 'group', tag, k) for k in range(5)]
    rs = [r['robust'] for r in rs if r and r.get('robust')]
    if rs: rob[m] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
V['robust_wustl'] = rob
# statistics
try:
    V['stats'] = S.tests(MODELS)
except Exception as e:
    V['stats'] = {'error': str(e)}
# ---------------- protocol experiment (folds available per setting)
prot = {}
for s, (d, p) in {'ref|random': ('edge_ferrag', 'random'), 'audited|random': ('edge_audited', 'random'),
                  'ref|group': ('edge_ferrag', 'group'), 'audited|group': ('edge_audited', 'group')}.items():
    prot[s] = {}
    for m in ['lgbm', 'psi+ple']:
        rs = [fj(d, p, m, k) for k in range(5)]; rs = [r for r in rs if r]
        if m != 'lgbm': rs = [r for r in rs if r['fold'] in (0, 1)]
        if rs: prot[s][m] = (float(np.mean([r['macro_f1'] for r in rs])), [r['macro_f1'] for r in rs],
                             float(np.mean([r['acc'] for r in rs])), [r['fold'] for r in rs])
    cs = [fj(d, p, 'lgbm', k) for k in range(5)]; cs = [r for r in cs if r]
    if cs:
        prot[s]['ceiling'] = float(np.mean([r['ceiling_macro_f1'] for r in cs]))
        prot[s]['ceiling_acc'] = float(np.mean([r['ceiling_acc'] for r in cs]))
V['protocol'] = prot
# ---------------- ablations (folds 0,1)
abl = {}
for d in ['edge_audited', 'wustl']:
    abl[d] = {}
    for m in ['psi+ple', 'psi', 'psi+ple+nomask', 'psi+ple+noseg', 'psi+ple+nocross', 'psi+ple+randin', 'tokinc+ple',
              'inc1d', 'inc1d+ser', 'inc1d+perm11', 'inc1d+perm12']:
        rs = [fj(d, 'group', m, k) for k in (0, 1)]
        if all(rs): abl[d][m] = dict(mf1=float(np.mean([r['macro_f1'] for r in rs])), folds=[r['macro_f1'] for r in rs],
                                    acc=float(np.mean([r['acc'] for r in rs])))
V['ablation'] = abl
# ---------------- conformal
import conf_groups as CG
conf = {}
for d in ['edge_audited', 'wustl']:
    for m in ['psi+ple']:
        res = []
        for k in range(5):
            f = f'{R}/{d}_group_{m}_f{k}_emb.npz'
            if not os.path.exists(f): continue
            gc, gt, _, _ = CG.ids(d, k)
            res.append({e: CE.evaluate_groups(f, gc, gt, e, draws=10) for e in [0.01, 0.05, 0.1]})
        if res: conf[f'{d}|{m}|closed'] = res
    meta = json.load(open(f'{L.OUT}/{d}.json'))
    for f in sorted(glob.glob(f'{R}/{d}_group_psi+ple_ho-*_f0_emb.npz')):
        fam = f.split('_ho-')[1].split('_f0')[0]
        gc, gt, _, _ = CG.ids(d, 0, fam)
        conf[f'{d}|psi+ple|ho|{fam}'] = {e: CE.evaluate_groups(f, gc, gt, e, meta['classes'].index(fam), draws=10) for e in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]}
        rj = json.load(open(f.replace('_emb.npz', '.json')))
        conf[f'{d}|psi+ple|ho|{fam}']['closed_metrics'] = {k: rj[k] for k in ('acc', 'macro_f1')}
V['conformal'] = conf
json.dump(V, open(L.OUT + '/V.json', 'w'), indent=1, default=float)
print('saved', list(V))
