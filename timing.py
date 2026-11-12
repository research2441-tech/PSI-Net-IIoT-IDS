"""Inference cost on CPU: throughput (batch 1024) and single-record latency, Edge-IIoTset fold 0."""
import time, json, numpy as np, jax, psi_lib as L, psi_models as M
import lightgbm as lgb
Xn, Xc, y, g, meta = L.load('edge_audited'); K = len(meta['classes'])
segn = meta['segments']; u = list(dict.fromkeys(segn)); sid = np.array([u.index(s) for s in segn]); G = len(u)
nc = [meta['cat_vocab'][c].index('NONE') for c in meta['cat_features']]
F = L.folds(y, g); tr, te = F[0]; fi, va, ca = L.inner_split(tr, y, g, 0.1, 0)
prep = L.Prep().fit(Xn[fi], Xc[fi]); prep.fit_ple(Xn[fi], 16)
Z = np.concatenate([prep.num(Xn[fi]), Xc[fi].astype(float)], 1); order = L.seriation_order(Z, sid)
seg_o = tuple(int(s) for s in sid[order]); masks = M.build_masks(seg_o)
te = te[:8192]
tokp = lambda idx: (np.concatenate([prep.num(Xn[idx]), prep.ple(Xn[idx]).reshape(len(idx), -1)], 1), Xc[idx].astype(np.int32), L.presence(Xn[idx], Xc[idx], sid, G, nc))
sc = lambda idx: (prep.scalar_view(Xn[idx], Xc[idx]),)
models = {'psi+ple': (M.PSINet(Xn.shape[1], tuple(prep.ncat), tuple(int(o) for o in order), seg_o, G, K, masks, ple=16), tokp),
          'ftt+ple': (M.FTTransformer(Xn.shape[1], tuple(prep.ncat), K, ple=16), tokp),
          'inc1d': (M.Inception1D(K), sc), 'cnn1d': (M.CNN1D(K), sc), 'mlp': (M.MLP(K), lambda idx: (prep.onehot_view(Xn[idx], Xc[idx]),)), 'inc2d': (M.Inception2D(K, side=7), sc), 'cnnbigru': (M.CNNBiGRU(K), sc)}
out = {}
for name, (m, view) in models.items():
    x = view(te); params = m.init({'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}, *[a[:2] for a in x])['params']
    f = jax.jit(lambda p, *b: m.apply({'params': p}, *b))
    for bs in (1024, 1):
        xb = tuple(a[:bs] for a in x); f(params, *xb).block_until_ready()
        n = 8192 // bs if bs > 1 else 300; t = time.time()
        for i in range(n):
            xb = tuple(a[(i * bs) % 8192:(i * bs) % 8192 + bs] for a in x); f(params, *xb).block_until_ready()
        dt = (time.time() - t) / (n * bs)
        out.setdefault(name, {})[f'ms_per_record_bs{bs}'] = 1000 * dt
    out[name]['params'] = M.n_params(params)
X = np.concatenate([Xn, Xc], 1)
clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1)
clf.fit(X[fi], y[fi], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
for bs in (1024, 1):
    n = 8 if bs > 1 else 300; t = time.time()
    for i in range(n): clf.predict_proba(X[te[(i * bs) % 8192:(i * bs) % 8192 + bs]])
    out.setdefault('lgbm', {})[f'ms_per_record_bs{bs}'] = 1000 * (time.time() - t) / (n * bs)
out['lgbm']['trees'] = int(clf.best_iteration_ or 600) * K
json.dump(out, open(L.OUT + '/timing.json', 'w'), indent=1); print(out)
