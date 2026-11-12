"""Run one (dataset, protocol, model, fold) experiment and store metrics + test probabilities.
Example: python run_exp.py --data edge_audited --protocol group --models psi,lgbm --folds 0,1
"""
import argparse, json, os, time, numpy as np
import psi_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='edge_audited'); ap.add_argument('--protocol', default='group')
ap.add_argument('--models', default='psi'); ap.add_argument('--folds', default='0,1,2,3,4')
ap.add_argument('--tag', default=''); ap.add_argument('--gamma', type=float, default=0.5)
ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--max_seconds', type=float, default=150)
ap.add_argument('--robust', type=int, default=0); ap.add_argument('--holdout', default=''); ap.add_argument('--out', default=L.OUT + '/results')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

Xn, Xc, y, groups, meta = L.load(args.data)
K = len(meta['classes'])
seg_names = meta['segments']; uniq = list(dict.fromkeys(seg_names))
seg_ids = np.array([uniq.index(s) for s in seg_names]); G = len(uniq)
none_codes = [meta['cat_vocab'][c].index('NONE') if 'NONE' in meta['cat_vocab'][c] else -1 for c in meta['cat_features']]
F = (L.folds(y, groups) if args.protocol == 'group' else L.random_folds(y))
keys_all = L.row_keys(Xn, Xc)


def run_one(mname, fold):
    tag = f'{args.data}_{args.protocol}_{mname}{args.tag}' + (f'_ho-{args.holdout}' if args.holdout else '') + f'_f{fold}'
    path = f'{args.out}/{tag}.json'
    if os.path.exists(path): print('skip', tag); return
    tr, te = F[fold]
    fi, va, ca = L.inner_split(tr, y, groups, 0.1, seed=fold, by_group=(args.protocol == 'group'))
    if args.holdout:
        h = meta['classes'].index(args.holdout)
        fi, va, ca = fi[y[fi] != h], va[y[va] != h], ca[y[ca] != h]
    prep = L.Prep().fit(Xn[fi], Xc[fi])
    PLE = 16 if '+ple' in mname else 0
    if PLE: prep.fit_ple(Xn[fi], PLE)
    t0 = time.time(); info = {}
    import psi_models as M
    import jax
    def views(idx, kind):
        if kind == 'tok':
            xn = prep.num(Xn[idx])
            if PLE: xn = np.concatenate([xn, prep.ple(Xn[idx]).reshape(len(idx), -1)], 1)
            return (xn, Xc[idx].astype(np.int32), L.presence(Xn[idx], Xc[idx], seg_ids, G, none_codes))
        if kind == 'scalar': return (prep.scalar_view(Xn[idx], Xc[idx]),)
        if kind == 'onehot': return (prep.onehot_view(Xn[idx], Xc[idx]),)

    def targets(idx, scheme):
        if scheme == 'standard':
            return idx, np.eye(K, dtype=np.float32)[y[idx]], np.ones(len(idx), np.float32)
        kk = keys_all[idx]; _, kk = np.unique(kk, return_inverse=True)
        first, n, Q = L.compress(kk, y[idx], K)
        g = float(scheme.split('=')[1])
        w = n ** g; w = w / w.mean()
        return idx[first], Q.astype(np.float32), w.astype(np.float32)

    base = mname.split('+')[0]; scheme = 'standard'
    for part in mname.split('+')[1:]:
        if part.startswith('cat'): scheme = 'cat=' + (part[3:] if len(part) > 3 else str(args.gamma))
    info['scheme'] = scheme
    if base in ('rf', 'lgbm', 'xgb', 'lr'):
        Xtr = np.concatenate([Xn[fi], Xc[fi]], 1); Xva = np.concatenate([Xn[va], Xc[va]], 1)
        Xte = np.concatenate([Xn[te], Xc[te]], 1)
        if base == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(300, n_jobs=2, random_state=fold, min_samples_leaf=1).fit(Xtr, y[fi])
            P = np.zeros((len(te), K)); P[:, m.classes_] = m.predict_proba(Xte)
        elif base == 'lgbm':
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=fold)
            m.fit(Xtr, y[fi], eval_set=[(Xva, y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
            P = np.zeros((len(te), K)); P[:, m.classes_] = m.predict_proba(Xte); info['iters'] = int(m.best_iteration_ or 600)
        elif base == 'xgb':
            import xgboost as xgb
            yv = y[va]; ok = np.isin(yv, np.unique(y[fi]))
            m = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.08, subsample=0.8,
                                  colsample_bytree=0.8, tree_method='hist', n_jobs=2, early_stopping_rounds=40,
                                  random_state=fold, num_class=K, objective='multi:softprob')
            m.fit(Xtr, y[fi], eval_set=[(Xva[ok], yv[ok])], verbose=False)
            P = m.predict_proba(Xte); info['iters'] = int(m.best_iteration)
        else:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=400, C=1.0).fit(prep.onehot_view(Xn[fi], Xc[fi]), y[fi])
            P = np.zeros((len(te), K)); P[:, m.classes_] = m.predict_proba(prep.onehot_view(Xn[te], Xc[te]))
        info['train_seconds'] = time.time() - t0
    else:
        Xs_fi = prep.scalar_view(Xn[fi], Xc[fi])
        ncat = tuple(prep.ncat); nn_ = Xn.shape[1]
        if base.startswith('psi') or base in ('tokinc',):
            Z = np.concatenate([prep.num(Xn[fi]), Xc[fi].astype(np.float64)], 1)
            order = L.seriation_order(Z, seg_ids, seed=fold)
            if 'randin' in mname:   # random order inside each segment
                rng = np.random.default_rng(100 + fold); order = []
                for g in range(G):
                    f = np.where(seg_ids == g)[0]; order += list(rng.permutation(f))
                order = np.array(order)
            seg_o = tuple(int(s) for s in seg_ids[order]); masks = M.build_masks(seg_o)
        if base == 'psi':
            model = M.PSINet(nn_, ncat, tuple(int(o) for o in order), seg_o, G, K, masks,
                             use_mask=('nomask' not in mname), use_seg=('noseg' not in mname),
                             use_cross=('nocross' not in mname), ple=PLE)
            kind = 'tok'
        elif base == 'tokinc':
            ordr = tuple(int(o) for o in order) if 'ser' in mname else tuple(range(len(seg_ids)))
            model = M.TokenInception(nn_, ncat, ordr, K, ple=PLE); kind = 'tok'
        elif base == 'ftt': model = M.FTTransformer(nn_, ncat, K, ple=PLE); kind = 'tok'
        elif base == 'mlp': model = M.MLP(K); kind = 'onehot'
        elif base == 'cnn1d': model = M.CNN1D(K); kind = 'scalar'
        elif base == 'cnnbigru': model = M.CNNBiGRU(K); kind = 'scalar'
        elif base == 'inc2d': model = M.Inception2D(K, side=int(np.ceil(np.sqrt(len(seg_ids))))); kind = 'scalar'
        elif base.startswith('inc1d'):
            if 'perm' in mname:
                s = int(mname.split('perm')[1].split('+')[0]); order = tuple(int(o) for o in np.random.default_rng(s).permutation(len(seg_ids)))
            elif 'ser' in mname:
                Z = np.concatenate([prep.num(Xn[fi]), Xc[fi].astype(np.float64)], 1)
                order = tuple(int(o) for o in L.seriation_order(Z, seg_ids, seed=fold))
            else: order = ()
            model = M.Inception1D(K, order); kind = 'scalar'
        idx_fit, Q, w = targets(fi, scheme)
        prior = np.bincount(y[fi], minlength=K) / len(fi)
        params, pred, tinfo = M.train_model(model, views(idx_fit, kind), Q, w, views(va, kind), y[va], K,
                                            epochs=args.epochs, seed=fold, log_prior=np.log(prior + 1e-8), patience=4,
                                            max_seconds=args.max_seconds)
        info.update(tinfo); info['n_params'] = M.n_params(params); info['n_fit_rows'] = int(len(idx_fit))
        t1 = time.time(); P = pred(views(te, kind)); info['test_seconds'] = time.time() - t1
        if base == 'psi' or base == 'inc1d' or base == 'ftt':
            emb = M.embed_fn(model, params)
            fsub = np.sort(np.random.default_rng(fold).choice(fi, min(20000, len(fi)), replace=False))
            Lc, Ec = emb(views(ca, kind)); Lt, Et = emb(views(te, kind)); _, Ef = emb(views(fsub, kind))
            np.savez_compressed(f'{args.out}/{tag}_emb.npz', Ec=Ec.astype(np.float32), Et=Et.astype(np.float32),
                                Ef=Ef.astype(np.float32), yc=y[ca], yf=y[fsub], Lc=Lc.astype(np.float32),
                                Lt=Lt.astype(np.float32), yt=y[te])
    rob = {}
    if args.robust and base not in ('rf', 'xgb', 'lr'):
        rng = np.random.default_rng(7 + fold)
        def predict_raw(XnT, XcT):
            if base == 'lgbm':
                Q = np.zeros((len(XnT), K)); Q[:, m.classes_] = m.predict_proba(np.concatenate([XnT, XcT], 1)); return Q
            if kind == 'tok':
                xn = prep.num(XnT)
                if PLE: xn = np.concatenate([xn, prep.ple(XnT).reshape(len(XnT), -1)], 1)
                return pred((xn, XcT.astype(np.int32), L.presence(XnT, XcT, seg_ids, G, none_codes)))
            if kind == 'scalar': return pred((prep.scalar_view(XnT, XcT),))
            return pred((prep.onehot_view(XnT, XcT),))
        sub = np.sort(rng.choice(len(te), min(10000, len(te)), replace=False)); teR = te[sub]
        XnT, XcT = Xn[teR], Xc[teR]
        rob['clean'] = L.metrics(predict_raw(XnT, XcT), y[teR])['macro_f1']
        for sg in [0.1, 0.3, 0.5]:
            # multiplicative log-normal jitter on non-zero numeric fields (measurement noise)
            Xp = XnT * np.exp(sg * rng.standard_normal(XnT.shape)) * (XnT != 0)
            rob[f'noise{sg}'] = L.metrics(predict_raw(Xp, XcT), y[teR])['macro_f1']
        nnum = Xn.shape[1]
        for gi, gname in enumerate(uniq):
            fs = np.where(seg_ids == gi)[0]
            Xp = XnT.copy(); Xq = XcT.copy()
            fn = fs[fs < nnum]; fc = fs[fs >= nnum] - nnum
            Xp[:, fn] = 0
            for j in fc: Xq[:, j] = none_codes[j] if none_codes[j] >= 0 else 0
            rob[f'drop_{gname}'] = L.metrics(predict_raw(Xp, Xq), y[teR])['macro_f1']
        if 'kind' in dir() and kind in ('scalar',) or base == 'inc1d':
            pass
    res = L.metrics(P, y[te]); res['robust'] = rob
    kt = keys_all[te]; _, kt = np.unique(kt, return_inverse=True)
    res['ceiling_acc'], res['ceiling_macro_f1'] = L.conflict_ceiling(kt, y[te], K)
    res.update(info); res['model'] = mname; res['fold'] = fold; res['data'] = args.data; res['protocol'] = args.protocol
    json.dump(res, open(path, 'w'))
    np.savez_compressed(f'{args.out}/{tag}_prob.npz', P=P.astype(np.float32), te=te)
    print(f"{tag}: acc {res['acc']:.4f} mF1 {res['macro_f1']:.4f} ceil {res['ceiling_acc']:.4f} "
          f"t {info.get('train_seconds',0):.0f}s ep {info.get('epochs_run','-')}", flush=True)


for f in [int(x) for x in args.folds.split(',')]:
    for m in args.models.split(','):
        run_one(m, f)
