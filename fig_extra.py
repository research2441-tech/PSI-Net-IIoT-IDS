import os
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import psi_lib as L
from fig_results import NAMES
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'mathtext.fontset': 'dejavuserif'})
V = json.load(open(L.OUT + '/V.json'))
E = V['main']['edge_audited|group']
fig, ax = plt.subplots(figsize=(3.45, 2.2)); x = np.arange(5)
ax.bar(x - 0.18, E['psi+ple']['ceiling_fold'], 0.34, color='white', edgecolor='black', lw=0.6, label='Ceiling, accuracy')
ax.bar(x + 0.18, E['psi+ple']['ceiling_mf1_fold'], 0.34, color='#d9d9d9', edgecolor='black', lw=0.6, hatch='////', label='Ceiling, macro-F1')
for m, mk, c in [('lgbm', 's', 'white'), ('psi+ple', 'o', 'black')]:
    ax.plot(x + 0.18, E[m]['fold_mf1'], marker=mk, ls='none', ms=4.5, mfc=c, mec='black', label=f'{NAMES[m]} macro-F1')
ax.set_xticks(x); ax.set_xticklabels([f'Fold {i + 1}' for i in range(5)]); ax.set_ylim(0.6, 1.0); ax.set_ylabel('Score')
ax.grid(axis='y', lw=0.3, color='#cccccc')
ax.legend(fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.28), columnspacing=1.0)
os.makedirs('figs', exist_ok=True); fig.savefig('figs/fig_ceiling.png', dpi=400, bbox_inches='tight'); plt.close(fig)
# seriation illustration: TCP segment, fold 1 fitting connections
Xn, Xc, y, g, meta = L.load('edge_audited')
F = L.folds(y, g); tr, te = F[0]; fi, va, ca = L.inner_split(tr, y, g, 0.1, 0)
prep = L.Prep().fit(Xn[fi], Xc[fi]); Z = np.concatenate([prep.num(Xn[fi]), Xc[fi].astype(float)], 1)
names = meta['num_features'] + meta['cat_features']; seg = np.array(meta['segments'])
idx = np.where(seg == 'TCP')[0]
rng = np.random.default_rng(0); S = Z[rng.choice(len(Z), 20000, replace=False)][:, idx]
R = np.abs(np.nan_to_num(spearmanr(S).statistic)); np.fill_diagonal(R, 1)
sid = np.array([list(dict.fromkeys(meta['segments'])).index(s) for s in meta['segments']])
order = L.seriation_order(Z, sid, seed=0); o = [list(idx).index(j) for j in order if j in idx]
fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.05), gridspec_kw={'wspace': 1.15})
for ax, ordr, t in [(axes[0], list(range(len(idx))), '(a) File order'), (axes[1], o, '(b) Fiedler order')]:
    im = ax.imshow(R[np.ix_(ordr, ordr)], cmap='Greys', vmin=0, vmax=1)
    lab = [names[idx[k]].replace('tcp.', '') for k in ordr]
    ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=90, fontsize=4.8); ax.set_yticklabels(lab, fontsize=4.8); ax.set_title(t, fontsize=7)
    for s in ax.spines.values(): s.set_visible(True)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03); cb.ax.tick_params(labelsize=5.5); cb.set_label('|Spearman|', fontsize=6)
os.makedirs('figs', exist_ok=True); fig.savefig('figs/fig_seriation.png', dpi=400, bbox_inches='tight'); plt.close(fig)
print([names[idx[k]] for k in o])
