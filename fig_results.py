"""Result figures (grayscale, column width) built from aggregated JSON files."""
import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'mathtext.fontset': 'dejavuserif'})
GR = ['#1a1a1a', '#555555', '#8c8c8c', '#bdbdbd', '#e0e0e0', '#ffffff']
HATCH = ['', '////', '....', 'xxxx', '\\\\\\\\', '']
NAMES = {'lr': 'LR', 'rf': 'RF', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'mlp': 'MLP', 'cnn1d': '1D-CNN',
         'cnnbigru': 'CNN-BiGRU', 'inc2d': 'Inception-2D', 'inc1d': 'Inception-1D', 'ftt+ple': 'FT-Transformer',
         'psi+ple': 'PSI-Net'}
OUT = 'figs'


def fig_main(agg, models, path):
    """Macro-F1 with group-bootstrap 95% CI for both datasets, plus per-fold dots."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), sharey=False)
    for ax, key, title in zip(axes, ['edge_audited|group', 'wustl|group'], ['(a) Edge-IIoTset', '(b) WUSTL-IIoT-2021']):
        rows = agg.get(key, {}); ms = [m for m in models if m in rows]
        x = np.arange(len(ms))
        v = [rows[m]['macro_f1'] for m in ms]; lo = [rows[m]['mf1_ci'][0] for m in ms]; hi = [rows[m]['mf1_ci'][1] for m in ms]
        cols = ['#2b2b2b' if m == 'psi+ple' else ('#7a7a7a' if m in ('lgbm', 'xgb', 'rf', 'lr') else '#c8c8c8') for m in ms]
        ax.bar(x, v, 0.62, color=cols, edgecolor='black', lw=0.5,
               yerr=[np.array(v) - lo, np.array(hi) - v], capsize=1.8, error_kw=dict(lw=0.6))
        for i, m in enumerate(ms):
            f = rows[m]['fold_mf1']; ax.scatter(np.full(len(f), i) + np.linspace(-0.18, 0.18, len(f)), f, s=4, c='white', edgecolors='black', linewidths=0.4, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in ms], rotation=55, ha='right')
        ax.set_ylabel('Macro-F1'); ax.set_title(title, fontsize=7.5)
        lowv = min(min(lo), min(min(rows[m]['fold_mf1']) for m in ms))
        ax.set_ylim(max(0, lowv - 0.05), 1.0); ax.grid(axis='y', lw=0.3, color='#cccccc')
    fig.tight_layout(w_pad=2.0); fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_perclass(agg, models, classes, path, key='edge_audited|group'):
    rows = agg[key]; ms = [m for m in models if m in rows]
    M = np.array([rows[m]['per_class_f1'] for m in ms])
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    im = ax.imshow(M.T, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([NAMES[m] for m in ms], rotation=55, ha='right', fontsize=6.5)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([c.replace('_', ' ') for c in classes], fontsize=6.3)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(i, j, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=4.6, color='white' if M[i, j] > 0.6 else 'black')
    for s in ax.spines.values(): s.set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label('F1', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_protocol(prot, path):
    """prot: dict setting -> model -> (mean mF1, list fold mF1, mean ceiling)."""
    settings = ['ref|random', 'audited|random', 'ref|group', 'audited|group']
    labels = ['Reference\nview,\nrecord split', 'Audited\nview,\nrecord split', 'Reference\nview,\nconn. split', 'Audited\nview,\nconn. split']
    models = ['lgbm', 'psi+ple']
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    w = 0.32; x = np.arange(len(settings))
    for k, m in enumerate(models):
        v = [prot[s][m][0] if m in prot.get(s, {}) else np.nan for s in settings]
        off = (k - 0.5) * w
        ax.bar(x + off, v, w, color=['#bdbdbd', '#2b2b2b'][k], edgecolor='black', lw=0.5,
               hatch=['////', ''][k], label=NAMES[m])
        for i, s in enumerate(settings):
            if m in prot.get(s, {}):
                f = prot[s][m][1]; ax.scatter(np.full(len(f), x[i] + off), f, s=3, c='white', edgecolors='black', linewidths=0.4, zorder=3)
    ce = [prot[s]['ceiling'] for s in settings]
    for i, c in enumerate(ce):
        ax.plot([x[i] - 1.15 * w, x[i] + 1.15 * w], [c, c], color='black', lw=0.9, ls=(0, (2, 1.5)))
    ax.plot([], [], color='black', lw=0.9, ls=(0, (2, 1.5)), label='Conflict ceiling (macro-F1)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.3); ax.set_ylabel('Macro-F1')
    ax.set_ylim(0.5, 1.02); ax.grid(axis='y', lw=0.3, color='#cccccc')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=6.3, handlelength=1.6, columnspacing=1.0)
    fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_robust(rob, seg_names, path):
    """rob: model -> dict of mean robust metrics (clean, noise*, drop_*), Edge-IIoTset."""
    models = [m for m in ['lgbm', 'inc1d', 'ftt+ple', 'psi+ple'] if m in rob]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw={'width_ratios': [1, 2.1]})
    ax = axes[0]; mk = ['s', '^', 'D', 'o']
    for k, m in enumerate(models):
        xs = [0, 0.1, 0.3, 0.5]; ys = [rob[m]['clean']] + [rob[m][f'noise{s}'] for s in [0.1, 0.3, 0.5]]
        ax.plot(xs, ys, marker=mk[k], ms=3.5, lw=1.0, color=['#8c8c8c', '#5a5a5a', '#b0b0b0', 'black'][k],
                ls=['-', '--', ':', '-'][k], mfc='white' if m != 'psi+ple' else 'black', label=NAMES[m])
    ax.set_xlabel('Log-normal jitter $\\sigma$'); ax.set_ylabel('Macro-F1'); ax.grid(lw=0.3, color='#cccccc')
    ax.set_title('(a) Measurement noise', fontsize=7.5); ax.legend(fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.33), columnspacing=1.0)
    ax = axes[1]; w = 0.8 / len(models); x = np.arange(len(seg_names))
    for k, m in enumerate(models):
        d = [rob[m]['clean'] - rob[m][f'drop_{s}'] for s in seg_names]
        ax.bar(x + (k - (len(models) - 1) / 2) * w, d, w, color=['#9a9a9a', '#d0d0d0', '#ffffff', '#2b2b2b'][k],
               edgecolor='black', lw=0.5, hatch=['', '////', '....', ''][k], label=NAMES[m])
    ax.set_xticks(x); ax.set_xticklabels(seg_names, fontsize=6.5); ax.set_ylabel('Macro-F1 drop')
    ax.axhline(0, color='black', lw=0.5); ax.grid(axis='y', lw=0.3, color='#cccccc')
    ax.set_title('(b) Protocol parser removed at test time', fontsize=7.5)
    ax.legend(fontsize=6, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(w_pad=2.5); fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_ablation(abl, path):
    """abl: list of (label, mean, [fold values]) on two datasets: dict ds -> list."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharex=False)
    for ax, (ds, items) in zip(axes, abl.items()):
        y = np.arange(len(items))[::-1]
        for yi, (lab, mu, f) in zip(y, items):
            ax.barh(yi, mu, 0.6, color='#2b2b2b' if lab.startswith('PSI-Net (full') else '#c8c8c8', edgecolor='black', lw=0.5)
            ax.scatter(f, np.full(len(f), yi), s=6, c='white', edgecolors='black', linewidths=0.5, zorder=3)
            ax.text(mu + 0.004, yi, f'{mu:.3f}', va='center', fontsize=6)
        ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=6.5)
        lo = min(min(i[2]) for i in items); hi = max(max(i[2]) for i in items)
        ax.set_xlim(max(0, lo - 0.04), min(1.0, hi + 0.05)); ax.set_xlabel('Macro-F1 (folds 1 and 2)'); ax.set_title(ds, fontsize=7.5)
        ax.grid(axis='x', lw=0.3, color='#cccccc')
    fig.tight_layout(w_pad=2.0); fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_conformal(conf, path):
    """conf: dict with 'eps_curve' {method: [(eps, fur)]} and 'families' {family: {method: detect}}."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw={'width_ratios': [1, 1.9]})
    ax = axes[0]; meths = ['conformal_mahalanobis', 'mahalanobis_global', 'msp', 'energy']
    lab = {'conformal_mahalanobis': 'Mondrian conformal', 'mahalanobis_global': 'Global Mahalanobis', 'msp': 'Max softmax', 'energy': 'Energy'}
    st = {'conformal_mahalanobis': ('black', '-', 'o'), 'mahalanobis_global': ('#6e6e6e', '--', 's'), 'msp': ('#9e9e9e', ':', '^'), 'energy': ('#4a4a4a', '-.', 'D')}
    ax.plot([0, 0.11], [0, 0.11], color='#bbbbbb', lw=0.7)
    for m in meths:
        pts = conf['eps_curve'][m]; c, ls, mk = st[m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker=mk, ms=3, lw=1, mfc='white', label=lab[m])
    ax.set_xlabel('Nominal level $\\varepsilon$'); ax.set_ylabel('Worst-class false-unknown rate')
    ax.set_title('(a) Error control on known classes', fontsize=7.5); ax.legend(fontsize=5.8, loc='upper left'); ax.grid(lw=0.3, color='#cccccc')
    ax = axes[1]; fam = list(conf['families']); x = np.arange(len(fam)); w = 0.2
    for k, m in enumerate(meths):
        v = [conf['families'][f][m] for f in fam]
        ax.bar(x + (k - 1.5) * w, v, w, color=['#2b2b2b', '#8c8c8c', '#d0d0d0', '#ffffff'][k], edgecolor='black', lw=0.5,
               hatch=['', '', '////', '....'][k], label=lab[m])
    ax.set_xticks(x); ax.set_xticklabels([f.replace('_', ' ') for f in fam], rotation=30, ha='right', fontsize=6.3)
    ax.set_ylabel('Unknown-detection rate'); ax.set_ylim(0, 1.0); ax.grid(axis='y', lw=0.3, color='#cccccc')
    ax.set_title('(b) Held-out families at $\\varepsilon=0.05$', fontsize=7.5)
    ax.legend(fontsize=5.8, ncol=2, loc='upper right')
    fig.tight_layout(w_pad=2.5); fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_eff(agg, models, path):
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    rows = agg['edge_audited|group']; mk = {'mlp': 'v', 'cnn1d': '<', 'cnnbigru': '>', 'inc2d': 'p', 'inc1d': '^', 'ftt+ple': 'D', 'psi+ple': 'o'}
    for m in models:
        if m not in rows or not rows[m].get('n_params'): continue
        r = rows[m]; t = r['test_seconds'] / r['n_test'] * 5 * 1000 if r['test_seconds'] else np.nan
        ax.scatter(r['n_params'] / 1e3, r['macro_f1'], s=18 + 1.5 * t, marker=mk[m], c='black' if m == 'psi+ple' else 'white', edgecolors='black', linewidths=0.7)
        ax.annotate(NAMES[m], (r['n_params'] / 1e3, r['macro_f1']), textcoords='offset points', xytext=(5, -2), fontsize=6)
    ax.set_xscale('log'); ax.set_xlabel('Trainable parameters (thousands, log scale)'); ax.set_ylabel('Macro-F1, Edge-IIoTset')
    ax.grid(lw=0.3, color='#cccccc')
    fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_ablation1(items, path):
    """items: list of (label, mean, [fold values]); Edge-IIoTset only."""
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    y = np.arange(len(items))[::-1]
    for yi, (lab, mu, f) in zip(y, items):
        ax.barh(yi, mu, 0.62, color='#2b2b2b' if lab.startswith('PSI-Net (full') else ('#8c8c8c' if lab.startswith('Inception') else '#d0d0d0'),
                edgecolor='black', lw=0.5)
        ax.scatter(f, np.full(len(f), yi), s=7, c='white', edgecolors='black', linewidths=0.5, zorder=3)
        ax.text(max(max(f), mu) + 0.006, yi, f'{mu:.3f}', va='center', fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=6.3)
    lo = min(min(i[2]) for i in items); hi = max(max(i[2]) for i in items)
    ax.set_xlim(max(0, lo - 0.05), min(1.0, hi + 0.07)); ax.set_xlabel('Macro-F1 (bar: mean; dots: folds 1 and 2)')
    ax.grid(axis='x', lw=0.3, color='#cccccc')
    fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)


def fig_conformal2(cov, curves, path):
    """cov: {label: [(nominal, observed)]}; curves: {method: [(fur, detect)]} averaged over held-out families."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    ax = axes[0]
    ax.plot([0.88, 1.0], [0.88, 1.0], color='#9e9e9e', lw=0.7, ls='-', label='Nominal $1-\\varepsilon$')
    st = [('black', '-', 'o', 'black'), ('black', '--', 'o', 'white'), ('#6e6e6e', '-', 's', '#6e6e6e'), ('#6e6e6e', '--', 's', 'white')]
    for (lab, pts), (c, ls, mk, fc) in zip(cov.items(), st):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker=mk, ms=3.5, lw=1, mfc=fc, label=lab)
    ax.set_xlabel('Nominal coverage $1-\\varepsilon$'); ax.set_ylabel('Connection-weighted coverage')
    ax.set_title('(a) Coverage of known-class prediction sets', fontsize=7.5); ax.legend(fontsize=5.8, loc='upper left', bbox_to_anchor=(0.0, 0.72)); ax.grid(lw=0.3, color='#cccccc'); ax.set_ylim(0.45, 1.03)
    ax = axes[1]
    lab = {'conformal_pred_conn': 'Predicted-class, connection cal.', 'conformal_pred_record': 'Predicted-class, record cal.',
           'msp': 'Max. softmax', 'energy': 'Energy', 'mahalanobis_global': 'Global Mahalanobis'}
    st = {'conformal_pred_conn': ('black', '-', 'o', 'black'), 'conformal_pred_record': ('black', '--', 'o', 'white'),
          'msp': ('#7a7a7a', ':', '^', 'white'), 'energy': ('#7a7a7a', '-.', 'D', 'white'), 'mahalanobis_global': ('#a8a8a8', '-', 's', '#a8a8a8')}
    for m, pts in curves.items():
        c, ls, mk, fc = st[m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker=mk, ms=3.2, lw=1, mfc=fc, label=lab[m])
    ax.set_xlabel('False-unknown rate on known connections'); ax.set_ylabel('Detection of held-out connections')
    ax.set_title('(b) Unknown-attack trade-off (mean over families)', fontsize=7.5); ax.grid(lw=0.3, color='#cccccc')
    ax.legend(fontsize=5.8, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout(w_pad=2.5); fig.savefig(path, dpi=400, bbox_inches='tight'); plt.close(fig)
