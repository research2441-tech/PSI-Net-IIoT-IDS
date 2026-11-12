import os
"""Architecture diagrams (grayscale) for the manuscript."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 6.6,
                     'mathtext.fontset': 'dejavuserif'})


class Box:
    def __init__(s, ax, x, y, w, h, text, fill='white', lw=1.0, ls='-', fs=6.6, bold=False):
        s.x, s.y, s.w, s.h = x, y, w, h
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                    fc=fill, ec='black', lw=lw, ls=ls))
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal',
                linespacing=1.25)

    def side(s, which):
        return {'l': (s.x - s.w / 2, s.y), 'r': (s.x + s.w / 2, s.y), 't': (s.x, s.y + s.h / 2),
                'b': (s.x, s.y - s.h / 2)}[which]


def arrow(ax, p, q, ls='-', lw=1.0, conn='arc3', head=9, color='black'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=head, lw=lw, ls=ls, color=color,
                                 connectionstyle=conn, shrinkA=1.5, shrinkB=1.5))


def poly_arrow(ax, pts, ls='-', lw=1.0, head=9):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=lw, ls=ls, solid_capstyle='butt')
    arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw, head=head)


# ------------------------------------------------------------------ Fig. 1 overall framework
fig, ax = plt.subplots(figsize=(7.16, 3.55)); ax.set_xlim(0, 14.3); ax.set_ylim(0, 7.1); ax.axis('off')
G1, G2, G3 = '#f2f2f2', '#d9d9d9', '#bfbfbf'
# stage frames
for (x0, x1, lab) in [(0.15, 4.35, 'Stage I: audited input'),
                      (4.55, 10.15, 'Stage II: PSI-Net (protocol-segmented Inception)'),
                      (10.35, 14.15, 'Stage III: calibrated decision')]:
    ax.add_patch(FancyBboxPatch((x0, 0.25), x1 - x0, 6.55, boxstyle='round,pad=0.0,rounding_size=0.12',
                                fc='none', ec='#7f7f7f', lw=0.8, ls=(0, (4, 2))))
    ax.text((x0 + x1) / 2, 6.5, lab, ha='center', va='center', fontsize=6.6, style='italic')
b_raw = Box(ax, 2.25, 5.65, 3.85, 0.85, 'Packet and flow records:\nEdge-IIoTset, WUSTL-IIoT-2021', G1)
b_aud = Box(ax, 2.25, 4.35, 3.85, 0.95, 'Identifier audit: addresses,\ntimestamps, raw seq. numbers\nand checksums removed', 'white')
b_grp = Box(ax, 2.25, 3.05, 3.6, 0.8, 'Connection-disjoint folds\n(train / val. / cal. / test)', 'white')
b_cmp = Box(ax, 2.25, 1.6, 3.6, 1.0, 'Duplicate and label-conflict\naudit: counts $n_{u,c}$ and\nfold ceiling $\\mathrm{Acc}^{\\ast}$', G2)
b_tok = Box(ax, 7.65, 5.65, 4.8, 0.85, 'Field tokenizer: linear and piecewise-\nlinear terms ($\\phi_{j,\\tau}$), embeddings $\\mathbf{E}_j$', G1)
b_ord = Box(ax, 7.65, 4.5, 4.8, 0.8, 'Protocol-segmented field order $\\pi$\n(Fiedler seriation inside each segment)', 'white')
b_blk = Box(ax, 7.65, 3.25, 4.8, 1.15, 'PS-Inception block $\\times 2$\nfield | masked 3-tap | masked 5-tap\nsegment pool | presence-gated cross', G2, lw=1.3)
b_pool = Box(ax, 7.65, 2.0, 4.8, 0.7, 'Presence-weighted pooling $\\to\\ \\mathbf{z}$', 'white')
b_loss = Box(ax, 7.9, 0.98, 3.0, 0.55, 'Cross-entropy $\\mathcal{L}(\\theta)$', G3)
b_head = Box(ax, 12.25, 5.65, 3.2, 0.8, 'Softmax head\n$p_\\theta(c\\mid\\mathbf{x})$', G1)
b_maha = Box(ax, 12.25, 4.35, 3.2, 0.8, 'Class-conditional\nMahalanobis score $a_c(\\mathbf{z})$', 'white')
b_cp = Box(ax, 12.25, 3.05, 3.2, 0.8, 'Mondrian conformal\np-values $p_c(\\mathbf{x})$', G2)
b_out = Box(ax, 12.25, 1.6, 3.2, 1.0, 'Known class $\\hat{c}$, or\nunknown-attack alert\nif $p_{\\hat{c}}(\\mathbf{x})\\leq\\varepsilon$', G3, lw=1.3)
arrow(ax, b_raw.side('b'), b_aud.side('t')); arrow(ax, b_aud.side('b'), b_grp.side('t'))
arrow(ax, b_grp.side('b'), b_cmp.side('t'))
poly_arrow(ax, [b_cmp.side('r'), (4.75, 1.6), (4.75, 5.85), (b_tok.x - b_tok.w / 2, 5.85)])
arrow(ax, b_tok.side('b'), b_ord.side('t')); arrow(ax, b_ord.side('b'), b_blk.side('t'))
arrow(ax, b_blk.side('b'), b_pool.side('t'))
arrow(ax, (b_loss.x, b_pool.y - b_pool.h / 2), b_loss.side('t'))
poly_arrow(ax, [b_pool.side('r'), (10.45, 2.0), (10.45, 5.65), b_head.side('l')])
arrow(ax, (10.45, 4.35), b_maha.side('l')); arrow(ax, b_maha.side('b'), b_cp.side('t'))
poly_arrow(ax, [b_head.side('r'), (14.05, 5.65), (14.05, 1.6), b_out.side('r')])
arrow(ax, b_cp.side('b'), b_out.side('t'))
# dashed reverse-mode gradient path: loss -> blocks -> tokenizer (training only)
poly_arrow(ax, [b_loss.side('l'), (5.0, 0.98), (5.0, 3.0), (b_blk.x - b_blk.w / 2, 3.0)], ls=(0, (3, 2)))
poly_arrow(ax, [(5.0, 3.0), (5.0, 5.45), (b_tok.x - b_tok.w / 2, 5.45)], ls=(0, (3, 2)))
ax.text(7.65, 0.5, 'dashed: reverse-mode gradient $\\nabla_\\theta\\mathcal{L}$ (training)', ha='center', va='center', fontsize=5.8)
# calibration split feeds the conformal layer
ax.text(12.25, 0.62, 'calibrated with one record\nper calibration connection', ha='center', va='center', fontsize=5.8, linespacing=1.15)
os.makedirs('figs', exist_ok=True); fig.savefig('figs/fig1_framework.png', dpi=400, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ Fig. 2 PS-Inception block
fig, ax = plt.subplots(figsize=(7.16, 3.6)); ax.set_xlim(0, 14.3); ax.set_ylim(-0.9, 6.95); ax.axis('off')
b_in = Box(ax, 1.3, 3.3, 2.1, 1.0, 'Input tokens\n$\\mathbf{H}^{(\\ell)}\\in\\mathbb{R}^{L\\times C}$', G1)
names = ['1-tap field mixing\n$\\mathbf{W}_1\\mathbf{h}_t$',
         'Masked 3-tap\n$\\Sigma_{o}\\, m_{t,o}\\mathbf{W}_o\\mathbf{h}_{t+o}$',
         'Masked 5-tap\n$|o|\\leq 2$, same segment',
         'Segment pooling\n$\\bar{\\mathbf{s}}_g=$ mean of $\\mathbf{h}_t,\\ t\\in\\mathcal{S}_g$',
         'Presence-gated cross\n$\\Sigma_g\\,\\rho_g\\bar{\\mathbf{s}}_g\\,/\\,\\Sigma_g\\,\\rho_g$']
ys = [5.75, 4.52, 3.3, 2.08, 0.85]
brs = [Box(ax, 5.0, yy, 3.5, 0.95, nm, 'white' if i < 3 else G2) for i, (yy, nm) in enumerate(zip(ys, names))]
b_cat = Box(ax, 8.55, 3.3, 1.35, 1.5, 'Concat\n$5b$ ch.\n+ GELU', G1)
b_proj = Box(ax, 10.45, 3.3, 1.35, 1.0, 'Dense\n$\\to C$', 'white')
b_add = Box(ax, 11.95, 3.3, 0.8, 0.8, '$+$', 'white', fs=12)
b_ln = Box(ax, 13.35, 3.3, 1.3, 1.0, 'Layer\nNorm\n$\\mathbf{H}^{(\\ell+1)}$', G1)
for b in brs:
    poly_arrow(ax, [b_in.side('r'), (2.75, 3.3), (2.75, b.y), b.side('l')] if abs(b.y - 3.3) > 1e-6 else [b_in.side('r'), b.side('l')])
    arrow(ax, b.side('r'), (b_cat.x - b_cat.w / 2, 3.3 + (b.y - 3.3) * 0.16))
arrow(ax, b_cat.side('r'), b_proj.side('l')); arrow(ax, b_proj.side('r'), b_add.side('l'))
arrow(ax, b_add.side('r'), b_ln.side('l'))
# residual connection over the top
poly_arrow(ax, [(b_in.x, b_in.y + b_in.h / 2), (b_in.x, 6.45), (b_add.x, 6.45), b_add.side('t')])
ax.text(7.6, 6.68, 'residual path', ha='center', fontsize=7.5, style='italic')
# presence vector enters the cross branch from below
b_rho = Box(ax, 5.0, -0.5, 3.9, 0.6, 'segment presence $\\boldsymbol{\\rho}\\in\\{0,1\\}^{G}$', G3)
arrow(ax, b_rho.side('t'), brs[4].side('b'))
ax.text(5.0, 6.42 - 0.08, '', ha='center')
os.makedirs('figs', exist_ok=True); fig.savefig('figs/fig2_psblock.png', dpi=400, bbox_inches='tight'); plt.close(fig)
print('ok')
