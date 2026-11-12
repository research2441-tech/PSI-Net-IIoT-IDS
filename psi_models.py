"""Flax models: PSI-Net (proposed) and deep baselines, plus a generic trainer."""
import time, numpy as np, jax, jax.numpy as jnp, optax
import flax.linen as nn
from functools import partial


# ----------------------------------------------------------------- tokenizers
class FieldTokenizer(nn.Module):
    n_num: int
    ncat: tuple
    C: int
    ple: int = 0

    @nn.compact
    def __call__(self, xn, xc):
        w = self.param('w', nn.initializers.normal(0.5), (self.n_num, self.C))
        b = self.param('b', nn.initializers.normal(0.1), (self.n_num, self.C))
        if self.ple:
            B = xn.shape[0]
            xp = xn[:, self.n_num:].reshape(B, self.n_num, self.ple); xn = xn[:, :self.n_num]
            Wp = self.param('wp', nn.initializers.normal(0.3), (self.n_num, self.ple, self.C))
            toks = [xn[..., None] * w + b + jnp.einsum('bjt,jtc->bjc', xp, Wp)]
        else:
            toks = [xn[..., None] * w + b]
        for j, k in enumerate(self.ncat):
            toks.append(nn.Embed(k, self.C, name=f'emb{j}')(xc[:, j])[:, None, :])
        return jnp.concatenate(toks, 1)  # (B, L, C) in canonical field order


# ----------------------------------------------------------------- PSI-Net
class MaskedConv(nn.Module):
    """1-D convolution over the field axis whose taps never cross a segment boundary."""
    k: int
    out: int
    masks: tuple   # tuple of (offset, mask array as tuple) built on host
    use_mask: bool = True

    @nn.compact
    def __call__(self, h):
        B, L, C = h.shape
        r = self.k // 2; cols = []
        for o in range(-r, r + 1):
            sh = jnp.roll(h, -o, axis=1)
            m = jnp.asarray(dict(self.masks)[o] if self.use_mask else dict(self.masks)[('b', o)], jnp.float32)
            cols.append(sh * m[None, :, None])
        return nn.Dense(self.out)(jnp.concatenate(cols, -1))


class PSIBlock(nn.Module):
    C: int
    bw: int
    seg: tuple          # segment id per position (after ordering)
    n_seg: int
    masks: tuple
    use_mask: bool = True
    use_seg: bool = True
    use_cross: bool = True

    @nn.compact
    def __call__(self, h, pres):
        seg = jnp.asarray(self.seg)
        P = jax.nn.one_hot(seg, self.n_seg)                                  # (L, G)
        br = [nn.Dense(self.bw, name='b1')(h),
              MaskedConv(3, self.bw, self.masks, self.use_mask, name='b3')(h),
              MaskedConv(5, self.bw, self.masks, self.use_mask, name='b5')(h)]
        s = jnp.einsum('blc,lg->bgc', h, P) / P.sum(0)[None, :, None]       # segment summaries
        if self.use_seg:
            br.append(jnp.einsum('bgc,lg->blc', nn.Dense(self.bw, name='bS')(s), P))
        if self.use_cross:
            sw = pres[..., None]                                             # presence gate
            c = (s * sw).sum(1) / jnp.maximum(sw.sum(1), 1.0)
            c = nn.Dense(self.bw, name='bX')(c)
            br.append(jnp.broadcast_to(c[:, None, :], (h.shape[0], h.shape[1], self.bw)))
        z = nn.Dense(self.C, name='proj')(nn.gelu(jnp.concatenate(br, -1)))
        return nn.LayerNorm()(h + z)


class PSINet(nn.Module):
    n_num: int
    ncat: tuple
    order: tuple
    seg: tuple
    n_seg: int
    n_cls: int
    masks: tuple
    C: int = 48
    bw: int = 16
    depth: int = 2
    use_mask: bool = True
    use_seg: bool = True
    use_cross: bool = True
    ple: int = 0

    @nn.compact
    def __call__(self, xn, xc, pres, train=False, return_emb=False):
        h = FieldTokenizer(self.n_num, self.ncat, self.C, self.ple)(xn, xc)[:, jnp.asarray(self.order), :]
        for i in range(self.depth):
            h = PSIBlock(self.C, self.bw, self.seg, self.n_seg, self.masks, self.use_mask,
                         self.use_seg, self.use_cross, name=f'blk{i}')(h, pres)
        P = jax.nn.one_hot(jnp.asarray(self.seg), self.n_seg)
        wpos = jnp.einsum('bg,lg->bl', pres, P) + 1e-3                       # tokens of present segments
        avg = (h * wpos[..., None]).sum(1) / wpos.sum(1, keepdims=True)
        e = nn.gelu(nn.Dense(64, name='emb')(jnp.concatenate([avg, h.max(1)], -1)))
        e = nn.Dropout(0.1, deterministic=not train)(e)
        out = nn.Dense(self.n_cls, name='head')(e)
        return (out, e) if return_emb else out


def build_masks(seg, kmax=5):
    seg = np.asarray(seg); L = len(seg); masks = []
    for o in range(-(kmax // 2), kmax // 2 + 1):
        j = np.arange(L) + o
        inb = (j >= 0) & (j < L)
        same = inb & (seg[np.clip(j, 0, L - 1)] == seg)
        masks.append((o, tuple(same.astype(np.float32))))
        masks.append((('b', o), tuple(inb.astype(np.float32))))
    return tuple(masks)


# ----------------------------------------------------------------- baselines
class MLP(nn.Module):
    n_cls: int

    @nn.compact
    def __call__(self, x, train=False, return_emb=False):
        h = nn.relu(nn.Dense(256)(x)); h = nn.Dropout(0.2, deterministic=not train)(h)
        e = nn.relu(nn.Dense(128)(h)); e = nn.Dropout(0.2, deterministic=not train)(e)
        out = nn.Dense(self.n_cls)(e)
        return (out, e) if return_emb else out


class CNN1D(nn.Module):
    n_cls: int

    @nn.compact
    def __call__(self, x, train=False, return_emb=False):
        h = x[..., None]
        h = nn.relu(nn.Conv(32, (3,), padding='SAME')(h)); h = nn.relu(nn.Conv(64, (3,), padding='SAME')(h))
        h = nn.max_pool(h, (2,), (2,)); h = nn.relu(nn.Conv(64, (3,), padding='SAME')(h))
        e = nn.relu(nn.Dense(64)(h.mean(1)))
        out = nn.Dense(self.n_cls)(e)
        return (out, e) if return_emb else out


class CNNBiGRU(nn.Module):
    n_cls: int

    @nn.compact
    def __call__(self, x, train=False, return_emb=False):
        h = nn.relu(nn.Conv(48, (3,), padding='SAME')(x[..., None]))
        h = nn.max_pool(h, (2,), (2,))
        h = nn.Bidirectional(nn.RNN(nn.GRUCell(32)), nn.RNN(nn.GRUCell(32)))(h)
        e = nn.relu(nn.Dense(64)(jnp.concatenate([h.mean(1), h[:, -1]], -1)))
        out = nn.Dense(self.n_cls)(e)
        return (out, e) if return_emb else out


class Inception1DModule(nn.Module):
    f: int = 24

    @nn.compact
    def __call__(self, h):
        b1 = nn.Conv(self.f, (1,))(h)
        b3 = nn.Conv(self.f, (3,), padding='SAME')(nn.relu(nn.Conv(self.f, (1,))(h)))
        b5 = nn.Conv(self.f, (5,), padding='SAME')(nn.relu(nn.Conv(self.f, (1,))(h)))
        bp = nn.Conv(self.f, (1,))(nn.max_pool(h, (3,), (1,), padding='SAME'))
        return nn.relu(jnp.concatenate([b1, b3, b5, bp], -1))


class Inception1D(nn.Module):
    """GoogLeNet/InceptionTime-style 1-D Inception CNN on scalar fields (vanilla baseline)."""
    n_cls: int
    order: tuple = ()

    @nn.compact
    def __call__(self, x, train=False, return_emb=False):
        if self.order: x = x[:, jnp.asarray(self.order)]
        h = nn.relu(nn.Conv(32, (3,), padding='SAME')(x[..., None]))
        for i in range(2):
            h = nn.LayerNorm()(Inception1DModule(24)(h))
        e = nn.relu(nn.Dense(64)(jnp.concatenate([h.mean(1), h.max(1)], -1)))
        e = nn.Dropout(0.1, deterministic=not train)(e)
        out = nn.Dense(self.n_cls)(e)
        return (out, e) if return_emb else out


class Inception2D(nn.Module):
    """Tabular-to-image Inception CNN: fields zero-padded to a square grid."""
    n_cls: int
    side: int = 7

    @nn.compact
    def __call__(self, x, train=False, return_emb=False):
        B, d = x.shape; s = self.side
        img = jnp.pad(x, ((0, 0), (0, s * s - d))).reshape(B, s, s, 1)
        h = nn.relu(nn.Conv(32, (3, 3), padding='SAME')(img))
        for i in range(2):
            b1 = nn.Conv(16, (1, 1))(h)
            b3 = nn.Conv(16, (3, 3), padding='SAME')(nn.relu(nn.Conv(16, (1, 1))(h)))
            b5 = nn.Conv(16, (5, 5), padding='SAME')(nn.relu(nn.Conv(16, (1, 1))(h)))
            bp = nn.Conv(16, (1, 1))(nn.max_pool(h, (3, 3), (1, 1), padding='SAME'))
            h = nn.LayerNorm()(nn.relu(jnp.concatenate([b1, b3, b5, bp], -1)))
        e = nn.relu(nn.Dense(64)(jnp.concatenate([h.mean((1, 2)), h.max((1, 2))], -1)))
        out = nn.Dense(self.n_cls)(e)
        return (out, e) if return_emb else out


class FTTransformer(nn.Module):
    n_num: int
    ncat: tuple
    n_cls: int
    C: int = 32
    depth: int = 2
    ple: int = 0

    @nn.compact
    def __call__(self, xn, xc, pres=None, train=False, return_emb=False):
        h = FieldTokenizer(self.n_num, self.ncat, self.C, self.ple)(xn, xc)
        cls = self.param('cls', nn.initializers.normal(0.1), (1, 1, self.C))
        h = jnp.concatenate([jnp.broadcast_to(cls, (h.shape[0], 1, self.C)), h], 1)
        for i in range(self.depth):
            a = nn.MultiHeadDotProductAttention(4, qkv_features=self.C)(nn.LayerNorm()(h))
            h = h + a
            h = h + nn.Dense(self.C)(nn.gelu(nn.Dense(2 * self.C)(nn.LayerNorm()(h))))
        e = nn.gelu(nn.Dense(64)(nn.LayerNorm()(h[:, 0])))
        out = nn.Dense(self.n_cls)(e)
        return (out, e) if return_emb else out


class TokenInception(nn.Module):
    """Ablation: vanilla Inception mixing (unmasked, no segment/cross branches) on field tokens."""
    n_num: int
    ncat: tuple
    order: tuple
    n_cls: int
    C: int = 48
    ple: int = 0

    @nn.compact
    def __call__(self, xn, xc, pres=None, train=False, return_emb=False):
        h = FieldTokenizer(self.n_num, self.ncat, self.C, self.ple)(xn, xc)[:, jnp.asarray(self.order), :]
        for i in range(2):
            b1 = nn.Conv(16, (1,))(h); b3 = nn.Conv(16, (3,), padding='SAME')(h)
            b5 = nn.Conv(16, (5,), padding='SAME')(h)
            bp = nn.Conv(16, (1,))(nn.max_pool(h, (3,), (1,), padding='SAME'))
            h = nn.LayerNorm()(h + nn.Dense(self.C)(nn.gelu(jnp.concatenate([b1, b3, b5, bp], -1))))
        e = nn.gelu(nn.Dense(64)(jnp.concatenate([h.mean(1), h.max(1)], -1)))
        out = nn.Dense(self.n_cls)(e)
        return (out, e) if return_emb else out


# ----------------------------------------------------------------- trainer
def train_model(model, inputs_fit, Q_fit, w_fit, inputs_val, y_val, n_cls, log_prior=None, tau=0.0,
                epochs=25, bs=1024, lr=3e-3, wd=1e-4, patience=5, seed=0, verbose=False, max_seconds=1e9):
    """inputs_*: tuple of arrays. Q_fit: soft targets (N, K); w_fit: per-sample weights.
    tau>0 applies logit adjustment with log class priors during training."""
    from sklearn.metrics import f1_score
    key = jax.random.PRNGKey(seed)
    N = len(Q_fit)
    params = model.init({'params': key, 'dropout': key}, *[a[:2] for a in inputs_fit])['params']
    steps = epochs * max(1, N // bs)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(300, steps // 10), steps)
    tx = optax.adamw(sched, weight_decay=wd)
    opt = tx.init(params)
    adj = jnp.asarray(tau * log_prior if (log_prior is not None and tau > 0) else np.zeros(n_cls), jnp.float32)

    @jax.jit
    def step(params, opt, batch, q, w, k):
        def loss_fn(p):
            logits = model.apply({'params': p}, *batch, train=True, rngs={'dropout': k}) + adj
            l = -(q * jax.nn.log_softmax(logits)).sum(-1)
            return (l * w).sum() / w.sum()
        l, g = jax.value_and_grad(loss_fn)(params)
        upd, opt = tx.update(g, opt, params)
        return optax.apply_updates(params, upd), opt, l

    @jax.jit
    def fwd(params, batch):
        return jax.nn.softmax(model.apply({'params': params}, *batch, train=False), -1)

    def predict(params, inputs, bsz=4096):
        out = []
        for i in range(0, len(inputs[0]), bsz):
            out.append(np.asarray(fwd(params, tuple(a[i:i + bsz] for a in inputs))))
        return np.concatenate(out)

    best, best_p, bad, hist, t0 = -1, params, 0, [], time.time()
    rng = np.random.default_rng(seed)
    uniform = np.allclose(w_fit, w_fit[0])
    pw = None if uniform else w_fit / w_fit.sum()
    ones = np.ones(bs, np.float32)
    for ep in range(epochs):
        # non-uniform weights are realised by importance sampling (unit loss weights),
        # which keeps the gradient variance of the duplicate-compressed set comparable to row sampling
        perm = rng.permutation(N) if uniform else rng.choice(N, size=N, replace=True, p=pw)
        for i in range(0, N - bs + 1, bs):
            b = perm[i:i + bs]; key, k = jax.random.split(key)
            params, opt, l = step(params, opt, tuple(a[b] for a in inputs_fit), Q_fit[b], ones, k)
        pv = predict(params, inputs_val)
        f1 = f1_score(y_val, pv.argmax(1), average='macro', labels=np.unique(y_val)); hist.append(float(f1))
        if verbose: print(f'ep{ep} loss {float(l):.4f} valF1 {f1:.4f} {time.time()-t0:.0f}s', flush=True)
        if f1 > best + 1e-4: best, best_p, bad = f1, params, 0
        else:
            bad += 1
            if bad >= patience: break
        if time.time() - t0 > max_seconds: break
    return best_p, (lambda inputs: predict(best_p, inputs)), dict(val_hist=hist, epochs_run=len(hist),
                                                                   train_seconds=time.time() - t0)


def embed_fn(model, params):
    @jax.jit
    def f(batch):
        return model.apply({'params': params}, *batch, train=False, return_emb=True)

    def run(inputs, bsz=4096):
        L, E = [], []
        for i in range(0, len(inputs[0]), bsz):
            l, e = f(tuple(a[i:i + bsz] for a in inputs)); L.append(np.asarray(l)); E.append(np.asarray(e))
        return np.concatenate(L), np.concatenate(E)
    return run


def n_params(params):
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))
