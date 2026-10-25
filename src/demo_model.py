"""Small JavaScript-friendly demo model (numpy forward used by tests; torch forward used for training).
The browser re-implements forward() in demo/model.js; tests/test_demo_parity.py checks they agree."""
import numpy as np

VARIANT = "gated"
LOOKBACK = 24
H = 12          # hidden units
R = 6           # fuzzy rules (anfis)


def features(X, mu, sd, xp):
    """Normalised lags and turbulence intensity Iu = std/mean of the last 6 readings."""
    Xn = (X - mu) / sd
    last6 = X[:, -6:]
    Iu = last6.std(axis=1) / (last6.mean(axis=1) + 0.1) if xp is np else None
    return Xn, Iu


def init_params(rng):
    g = lambda *s: (rng.standard_normal(s) * 0.1)
    if VARIANT == "anfis":
        return dict(c=rng.uniform(-1.5, 1.5, (R, 3)), logs=np.zeros((R, 3)), p=g(R, 3), q=g(R))
    P = dict(W1=g(H, LOOKBACK + 1), b1=np.zeros(H), W2=g(H), b2=np.zeros(1))
    if VARIANT == "attn":
        P.update(v=g(1), pos=np.zeros(LOOKBACK), W1=g(H, 3))
    if VARIANT == "gated":
        P.update(alpha=np.array([5.0]), tau=np.array([0.15]))
    return P


def forward_generic(P, Xn, Iu, xp):
    """Returns the predicted change from the last reading, in units of dsd."""
    tanh = xp.tanh; exp = xp.exp
    is_np = xp is np
    ssum = lambda a: a.sum(1, keepdims=True) if is_np else a.sum(1, keepdim=True)
    smax = lambda a: a.max(1, keepdims=True) if is_np else a.max(1, keepdim=True).values
    if VARIANT == "anfis":
        x = Xn[:, -3:]                                            # last 3 lags
        s = exp(P["logs"])
        mu = exp(-(((x[:, None, :] - P["c"][None]) ** 2) / (2 * s[None] ** 2)).sum(-1))   # (N,R)
        w = mu / (ssum(mu) + 1e-9)
        f = (x[:, None, :] * P["p"][None]).sum(-1) + P["q"][None]
        return (w * f).sum(1)
    if VARIANT == "attn":
        sc = P["v"][0] * Xn + P["pos"][None]
        a = exp(sc - smax(sc)); a = a / ssum(a)
        ctx = (a * Xn).sum(1)
        z = xp.stack([ctx, Xn[:, -1], Xn[:, -1] - Xn[:, -2]], 1)
    else:
        z = xp.concatenate([Xn, Iu[:, None]], 1)
    h = tanh(z @ P["W1"].T + P["b1"])
    o = h @ P["W2"] + P["b2"][0]
    if VARIANT == "gated":
        g = 1 / (1 + exp(P["alpha"][0] * (Iu - P["tau"][0])))  # gate closes (->persistence) in turbulence
        o = g * o
    return o


def forward_np(W, X):
    """Raw windows (N, LOOKBACK) in m/s -> predicted change in m/s."""
    X = np.asarray(X, "float64"); P = {k: np.asarray(v, "float64") for k, v in W["params"].items()}
    Xn, Iu = features(X, W["mu"], W["sd"], np)
    return forward_generic(P, Xn, Iu, np) * W["dsd"]


def attention_np(W, X):
    P = {k: np.asarray(v) for k, v in W["params"].items()}
    Xn = (np.asarray(X) - W["mu"]) / W["sd"]; sc = P["v"][0] * Xn + P["pos"][None]
    a = np.exp(sc - sc.max(1, keepdims=True)); return a / a.sum(1, keepdims=True)
