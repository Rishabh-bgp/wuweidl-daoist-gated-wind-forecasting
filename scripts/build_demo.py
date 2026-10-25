"""Train the small browser-demo model on the committed synthetic CSV and export it to demo/data.js.

The demo model is a deliberately small, JavaScript-friendly cousin of the full method in src/
(see README). It predicts the *change* from the last reading (residual on persistence).
Usage:  python scripts/build_demo.py      (writes demo/data.js; deterministic, ~1 min on CPU)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.demo_model import VARIANT, LOOKBACK, forward_np  # noqa: E402

CSV = ROOT / "data" / "synthetic" / ("synthetic_sites.csv" if VARIANT == "anfis" else "synthetic_scada.csv")


def load_series():
    df = pd.read_csv(CSV)
    if "site" in df.columns:  # multi-site file: forecast the first site
        df = df[df["site"] == sorted(df["site"].unique())[0]]
    return df["wind_speed"].to_numpy("float64")


def windows(s):
    X = np.stack([s[i - LOOKBACK:i] for i in range(LOOKBACK, len(s))]); y = s[LOOKBACK:]
    return X, y


class TorchModel(torch.nn.Module):
    """Same maths as demo_model.forward_np, written in torch for training."""
    def __init__(self, P):
        super().__init__()
        self.p = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for k, v in P.items()})

    def forward(self, Xn, Iu):
        from src.demo_model import forward_generic
        return forward_generic({k: v for k, v in self.p.items()}, Xn, Iu, torch)


def main():
    from src.demo_model import init_params, features
    torch.manual_seed(0); rng = np.random.default_rng(0)
    s = load_series(); X, y = windows(s); n = len(X); a, b = int(n * .7), int(n * .85)
    mu, sd = float(X[:a].mean()), float(X[:a].std())
    d = y - X[:, -1]; dsd = float(d[:a].std())
    Xn, Iu = features(X, mu, sd, np)
    m = TorchModel(init_params(rng)); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    Xt, It, dt = T(Xn[:a]), T(Iu[:a]), T(d[:a] / dsd); Xv, Iv, dv = T(Xn[a:b]), T(Iu[a:b]), T(d[a:b] / dsd)
    best, state = np.inf, None
    for ep in range(300):
        opt.zero_grad(); loss = torch.mean((m(Xt, It) - dt) ** 2); loss.backward(); opt.step()
        with torch.no_grad(): v = float(torch.mean((m(Xv, Iv) - dv) ** 2))
        if v < best: best, state = v, {k: p.detach().numpy().copy() for k, p in m.p.items()}
    P = {k: v.tolist() for k, v in state.items()}
    W = dict(variant=VARIANT, lookback=LOOKBACK, mu=mu, sd=sd, dsd=dsd, params=P)
    pred = X[b:, -1] + forward_np(W, X[b:]) ; yt = y[b:]; pers = X[b:, -1]
    def met(p):
        e = yt - p; return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                                R2=float(1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum()))
    out = dict(weights=W, test_series=s[LOOKBACK + b - LOOKBACK:].round(4).tolist(),
               metrics=dict(model=met(pred), persistence=met(pers)))
    (ROOT / "demo").mkdir(exist_ok=True)
    (ROOT / "demo" / "data.js").write_text("var DEMO = " + json.dumps(out) + ";\nif (typeof module !== 'undefined') module.exports = DEMO; else window.DEMO = DEMO;\n")
    print(json.dumps(out["metrics"], indent=1))


if __name__ == "__main__":
    main()
