"""Train WuWeiDL + baselines (persistence, plain CNN-BiLSTM) and run a small ablation."""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from .data import load_scada_csv, synthetic_scada, add_features, make_windows, chrono_split, Scaler
from .metrics import regression_metrics
from .model import WuWeiDL, regime_labels

LAMBDA_AUX, GAMMA_SPARSE = 0.1, 0.01


def turbulence_proxies(win_ws):
    """phi(t) from the raw wind-speed window (B,T). The paper's Richardson number, z/L and
    vertical shear need met-mast data we do not have, so proxies are used (see README)."""
    s6, m6 = win_ws[:, -6:].std(1), win_ws[:, -6:].mean(1).clip(min=0.1)
    s24, m24 = win_ws.std(1), win_ws.mean(1).clip(min=0.1)
    iu = s6 / m6
    lap = win_ws[:, -1] - 2 * win_ws[:, -2] + win_ws[:, -3]
    dv = win_ws[:, -1] - win_ws[:, -4]
    phi = np.stack([iu, s24 / m24, lap, dv, m6 - m24], 1)
    return phi.astype("float32"), iu.astype("float32"), lap.astype("float32")


def run(model, data, epochs, lr=2e-3, bs=128, ablate=()):
    (Xtr, Ptr, Itr, Ltr, ytr), (Xva, Pva, Iva, Lva, yva) = data
    T = lambda *a: [torch.tensor(v) for v in a]
    Xtr, Ptr, Itr, Ltr, ytr = T(Xtr, Ptr, Itr, Ltr, ytr); Xva, Pva, Iva, Lva, yva = T(Xva, Pva, Iva, Lva, yva)
    opt = torch.optim.Adam(model.parameters(), lr=lr); best, state = np.inf, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            p = model(Xtr[b], Ptr[b], Itr[b], Ltr[b]); c = model.cache
            loss = nn.functional.mse_loss(p, ytr[b])
            if "aux" not in ablate:
                loss = loss + LAMBDA_AUX * nn.functional.cross_entropy(c["regime_logits"], regime_labels(Itr[b]))
            if "sparse" not in ablate:
                loss = loss + GAMMA_SPARSE * ((1 - c["G"]).mean(1) * Itr[b]).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(Xva, Pva, Iva, Lva), yva))
        if v < best: best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/synthetic/synthetic_scada.csv" if os.path.exists("data/synthetic/synthetic_scada.csv") else None, help="SCADA-style CSV (default: committed synthetic CSV)"); ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gate_mode", choices=["paper", "patent"], default="paper")
    ap.add_argument("--no_ablation", action="store_true"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.quick: a.epochs, a.lookback, a.no_ablation = 4, 24, True
    torch.manual_seed(0)
    raw = load_scada_csv(a.csv) if a.csv else synthetic_scada(24 * 90)
    if a.quick: raw = raw.iloc[-24 * 60:]
    df = add_features(raw)
    X, y = make_windows(df, lookback=a.lookback)
    ws_col = list(df.columns).index("wind_speed")
    phi, iu, lap = turbulence_proxies(X[:, :, ws_col])
    tr, va, te = chrono_split(len(X))
    sc = Scaler().fit(X[tr]); Xs = sc.transform(X).astype("float32")
    pm, ps = phi[tr].mean(0), phi[tr].std(0) + 1e-6; phin = ((phi - pm) / ps).astype("float32")
    last = X[:, -1, ws_col]; d = y - last   # predict the change from the last reading
    ym, ys = d[tr].mean(), d[tr].std(); yn = ((d - ym) / ys).astype("float32")
    pack = lambda s: (Xs[s], phin[s], iu[s], lap[s], yn[s])
    results = {"persistence": regression_metrics(y[te], X[te][:, -1, ws_col])}
    variants = {"WuWeiDL": {}} if a.no_ablation else {
        "WuWeiDL": {}, "no_gate (G=1)": {"nogate": True}, "no_aux_loss": {"ablate": ("aux",)},
        "no_sparse_penalty": {"ablate": ("sparse",)}}
    for name, cfg in variants.items():
        m = WuWeiDL(X.shape[2], phi.shape[1], a.lookback, gate_mode=a.gate_mode)
        if cfg.get("nogate"):
            m.alpha.data.zero_(); m.beta.data.zero_(); m.alpha.requires_grad_(False); m.beta.requires_grad_(False)
            m.tau.requires_grad_(False); m.Wq.weight.data.zero_(); m.Wq.bias.data.fill_(20.)
            m.Wq.requires_grad_(False)   # sigma(0)=0.5 scalar gate kept constant; head gates ~1
        m = run(m, (pack(tr), pack(va)), a.epochs, ablate=cfg.get("ablate", ()))
        m.eval()
        with torch.no_grad():
            p = m(*[torch.tensor(v) for v in pack(te)[:4]]).numpy() * ys + ym + last[te]
            G = m.cache["G"].mean(1).numpy()
        results[name] = regression_metrics(y[te], p)
        if name == "WuWeiDL":
            hi = iu[te] > np.median(iu[te])
            results["gate_mean_high_turbulence"] = float(G[hi].mean())
            results["gate_mean_low_turbulence"] = float(G[~hi].mean())
            results["n_params"] = sum(p.numel() for p in m.parameters())
    os.makedirs(a.out, exist_ok=True); json.dump(results, open(f"{a.out}/results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
