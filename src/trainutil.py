import numpy as np
import torch


def fit(model, Xtr, ytr, Xva, yva, epochs=15, lr=2e-3, bs=128, loss_fn=None, patience=4, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or torch.nn.MSELoss()
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva, yva = torch.tensor(Xva), torch.tensor(yva)
    best, state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xva), yva))
        if v < best:
            best, wait = v, 0; state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(state); return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X))
    return out.numpy()
