"""WuWeiDL (small CPU reference implementation).

Modules (paper section 4): Feng-Qi Turbulence Encoder (FQTE) -> dilated temporal CNN ->
Wu Wei Gating Module (per-head gates G_k and context weight w_ctx) ->
BiLSTM + gated multi-head self-attention, blended h = w*h_lstm + (1-w)*h_attn -> multi-horizon head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

REGIME_EDGES = (0.08, 0.15, 0.25)   # Iu thresholds: calm / moderate / active / storm


class FengQiEncoder(nn.Module):
    def __init__(self, n_phi, n_basis=32, d=64, n_regimes=4):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_basis, n_phi))
        self.b = nn.Parameter(torch.rand(n_basis) * 6.283)
        self.mlp = nn.Sequential(nn.Linear(n_basis, d), nn.BatchNorm1d(d), nn.ReLU(),
                                 nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())
        self.proj = nn.Linear(d, d)
        self.regime = nn.Linear(d, n_regimes)

    def forward(self, phi):                                 # phi: (B, n_phi)
        psi = torch.cos(phi @ self.omega.T + self.b)        # harmonic Feng-Qi basis
        h = self.mlp(psi)
        return self.proj(h), self.regime(h)


class TemporalCNN(nn.Module):
    def __init__(self, n_in, ch=32, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=d, padding=d) for d in dilations])

    def forward(self, x):                                   # (B, T, F) -> (B, T, ch)
        h = self.inp(x.transpose(1, 2))
        for c in self.blocks:
            h = h + F.relu(c(h))
        return h.transpose(1, 2)


class GatedMHA(nn.Module):
    """Self-attention where A' = g*A + (1-g)*A_prior per head (paper eq.), then head outputs
    are scaled by the same per-head gate G_k."""
    def __init__(self, d, heads, T):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.prior_logits = nn.Parameter(torch.zeros(heads, T, T))   # climatological prior

    def forward(self, x, g):                                # x (B,T,d), g (B,K)
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / self.dk ** .5, -1)
        Ap = torch.softmax(self.prior_logits, -1).unsqueeze(0)
        gg = g[:, :, None, None]
        A = gg * A + (1 - gg) * Ap
        out = (A @ v) * gg                                  # (B,K,T,dk)
        return self.o(out.transpose(1, 2).reshape(B, T, d)), A


class WuWeiDL(nn.Module):
    def __init__(self, n_features, n_phi, T, horizons=1, d=64, heads=8, hidden=32, gate_mode="paper"):
        super().__init__()
        self.gate_mode = gate_mode
        self.fq = FengQiEncoder(n_phi, d=d)
        self.cnn = TemporalCNN(n_features + 0, ch=d)
        self.fuse = nn.Linear(2 * d, d)
        self.Wq = nn.Linear(d, heads)                       # q_turb -> head gates
        self.Wctx = nn.Linear(d + 1, 1)
        # explicit closed-form gate from the paper: sigma(-alpha (Iu - tau) + beta * lap(v))
        self.alpha, self.beta, self.tau = (nn.Parameter(torch.tensor(v)) for v in (10.0, 1.0, 0.15))
        self.lstm = nn.LSTM(d, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(2 * hidden, d)
        self.attn = GatedMHA(d, heads, T)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, horizons))
        self.cache = {}

    def forward(self, x, phi, iu, lap):
        e_fq, regime_logits = self.fq(phi)
        f = self.cnn(x)
        f = self.fuse(torch.cat([f, e_fq.unsqueeze(1).expand_as(f)], -1))
        g_scalar = torch.sigmoid(-self.alpha * (iu - self.tau) + self.beta * lap)       # (B,)
        if self.gate_mode == "patent":       # patent draft: stable wind -> *less* attention
            g_scalar = 1 - g_scalar
        G = torch.sigmoid(self.Wq(e_fq)) * g_scalar.unsqueeze(1)                       # (B,K)
        w = torch.sigmoid(self.Wctx(torch.cat([e_fq, iu.unsqueeze(1)], 1))).squeeze(1)
        hl, _ = self.lstm(f); h_lstm = self.lstm_proj(hl[:, -1])
        ha, _ = self.attn(f, G); h_attn = ha[:, -1]
        h = w.unsqueeze(1) * h_lstm + (1 - w.unsqueeze(1)) * h_attn
        self.cache = dict(G=G, w=w, regime_logits=regime_logits)
        out = self.head(h)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def regime_labels(iu):
    return torch.bucketize(iu, torch.tensor(REGIME_EDGES))
