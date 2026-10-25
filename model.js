// Browser re-implementation of src/demo_model.py forward(). Pure JS, no libraries.
// tests/test_demo_parity.py runs this file with node and compares against Python.
(function (root) {
  function predictChange(W, win) {           // win: array of LOOKBACK raw wind speeds (m/s)
    const P = W.params, L = W.lookback, xn = win.map(v => (v - W.mu) / W.sd);
    const last6 = win.slice(-6), m6 = last6.reduce((a, b) => a + b, 0) / 6;
    const iu = Math.sqrt(last6.reduce((a, b) => a + (b - m6) ** 2, 0) / 6) / (m6 + 0.1);
    let o, attn = null;
    if (W.variant === 'anfis') {
      const x = xn.slice(-3); let ws = 0, num = 0;
      for (let r = 0; r < P.c.length; r++) {
        let e = 0, f = P.q[r];
        for (let j = 0; j < 3; j++) { const s = Math.exp(P.logs[r][j]); e += (x[j] - P.c[r][j]) ** 2 / (2 * s * s); f += x[j] * P.p[r][j]; }
        const mu = Math.exp(-e); ws += mu; num += mu * f;
      }
      o = num / (ws + 1e-9);
    } else {
      let z;
      if (W.variant === 'attn') {
        const sc = xn.map((v, i) => P.v[0] * v + P.pos[i]), mx = Math.max(...sc);
        const ex = sc.map(s => Math.exp(s - mx)), tot = ex.reduce((a, b) => a + b, 0);
        attn = ex.map(e => e / tot);
        z = [attn.reduce((a, w, i) => a + w * xn[i], 0), xn[L - 1], xn[L - 1] - xn[L - 2]];
      } else z = xn.concat([iu]);
      o = P.b2[0];
      for (let h = 0; h < P.W1.length; h++) {
        let a = P.b1[h]; for (let j = 0; j < z.length; j++) a += P.W1[h][j] * z[j];
        o += P.W2[h] * Math.tanh(a);
      }
      if (W.variant === 'gated') o *= 1 / (1 + Math.exp(P.alpha[0] * (iu - P.tau[0])));
    }
    return { change: o * W.dsd, iu: iu, attn: attn,
             gate: W.variant === 'gated' ? 1 / (1 + Math.exp(P.alpha[0] * (iu - P.tau[0]))) : null };
  }
  const api = { predictChange };
  if (typeof module !== 'undefined') module.exports = api; else root.DemoModel = api;
})(this);
