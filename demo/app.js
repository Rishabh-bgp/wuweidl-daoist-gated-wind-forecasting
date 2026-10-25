// Live demo UI: chart (canvas), sliders, metrics. Uses window.DEMO (data.js) and DemoModel (model.js).
(function () {
  const D = window.DEMO, W = D.weights, L = W.lookback, S = D.test_series, C = window.DEMO_CFG || {};
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = v => v.toFixed(3);
  // metrics table (computed in Python on the same test period, re-checked here in JS)
  const preds = [], pers = [], obs = [];
  for (let t = L; t < S.length; t++) { const w = S.slice(t - L, t); preds.push(w[L - 1] + DemoModel.predictChange(W, w).change); pers.push(w[L - 1]); obs.push(S[t]); }
  const met = p => { const n = obs.length, m = obs.reduce((a, b) => a + b, 0) / n; let ae = 0, se = 0, st = 0;
    for (let i = 0; i < n; i++) { const e = obs[i] - p[i]; ae += Math.abs(e); se += e * e; st += (obs[i] - m) ** 2; }
    return { MAE: ae / n, RMSE: Math.sqrt(se / n), R2: 1 - se / st }; };
  const mm = met(preds), mp = met(pers);
  $('metrics').innerHTML = [['Demo model', mm, ''], ['Persistence', mp, ' p']].map(([n, m, c]) => `<div class=grp>${n}</div><div class=seals>` +
    [['MAE', m.MAE, 'm/s'], ['RMSE', m.RMSE, 'm/s'], ['R²', m.R2, '']].map(([k, v, u]) => `<div class="seal${c}">${fmt(v)}<small>${k} ${u}</small></div>`).join('') + '</div>').join('');
  $('verdict').textContent = mm.RMSE < mp.RMSE ? `On this synthetic test period the demo model beats persistence (RMSE ${fmt(mm.RMSE)} vs ${fmt(mp.RMSE)} m/s).`
    : `On this synthetic test period the demo model does NOT beat persistence (RMSE ${fmt(mm.RMSE)} vs ${fmt(mp.RMSE)} m/s).`;
  const start = $('start'), gust = $('gust'); start.max = Math.floor((obs.length - 168) / 24);
  function draw() {
    const d0 = +start.value * 24, g = +gust.value; $('startv').textContent = start.value; $('gustv').textContent = g.toFixed(1);
    const cv = $('chart'), ctx = cv.getContext('2d'), Wd = cv.width = cv.clientWidth * devicePixelRatio, Hd = cv.height = cv.clientHeight * devicePixelRatio;
    const o = obs.slice(d0, d0 + 168), p = preds.slice(d0, d0 + 168), q = pers.slice(d0, d0 + 168);
    const lo = Math.min(...o, ...p) - 1, hi = Math.max(...o, ...p) + 1, pad = 40 * devicePixelRatio;
    const X = i => pad + i / 167 * (Wd - 2 * pad), Y = v => Hd - pad - (v - lo) / (hi - lo) * (Hd - 2 * pad);
    ctx.clearRect(0, 0, Wd, Hd); ctx.font = `${11 * devicePixelRatio}px sans-serif`; ctx.fillStyle = css('--muted');
    for (let v = Math.ceil(lo); v <= hi; v += 2) { ctx.fillText(v + ' m/s', 2, Y(v)); ctx.strokeStyle = css('--grid'); ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(Wd - pad, Y(v)); ctx.stroke(); }
    [[o, css('--obs'), 2.2], [q, css('--pers'), 1.2], [p, css('--model'), 1.8]].forEach(([s, c, lw]) => {
      ctx.strokeStyle = c; ctx.lineWidth = lw * devicePixelRatio; ctx.beginPath(); s.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))); ctx.stroke(); });
    // what-if: next-hour forecast at the end of the window, with a gust added to the latest reading
    const t = L + d0 + 167, w = S.slice(t - L, t).slice(); w[L - 1] += g; const r = DemoModel.predictChange(W, w);
    $('whatif').innerHTML = `Latest reading: <b>${w[L - 1].toFixed(2)} m/s</b> → next hour: model <b>${(w[L - 1] + r.change).toFixed(2)} m/s</b>, persistence <b>${w[L - 1].toFixed(2)} m/s</b>. Turbulence intensity I<sub>u</sub> = ${r.iu.toFixed(3)}` +
      (r.gate !== null ? `; wu-wei gate = ${r.gate.toFixed(2)} (0 = defer to persistence)` : '') + '.';
    if (C.extra) C.extra(r, w);
  }
  start.oninput = gust.oninput = draw; window.onresize = draw; draw();
})();
