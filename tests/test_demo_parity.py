"""The browser demo (demo/model.js) must give the same predictions as Python (src/demo_model.py)."""
import json, shutil, subprocess
from pathlib import Path
import numpy as np, pytest
from src.demo_model import forward_np

ROOT = Path(__file__).resolve().parents[1]


def load_demo():
    txt = (ROOT / "demo" / "data.js").read_text()
    return json.loads(txt[txt.index("=") + 1: txt.index(";\n")])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python():
    D = load_demo(); W = D["weights"]; L = W["lookback"]; s = np.array(D["test_series"])
    X = np.stack([s[i - L:i] for i in range(L, len(s), 7)])
    X[::3, -1] += 2.5                                    # include 'gust' what-if inputs
    py = forward_np(W, X)
    js_code = ("const M=require(process.argv[1]+'/demo/model.js'),D=require(process.argv[1]+'/demo/data.js');"
               "const X=JSON.parse(require('fs').readFileSync(0,'utf8'));"
               "console.log(JSON.stringify(X.map(w=>M.predictChange(D.weights,w).change)));")
    out = subprocess.run(["node", "-e", js_code, str(ROOT)], input=json.dumps(X.tolist()),
                         capture_output=True, text=True, check=True).stdout
    js = np.array(json.loads(out))
    assert js.shape == py.shape and np.allclose(js, py, atol=1e-9), np.abs(js - py).max()


def test_demo_metrics_are_reported_against_persistence():
    m = load_demo()["metrics"]
    assert set(m) == {"model", "persistence"} and all(np.isfinite(v) for d in m.values() for v in d.values())
