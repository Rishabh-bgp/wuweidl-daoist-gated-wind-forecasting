"""Write a small synthetic SCADA-style CSV to data/synthetic/synthetic_scada.csv."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import synthetic_scada

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hours", type=int, default=24 * 120)
    ap.add_argument("--out", default="data/synthetic/synthetic_scada.csv"); a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df = synthetic_scada(a.hours); df.index.name = "timestamp"; df.to_csv(a.out)
    print(f"wrote {a.out} ({len(df)} rows)")
