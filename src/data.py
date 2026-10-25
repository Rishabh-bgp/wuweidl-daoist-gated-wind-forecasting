"""SCADA-style CSV loader and synthetic fallback generator.

Expected CSV columns (case-insensitive; extras ignored):
  timestamp, wind_speed [m/s], and optionally wind_direction [deg],
  temperature [C], pressure [hPa], humidity [%], power [kW].
"""
import numpy as np
import pandas as pd

ALIASES = {
    "wind_speed": ["wind_speed", "windspeed", "wind speed (m/s)", "ws", "wind_speed_ms"],
    "wind_direction": ["wind_direction", "wind direction (°)", "wd", "winddirection"],
    "temperature": ["temperature", "temp", "ambient_temperature"],
    "pressure": ["pressure", "pres"],
    "humidity": ["humidity", "rh"],
    "power": ["power", "lv activepower (kw)", "active_power", "power_kw"],
}


def load_scada_csv(path, freq="1h"):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    tcol = next((cols[c] for c in cols if c in ("timestamp", "date/time", "datetime", "time", "date")), None)
    if tcol is None:
        raise ValueError("CSV needs a timestamp column")
    out["timestamp"] = pd.to_datetime(df[tcol], dayfirst=True, errors="coerce")
    for std, names in ALIASES.items():
        for n in names:
            if n in cols:
                out[std] = pd.to_numeric(df[cols[n]], errors="coerce")
                break
    if "wind_speed" not in out:
        raise ValueError("CSV needs a wind_speed column")
    out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    out = out.resample(freq).mean().interpolate(limit=6).dropna()
    return out


def synthetic_scada(n_hours=24 * 120, seed=0):
    """Synthetic hourly SCADA-like series: diurnal + seasonal cycle, AR(1) turbulence,
    occasional gust regimes, Weibull-like marginal. Purely for smoke tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    base = 7 + 1.5 * np.sin(2 * np.pi * t / 24 - 1.0) + 2.0 * np.sin(2 * np.pi * t / (24 * 365))
    ar = np.zeros(n_hours)
    regime = (np.sin(2 * np.pi * t / 240) + rng.normal(0, .3, n_hours)) > 0.8  # gusty spells
    for i in range(1, n_hours):
        ar[i] = 0.85 * ar[i - 1] + rng.normal(0, 0.6 + 0.9 * regime[i])
    ws = np.clip(base + ar, 0.2, None)
    wd = (200 + 40 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 15, n_hours)) % 360
    temp = 28 + 4 * np.sin(2 * np.pi * t / 24 - 2) + rng.normal(0, .5, n_hours)
    pres = 1008 - 0.6 * ar + rng.normal(0, .4, n_hours)
    hum = np.clip(70 - 8 * np.sin(2 * np.pi * t / 24 - 2) + rng.normal(0, 3, n_hours), 10, 100)
    power = np.clip(3600 / (1 + np.exp(-(ws - 9) * 0.9)) * (ws > 3) * (ws < 25), 0, 3600)
    idx = pd.date_range("2018-01-01", periods=n_hours, freq="h")
    return pd.DataFrame(dict(wind_speed=ws, wind_direction=wd, temperature=temp, pressure=pres,
                             humidity=hum, power=power), index=idx)


def add_features(df):
    d = df.copy()
    h, doy = d.index.hour, d.index.dayofyear
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    if "wind_direction" in d:
        r = np.deg2rad(d.pop("wind_direction"))
        d["wd_sin"], d["wd_cos"] = np.sin(r), np.cos(r)
    roll = d["wind_speed"].rolling(6, min_periods=2)
    d["turb_intensity"] = (roll.std() / roll.mean().clip(lower=0.1)).fillna(0)
    d["ws_rate3h"] = d["wind_speed"].diff(3).fillna(0)
    return d


def make_windows(df, target="wind_speed", lookback=24, horizon=1):
    X, y = df.values.astype("float32"), df[target].values.astype("float32")
    xs, ys = [], []
    for i in range(lookback, len(df) - horizon + 1):
        xs.append(X[i - lookback:i]); ys.append(y[i + horizon - 1])
    return np.stack(xs), np.array(ys)


def chrono_split(n, val=0.15, test=0.15):
    a, b = int(n * (1 - val - test)), int(n * (1 - test))
    return slice(0, a), slice(a, b), slice(b, n)


class Scaler:
    def fit(self, X):
        self.m = X.reshape(-1, X.shape[-1]).mean(0); self.s = X.reshape(-1, X.shape[-1]).std(0) + 1e-6; return self
    def transform(self, X):
        return (X - self.m) / self.s
