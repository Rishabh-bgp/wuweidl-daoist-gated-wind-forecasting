import numpy as np


def regression_metrics(y, p, eps=0.5):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    ss = ((y - y.mean()) ** 2).sum()
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            # MAPE skips near-calm samples (|y| < eps m/s) to avoid division blow-up
            "MAPE_%": float(100 * np.mean(np.abs(e[np.abs(y) >= eps] / y[np.abs(y) >= eps]))),
            "R2": float(1 - (e ** 2).sum() / ss)}
