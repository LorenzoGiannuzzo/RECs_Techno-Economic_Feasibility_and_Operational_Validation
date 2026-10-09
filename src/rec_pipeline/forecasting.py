"""Day-ahead input set and target of the MLP load forecaster (Section 2.4, Eqs. 34-37)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from . import config as C

FEATURES = ["Hour (sine)", "Hour (cosine)", "Day of the week (sine)", "Day of the week (cosine)", "Non-working day",
            "Load 24 h before", "Load 168 h before", "Load 48 h before", "Temperature change over 24 h",
            "Outdoor temperature"]


def lag(v, k):
    return np.r_[np.full(k, np.nan), v[:-k]]


@dataclass
class ForecastData:
    """Inputs of the forecaster on the measurement calendar.

    y: residential sample [kWh] with the zeros interpolated; X: raw inputs in the order of FEATURES; r: target, ratio
    between the hourly load and the mean load of the previous day; day_mean: mean hourly load of the previous day
    [kWh]; valid: hours for which all the lags are available.
    """
    ts: pd.DatetimeIndex
    y: np.ndarray
    X: np.ndarray
    r: np.ndarray
    day_mean: np.ndarray
    valid: np.ndarray


def build(inp):
    """Day-ahead information set: the forecast of day D is issued at the end of day D-1, so that only lags of at least
    24 h are used, and the target is normalized by the mean load of day D-1."""
    ts = inp.ts_meas
    y = pd.Series(inp.res_sample.copy())
    y[y == 0] = np.nan
    y = y.interpolate().bfill().ffill().values
    h, d = ts.hour, ts.dayofweek
    off = ((d >= 5) | ts.normalize().isin(pd.to_datetime(C.HOLIDAYS))).astype(float)
    t_ext = inp.temperature
    day_mean = lag(pd.Series(y).groupby(np.arange(C.HOURS) // 24).transform("mean").values, 24)
    X = np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / 7),
              np.cos(2 * np.pi * d / 7), off, lag(y, 24) / day_mean, lag(y, 168) / day_mean,
              lag(y, 48) / day_mean, t_ext - lag(t_ext, 24), t_ext]
    return ForecastData(ts=ts, y=y, X=X, r=y / day_mean, day_mean=day_mean, valid=~np.isnan(X).any(1))


def make_mlp():
    """MLP regressor of Section 2.4 (architecture 10-256-128-64-1, Adam, L2 penalty 1e-3); early stopping is handled
    by fit_chrono on a chronological validation set."""
    return MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam", alpha=1e-3,
                        learning_rate_init=1e-3, random_state=C.SEED)


def fit_chrono(X, y, val_fraction=0.1, max_epochs=400, patience=20, tol=1e-4):
    """Training with early stopping on the last part of the training window, in chronological order.

    The rows of X are in time order: the first (1 - val_fraction) of them train the network, the remaining ones form
    the validation set; training stops after `patience` epochs without an improvement of the validation loss larger
    than `tol`, and the weights of the best epoch are restored.
    """
    k = int(round(len(y) * (1 - val_fraction)))
    Xt, yt, Xv, yv = X[:k], y[:k], X[k:], y[k:]
    mlp = make_mlp()
    best, best_loss, wait, best_ep = None, np.inf, 0, 0
    for ep in range(max_epochs):
        mlp.partial_fit(Xt, yt)
        loss = float(np.mean((mlp.predict(Xv) - yv) ** 2))
        if loss < best_loss - tol:
            best_loss, wait, best_ep = loss, 0, ep + 1
            best = ([w.copy() for w in mlp.coefs_], [b.copy() for b in mlp.intercepts_])
        else:
            wait += 1
            if wait >= patience:
                break
    mlp.coefs_, mlp.intercepts_ = best
    mlp.best_epoch_ = best_ep
    return mlp
