import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    # cyclical encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    return out.drop(columns=["hour", "dow"])


def make_lag_features(
    df: pd.DataFrame, target_col: str, lags: list[int]
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out


def make_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int]
) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"roll_mean_{w}"] = out[target_col].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[target_col].shift(1).rolling(w).std()
    return out


def build_ts_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    windows: list[int],
    add_time: bool,
) -> pd.DataFrame:
    out = df.copy()
    if add_time:
        out = add_time_features(out)
    out = make_lag_features(out, target_col, lags)
    out = make_rolling_features(out, target_col, windows)
    out = out.dropna()
    return out
