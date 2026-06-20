from __future__ import annotations

import numpy as np
import pandas as pd


REVERSAL_QUALITY_FEATURE_COLUMNS = [
    "reversal_impulse_4",
    "reversal_impulse_8",
    "reversal_impulse_16",
    "reversal_ma20_extension",
    "reversal_ema12_extension",
    "reversal_level_distance",
    "reversal_level_touch",
    "reversal_breakout_depth",
    "reversal_close_stretch",
    "reversal_rejection_wick",
    "reversal_body_against_impulse",
    "reversal_range_ratio",
    "reversal_body_ratio",
    "reversal_volatility_ratio",
]


def add_reversal_quality_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add CUSUM-reversal quality features known at the candle close."""
    df = dataframe.copy()
    direction = df.get("event_cusum_direction", pd.Series(0, index=df.index)).astype(float)
    close = df["close"].replace(0, np.nan)

    for window in (4, 8, 16):
        source = df.get(f"momentum_{window}", df["close"].pct_change(window))
        df[f"reversal_impulse_{window}"] = direction * source

    df["reversal_ma20_extension"] = direction * ((df["close"] - df.get("ma_20", df["close"].rolling(20).mean())) / close)
    df["reversal_ema12_extension"] = direction * ((df["close"] - df.get("ema_12", df["close"].ewm(span=12, adjust=False).mean())) / close)

    dist_res = df[["dist_to_prev_res", "dist_to_prev_res_96"]].abs().min(axis=1)
    dist_sup = df[["dist_to_prev_sup", "dist_to_prev_sup_96"]].abs().min(axis=1)
    df["reversal_level_distance"] = np.where(direction >= 0, dist_res, dist_sup)

    touch_res = df.get("near_resistance", dist_res < 0.0008).astype(int)
    touch_sup = df.get("near_support", dist_sup < 0.0008).astype(int)
    df["reversal_level_touch"] = np.where(direction >= 0, touch_res, touch_sup)

    up_break_depth = np.maximum(
        (df["close"] - df[["prev_resistance_50", "prev_resistance_96"]].min(axis=1)) / close,
        0,
    )
    down_break_depth = np.maximum(
        (df[["prev_support_50", "prev_support_96"]].max(axis=1) - df["close"]) / close,
        0,
    )
    df["reversal_breakout_depth"] = np.where(direction >= 0, up_break_depth, down_break_depth)

    close_position = df.get("close_position", (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan))
    df["reversal_close_stretch"] = np.where(direction >= 0, close_position, 1 - close_position)

    upper_wick = df.get("upper_wick_pct", (df["high"] - df[["open", "close"]].max(axis=1)) / close)
    lower_wick = df.get("lower_wick_pct", (df[["open", "close"]].min(axis=1) - df["low"]) / close)
    candle_range = df.get("candle_range", (df["high"] - df["low"]) / close).replace(0, np.nan)
    rejection_wick = np.where(direction >= 0, upper_wick, lower_wick)
    df["reversal_rejection_wick"] = rejection_wick / candle_range

    signed_body = df.get("candle_body_signed", (df["close"] - df["open"]) / df["open"].replace(0, np.nan))
    df["reversal_body_against_impulse"] = -direction * signed_body

    df["reversal_range_ratio"] = df.get("range_ratio_20", df.get("candle_range", pd.Series(0, index=df.index)))
    df["reversal_body_ratio"] = df.get("body_ratio_20", df.get("candle_body", pd.Series(0, index=df.index)))
    vol_20 = df.get("volatility_20", df["close"].pct_change().rolling(20).std())
    vol_96 = df.get("volatility_96", df["close"].pct_change().rolling(96).std())
    df["reversal_volatility_ratio"] = vol_20 / vol_96.replace(0, np.nan)

    return df.replace([np.inf, -np.inf], np.nan)
