import numpy as np
import pandas as pd

from src.config import (
    CUSUM_THRESHOLD_MULT,
    CUSUM_VOLATILITY_WINDOW,
    NEAR_LEVEL_THRESHOLD,
    STRONG_BODY_MULTIPLIER,
    STRONG_RANGE_MULTIPLIER,
)


def add_cusum_events(
    dataframe: pd.DataFrame,
    return_col: str = "returns",
    volatility_window: int = CUSUM_VOLATILITY_WINDOW,
    threshold_mult: float = CUSUM_THRESHOLD_MULT,
) -> pd.DataFrame:
    df = dataframe.copy()
    rolling_vol = df[return_col].rolling(volatility_window).std().shift(1)
    vol_floor = rolling_vol.rolling(volatility_window * 5, min_periods=volatility_window).quantile(0.10).shift(1)
    stable_vol = pd.Series(np.maximum(rolling_vol, vol_floor), index=df.index)
    stable_vol = stable_vol.bfill().fillna(stable_vol.median())
    df["cusum_threshold"] = stable_vol * threshold_mult

    pos_sum = 0.0
    neg_sum = 0.0
    events = []
    directions = []

    for ret, threshold in zip(df[return_col].fillna(0).values, df["cusum_threshold"].fillna(0).values):
        pos_sum = max(0.0, pos_sum + ret)
        neg_sum = min(0.0, neg_sum + ret)

        if threshold > 0 and pos_sum > threshold:
            events.append(1)
            directions.append(1)
            pos_sum = 0.0
            neg_sum = 0.0
        elif threshold > 0 and neg_sum < -threshold:
            events.append(1)
            directions.append(-1)
            pos_sum = 0.0
            neg_sum = 0.0
        else:
            events.append(0)
            directions.append(0)

    df["event_cusum"] = events
    df["event_cusum_direction"] = directions
    return df


def add_event_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["prev_resistance_50"] = df["high"].rolling(50).max().shift(1)
    df["prev_support_50"] = df["low"].rolling(50).min().shift(1)
    df["prev_resistance_96"] = df["high"].rolling(96).max().shift(1)
    df["prev_support_96"] = df["low"].rolling(96).min().shift(1)
    df["dist_to_prev_res"] = (df["prev_resistance_50"] - df["close"]) / df["close"]
    df["dist_to_prev_sup"] = (df["close"] - df["prev_support_50"]) / df["close"]
    df["dist_to_prev_res_96"] = (df["prev_resistance_96"] - df["close"]) / df["close"]
    df["dist_to_prev_sup_96"] = (df["close"] - df["prev_support_96"]) / df["close"]
    df["candle_range"] = df.get("range_pct", (df["high"] - df["low"]) / df["open"])
    df["candle_body"] = df.get("body_abs_pct", (df["close"] - df["open"]).abs() / df["open"])
    df["candle_body_signed"] = df.get("body_pct", (df["close"] - df["open"]) / df["open"])
    df["avg_range_20"] = df["candle_range"].rolling(20).mean().shift(1)
    df["avg_body_20"] = df["candle_body"].rolling(20).mean().shift(1)
    df["range_ratio_20"] = df["candle_range"] / df["avg_range_20"].replace(0, np.nan)
    df["body_ratio_20"] = df["candle_body"] / df["avg_body_20"].replace(0, np.nan)
    return df


def detect_events(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = add_event_features(dataframe)
    df = add_cusum_events(df)

    df["near_resistance"] = (
        (df["dist_to_prev_res"].abs() < NEAR_LEVEL_THRESHOLD)
        | (df["dist_to_prev_res_96"].abs() < NEAR_LEVEL_THRESHOLD)
    )
    df["near_support"] = (
        (df["dist_to_prev_sup"].abs() < NEAR_LEVEL_THRESHOLD)
        | (df["dist_to_prev_sup_96"].abs() < NEAR_LEVEL_THRESHOLD)
    )
    df["near_level"] = df["near_resistance"] | df["near_support"]

    df["strong_range"] = df["range_ratio_20"] > STRONG_RANGE_MULTIPLIER
    df["strong_body"] = df["body_ratio_20"] > STRONG_BODY_MULTIPLIER
    df["strong_candle"] = df["strong_range"] | df["strong_body"]

    df["breakout_up"] = (df["close"] > df["prev_resistance_50"]) | (df["close"] > df["prev_resistance_96"])
    df["breakout_down"] = (df["close"] < df["prev_support_50"]) | (df["close"] < df["prev_support_96"])
    df["breakout"] = df["breakout_up"] | df["breakout_down"]

    df["event"] = (
        (df["event_cusum"] == 1) & (df["near_level"] | df["strong_candle"] | df["breakout"])
    ).astype(int)
    return df.replace([np.inf, -np.inf], np.nan).dropna()
