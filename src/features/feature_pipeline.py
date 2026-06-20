import numpy as np
import pandas as pd


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    price = data["close"].replace(0, np.nan)
    open_price = data["open"].replace(0, np.nan)

    data["returns"] = data["close"].pct_change()
    data["log_returns"] = np.log(data["close"]).diff()
    for window in [4, 8, 16, 32, 96]:
        data[f"momentum_{window}"] = data["close"].pct_change(window)

    data["range_pct"] = (data["high"] - data["low"]) / price
    data["body_pct"] = (data["close"] - data["open"]) / open_price
    data["body_abs_pct"] = data["body_pct"].abs()
    data["upper_wick_pct"] = (data["high"] - data[["open", "close"]].max(axis=1)) / price
    data["lower_wick_pct"] = (data[["open", "close"]].min(axis=1) - data["low"]) / price
    data["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0, np.nan)

    data["ma_10"] = data["close"].rolling(10).mean()
    data["ma_20"] = data["close"].rolling(20).mean()
    data["ma_50"] = data["close"].rolling(50).mean()
    data["ema_12"] = data["close"].ewm(span=12, adjust=False).mean()
    data["ema_26"] = data["close"].ewm(span=26, adjust=False).mean()

    data["ma_10_diff"] = (data["close"] - data["ma_10"]) / price
    data["ma_20_diff"] = (data["close"] - data["ma_20"]) / price
    data["ma_50_diff"] = (data["close"] - data["ma_50"]) / price
    data["ema_12_diff"] = (data["close"] - data["ema_12"]) / price
    data["ema_26_diff"] = (data["close"] - data["ema_26"]) / price
    data["volatility_20"] = data["returns"].rolling(20).std()
    data["volatility_96"] = data["returns"].rolling(96).std()
    data["range_mean_20"] = data["range_pct"].rolling(20).mean()
    data["volume_change"] = data["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    data["volume_zscore_96"] = (
        (data["volume"] - data["volume"].rolling(96).mean())
        / data["volume"].rolling(96).std().replace(0, np.nan)
    )

    data["hour_float"] = data.index.hour + data.index.minute / 60
    data["time_sin"] = np.sin(2 * np.pi * data["hour_float"] / 24)
    data["time_cos"] = np.cos(2 * np.pi * data["hour_float"] / 24)

    data["day_of_week"] = data.index.dayofweek
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    data["rolling_max_50"] = data["high"].rolling(50).max()
    data["rolling_min_50"] = data["low"].rolling(50).min()
    data["dist_to_res"] = (data["rolling_max_50"] - data["close"]) / data["close"]
    data["dist_to_sup"] = (data["close"] - data["rolling_min_50"]) / data["close"]
    data["break_res"] = (data["close"] > data["rolling_max_50"].shift(1)).astype(int)
    data["break_sup"] = (data["close"] < data["rolling_min_50"].shift(1)).astype(int)

    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))
    data["rsi_14_norm"] = (data["rsi_14"] - 50) / 100

    return data


def missing_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [column for column in feature_columns if column not in df.columns]
