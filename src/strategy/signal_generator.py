from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import (
    ACTIVE_RULE_CUSUM_THRESHOLD_MULT,
    ACTIVE_RULE_CUSUM_VOLATILITY_WINDOW,
    ACTIVE_RULE_DIRECTION,
    DEFAULT_LABEL_THRESHOLD,
    EVENT_MODEL_PATHS,
    EVENT_SCALER_PATHS,
    FEATURE_COLUMNS,
    NEAR_LEVEL_THRESHOLD,
    MODEL_PATH,
    MODEL_TYPE,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    STRONG_BODY_MULTIPLIER,
    STRONG_RANGE_MULTIPLIER,
)
from src.features.event_detector import add_cusum_events, add_event_features, detect_events
from src.features.feature_pipeline import generate_features, missing_feature_columns
from src.features.reversal_quality import add_reversal_quality_features
from src.models.sequence_models import load_model


def load_scaler(scaler_path: Path = SCALER_PATH):
    if not scaler_path.exists():
        raise FileNotFoundError(f"Файл scaler не найден: {scaler_path}")
    try:
        return joblib.load(scaler_path)
    except Exception:
        with open(scaler_path, "rb") as file:
            return pickle.load(file)


def prepare_inference_frame(price_df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    featured = generate_features(price_df)
    detected = detect_events(featured)
    missing = missing_feature_columns(detected, feature_columns)
    if missing:
        raise ValueError(f"Не хватает признаков для модели: {missing}")
    if len(detected) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Недостаточно строк после генерации признаков: {len(detected)}. Нужно минимум {SEQUENCE_LENGTH}."
        )
    return detected


def predict_next_signal(
    price_df: pd.DataFrame,
    threshold: float = 0.60,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    model_type: str = MODEL_TYPE,
    require_event: bool = True,
    model=None,
    scaler=None,
    feature_columns: list[str] | None = None,
):
    feature_columns = feature_columns or FEATURE_COLUMNS
    df = prepare_inference_frame(price_df, feature_columns=feature_columns)
    model_path = model_path or EVENT_MODEL_PATHS[model_type]
    scaler_path = scaler_path or EVENT_SCALER_PATHS[model_type]
    scaler = scaler or load_scaler(scaler_path)
    model = model or load_model(model_path, model_type=model_type)

    feature_frame = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    scaled = scaler.transform(feature_frame)
    window = scaled[-SEQUENCE_LENGTH:]
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        if logits.shape[1] == 1:
            probability_up = torch.sigmoid(logits).item()
        else:
            probability_up = torch.softmax(logits, dim=1)[0, 1].item()

    prediction = 1 if probability_up >= 0.5 else 0
    prediction_label = "UP" if prediction == 1 else "DOWN"
    confidence = max(probability_up, 1 - probability_up)

    last = df.iloc[-1]
    event_detected = int(last["event"]) == 1
    event_allowed = event_detected or not require_event
    if event_allowed and prediction == 1 and confidence >= threshold:
        decision = "BUY"
    elif event_allowed and prediction == 0 and confidence >= threshold:
        decision = "SELL"
    else:
        decision = "NO TRADE"

    return {
        "timestamp": df.index[-1],
        "event": int(last["event"]),
        "prediction": prediction_label,
        "probability_up": float(probability_up),
        "confidence": float(confidence),
        "decision": decision,
        "prepared_df": df,
    }


def generate_signal_history(
    price_df: pd.DataFrame,
    threshold: float = 0.60,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    model_type: str = MODEL_TYPE,
    require_event: bool = True,
    max_rows: int = 500,
    model=None,
    scaler=None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    df = prepare_inference_frame(price_df, feature_columns=feature_columns)
    model_path = model_path or EVENT_MODEL_PATHS[model_type]
    scaler_path = scaler_path or EVENT_SCALER_PATHS[model_type]
    scaler = scaler or load_scaler(scaler_path)
    model = model or load_model(model_path, model_type=model_type)
    device = next(model.parameters()).device

    feature_frame = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    scaled = scaler.transform(feature_frame)
    start = max(SEQUENCE_LENGTH - 1, len(df) - max_rows)
    positions = np.arange(start, len(df))
    probabilities = np.empty(len(positions), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(positions), 512):
            batch_positions = positions[batch_start : batch_start + 512]
            windows = np.stack([scaled[i - SEQUENCE_LENGTH + 1 : i + 1] for i in batch_positions])
            x = torch.tensor(windows, dtype=torch.float32).to(device)
            logits = model(x)
            if logits.shape[1] == 1:
                batch_probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            else:
                batch_probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probabilities[batch_start : batch_start + len(batch_positions)] = batch_probabilities

    rows = []
    for i, probability_up in zip(positions, probabilities):
        prediction = "UP" if probability_up >= 0.5 else "DOWN"
        confidence = max(float(probability_up), 1 - float(probability_up))
        event = int(df.iloc[i]["event"])
        event_cusum_direction = int(df.iloc[i].get("event_cusum_direction", 0))
        event_allowed = event == 1 or not require_event
        if event_allowed and prediction == "UP" and confidence >= threshold:
            decision = "BUY"
        elif event_allowed and prediction == "DOWN" and confidence >= threshold:
            decision = "SELL"
        else:
            decision = "NO TRADE"
        rows.append(
            {
                "time": df.index[i],
                "event": event,
                "event_cusum_direction": event_cusum_direction,
                "prediction": prediction,
                "probability_up": float(probability_up),
                "confidence": confidence,
                "decision": decision,
                "close": df.iloc[i]["close"],
            }
        )

    return pd.DataFrame(rows)


def first_hit_label_at(
    price_df: pd.DataFrame,
    entry_idx: int,
    threshold: float = DEFAULT_LABEL_THRESHOLD,
    horizon: int = 8,
) -> str | None:
    entry_price = float(price_df.iloc[entry_idx]["close"])
    up_level = entry_price + threshold
    down_level = entry_price - threshold
    end_idx = min(entry_idx + horizon, len(price_df) - 1)

    for idx in range(entry_idx + 1, end_idx + 1):
        high = float(price_df.iloc[idx]["high"])
        low = float(price_df.iloc[idx]["low"])
        up_hit = high >= up_level
        down_hit = low <= down_level
        if up_hit and down_hit:
            return None
        if up_hit:
            return "UP"
        if down_hit:
            return "DOWN"
    return None


def add_first_hit_direction_labels(
    frame: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int,
    threshold: float = DEFAULT_LABEL_THRESHOLD,
    time_col: str = "time",
) -> pd.DataFrame:
    if frame.empty:
        return frame

    price_df = price_df.sort_index()
    labeled = frame.copy()
    actual = []
    for timestamp in labeled[time_col]:
        if timestamp not in price_df.index:
            entry_idx = price_df.index.get_indexer([timestamp], method="nearest")[0]
        else:
            entry_idx = price_df.index.get_loc(timestamp)
            if not isinstance(entry_idx, int):
                entry_idx = int(entry_idx[0])
        actual.append(first_hit_label_at(price_df, entry_idx, threshold=threshold, horizon=horizon))

    labeled["actual"] = actual
    return labeled.dropna(subset=["actual"])


def add_labels_for_metrics(
    signals: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int,
    threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> pd.DataFrame:
    if signals.empty:
        return signals
    return add_first_hit_direction_labels(signals, price_df, horizon=horizon, threshold=threshold)


def generate_rule_based_signal_history(price_df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    return generate_active_rule_signal_history(price_df, max_rows=max_rows)


def prepare_rule_frame(
    price_df: pd.DataFrame,
    volatility_window: int = ACTIVE_RULE_CUSUM_VOLATILITY_WINDOW,
    threshold_mult: float = ACTIVE_RULE_CUSUM_THRESHOLD_MULT,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    df = generate_features(price_df)
    df = add_event_features(df)
    df = add_cusum_events(df, volatility_window=volatility_window, threshold_mult=threshold_mult)

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
    df["event"] = ((df["event_cusum"] == 1) & (df["near_level"] | df["strong_candle"] | df["breakout"])).astype(int)
    df = add_reversal_quality_features(df)

    missing = missing_feature_columns(df, feature_columns)
    if missing:
        raise ValueError(f"Не хватает признаков для rule baseline: {missing}")
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def rule_direction_from_row(row: pd.Series, direction_rule: str = ACTIVE_RULE_DIRECTION) -> str | None:
    cusum_direction = int(row.get("event_cusum_direction", 0))
    if direction_rule == "cusum_momentum":
        if cusum_direction == 1:
            return "BUY"
        if cusum_direction == -1:
            return "SELL"
        return None
    if direction_rule == "cusum_reversal":
        if cusum_direction == 1:
            return "SELL"
        if cusum_direction == -1:
            return "BUY"
        return None
    raise ValueError(f"Unsupported rule direction: {direction_rule}")


def generate_active_rule_signal_history(
    price_df: pd.DataFrame,
    max_rows: int = 500,
    direction_rule: str = ACTIVE_RULE_DIRECTION,
    volatility_window: int = ACTIVE_RULE_CUSUM_VOLATILITY_WINDOW,
    threshold_mult: float = ACTIVE_RULE_CUSUM_THRESHOLD_MULT,
) -> pd.DataFrame:
    df = prepare_rule_frame(price_df, volatility_window=volatility_window, threshold_mult=threshold_mult)
    start = max(SEQUENCE_LENGTH - 1, len(df) - max_rows)
    rows = []
    for i in range(start, len(df)):
        event = int(df.iloc[i].get("event", 0))
        decision = rule_direction_from_row(df.iloc[i], direction_rule) if event == 1 else None
        if decision == "BUY":
            prediction = "UP"
        elif decision == "SELL":
            prediction = "DOWN"
        else:
            decision = "NO TRADE"
            prediction = "UP" if int(df.iloc[i].get("event_cusum_direction", 0)) >= 0 else "DOWN"
        rows.append(
            {
                "time": df.index[i],
                "event": event,
                "event_cusum_direction": int(df.iloc[i].get("event_cusum_direction", 0)),
                "rule_direction": direction_rule,
                "prediction": prediction,
                "probability_up": 1.0 if prediction == "UP" else 0.0,
                "confidence": 1.0 if decision in ("BUY", "SELL") else 0.0,
                "decision": decision,
                "close": df.iloc[i]["close"],
            }
        )
    return pd.DataFrame(rows)
