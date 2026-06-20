import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src import config
from src.connector.data_fetcher import load_all_price_data
from src.features.event_detector import detect_events
from src.features.feature_pipeline import generate_features
from src.strategy.backtest import build_trades, calculate_trade_metrics
from src.models.sequence_models import load_model
from src.strategy.signal_generator import (
    add_labels_for_metrics,
    generate_rule_based_signal_history,
    generate_signal_history,
)
from src.models.training import train_direction_model


def _paths(model_type: str, dataset_mode: str) -> tuple[Path, Path]:
    if dataset_mode == "event":
        return config.EVENT_MODEL_PATHS[model_type], config.EVENT_SCALER_PATHS[model_type]
    if dataset_mode == "full":
        return config.FULL_MODEL_PATHS[model_type], config.FULL_SCALER_PATHS[model_type]
    raise ValueError("dataset_mode must be 'event' or 'full'")


def _classification_from_signals(
    signals: pd.DataFrame,
    prepared_df: pd.DataFrame,
    horizon: int,
    label_threshold: float,
    confidence_threshold: float | None = None,
    event_only: bool = True,
) -> dict:
    labeled = add_labels_for_metrics(signals, prepared_df, horizon=horizon, threshold=label_threshold)
    if event_only:
        labeled = labeled[labeled["event"] == 1]
    if confidence_threshold is not None:
        labeled = labeled[labeled["confidence"] >= confidence_threshold]
    if labeled.empty:
        return {"accuracy": 0.0, "f1": 0.0, "balanced_accuracy": 0.0, "count": 0}

    y_true = (labeled["actual"] == "UP").astype(int)
    y_pred = (labeled["prediction"] == "UP").astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "count": int(len(labeled)),
    }


def evaluate_saved_strategy(
    price_df: pd.DataFrame,
    model_type: str,
    dataset_mode: str,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    label_threshold: float = config.DEFAULT_LABEL_THRESHOLD,
    max_rows: int | None = None,
) -> dict:
    model_path, scaler_path = _paths(model_type, dataset_mode)
    if not model_path.exists() or not scaler_path.exists():
        return {
            "strategy": f"{dataset_mode}+{model_type}".upper(),
            "status": "missing model/scaler",
            "model_path": str(model_path),
        }

    prepared_df = detect_events(generate_features(price_df))
    model = load_model(model_path, model_type=model_type)
    scaler = joblib.load(scaler_path)
    signals = generate_signal_history(
        price_df,
        threshold=confidence_threshold,
        model_path=model_path,
        scaler_path=scaler_path,
        model_type=model_type,
        require_event=(dataset_mode == "event"),
        max_rows=max_rows or len(prepared_df),
        model=model,
        scaler=scaler,
    )

    all_event = _classification_from_signals(
        signals,
        prepared_df,
        horizon=horizon,
        label_threshold=label_threshold,
        confidence_threshold=None,
        event_only=True,
    )
    confident = _classification_from_signals(
        signals,
        prepared_df,
        horizon=horizon,
        label_threshold=label_threshold,
        confidence_threshold=confidence_threshold,
        event_only=True,
    )
    trades = build_trades(signals, prepared_df, horizon=horizon)
    trade_metrics = calculate_trade_metrics(trades)

    return {
        "strategy": f"{dataset_mode}+{model_type}".upper(),
        "status": "ok",
        "all_event": all_event,
        "confident": confident,
        "trades": int(trade_metrics["Trades"]),
        "total_return": trade_metrics["total_return"],
        "winrate": trade_metrics["winrate"],
        "max_drawdown": trade_metrics["max_drawdown"],
        "profit_factor": trade_metrics["profit_factor"],
        "avg_trade": trade_metrics["avg_trade"],
    }


def evaluate_rule_baseline(
    price_df: pd.DataFrame,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    label_threshold: float = config.DEFAULT_LABEL_THRESHOLD,
    max_rows: int | None = None,
) -> dict:
    prepared_df = detect_events(generate_features(price_df))
    signals = generate_rule_based_signal_history(price_df, max_rows=max_rows or len(prepared_df))
    all_event = _classification_from_signals(
        signals,
        prepared_df,
        horizon=horizon,
        label_threshold=label_threshold,
        confidence_threshold=None,
        event_only=True,
    )
    trades = build_trades(signals, prepared_df, horizon=horizon)
    trade_metrics = calculate_trade_metrics(trades)
    return {
        "strategy": "RULE_CUSUM",
        "status": "ok",
        "all_event": all_event,
        "confident": all_event,
        "trades": int(trade_metrics["Trades"]),
        "total_return": trade_metrics["total_return"],
        "winrate": trade_metrics["winrate"],
        "max_drawdown": trade_metrics["max_drawdown"],
        "profit_factor": trade_metrics["profit_factor"],
        "avg_trade": trade_metrics["avg_trade"],
    }


def confidence_analysis(
    signals: pd.DataFrame,
    prepared_df: pd.DataFrame,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    label_threshold: float = config.DEFAULT_LABEL_THRESHOLD,
    thresholds: tuple[float, ...] = (0.50, 0.52, 0.55, 0.57, 0.60),
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = _classification_from_signals(
            signals,
            prepared_df,
            horizon=horizon,
            label_threshold=label_threshold,
            confidence_threshold=threshold,
            event_only=True,
        )
        rows.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "trades": metrics["count"],
            }
        )
    return pd.DataFrame(rows)


def run_comparison(
    price_df: pd.DataFrame,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    label_threshold: float = config.DEFAULT_LABEL_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for dataset_mode in ["event", "full"]:
        for model_type in ["gru", "lstm"]:
            rows.append(
                evaluate_saved_strategy(
                    price_df,
                    model_type=model_type,
                    dataset_mode=dataset_mode,
                    confidence_threshold=confidence_threshold,
                    horizon=horizon,
                    label_threshold=label_threshold,
                )
            )
    rows.append(evaluate_rule_baseline(price_df, horizon=horizon, label_threshold=label_threshold))
    return pd.json_normalize(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare Event-based, Full-data, GRU, LSTM and rule baselines.")
    parser.add_argument("--train", action="store_true", help="Train the requested model before comparison.")
    parser.add_argument("--model-type", choices=["gru", "lstm"], default="lstm")
    parser.add_argument("--dataset-mode", choices=["event", "full"], default="event")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--confidence-threshold", type=float, default=config.DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--label-threshold", type=float, default=config.DEFAULT_LABEL_THRESHOLD)
    parser.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_CANDLES)
    args = parser.parse_args()

    price_df, loaded = load_all_price_data(config.DATA_DIR)
    print(f"Loaded {len(loaded)} CSV files, rows={len(price_df)}")

    if args.train:
        model_path, scaler_path = _paths(args.model_type, args.dataset_mode)
        result = train_direction_model(
            price_df,
            model_type=args.model_type,
            event_only=(args.dataset_mode == "event"),
            label_threshold=args.label_threshold,
            horizon=args.horizon,
            epochs=args.epochs,
            model_path=model_path,
            scaler_path=scaler_path,
        )
        print(f"Trained {args.dataset_mode}+{args.model_type}: valid_samples={result['valid_samples']}")

    comparison = run_comparison(
        price_df,
        confidence_threshold=args.confidence_threshold,
        horizon=args.horizon,
        label_threshold=args.label_threshold,
    )
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(comparison)


if __name__ == "__main__":
    main()
