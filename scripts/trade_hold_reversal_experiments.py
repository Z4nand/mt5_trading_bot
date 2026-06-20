from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config  # noqa: E402
from src.connector.data_fetcher import load_all_price_data  # noqa: E402
from src.features.news_features import add_news_features  # noqa: E402
from src.features.reversal_quality import REVERSAL_QUALITY_FEATURE_COLUMNS  # noqa: E402
from src.models.sequence_models import create_model  # noqa: E402
from src.models.trade_skip_training import (  # noqa: E402
    TradeSkipSequenceDataset,
    class_weights,
    evaluate_trade_skip_model,
    prepare_trade_skip_arrays,
)
from src.strategy.backtest import _resolve_exit, build_trades, calculate_trade_metrics  # noqa: E402
from src.strategy.signal_generator import prepare_rule_frame, rule_direction_from_row  # noqa: E402


OUTPUT_DIR = config.DATA_DIR / "processed" / "trade_hold_reversal_experiments"
REPORT_PATH = BASE_DIR / "reports" / "trade_hold_reversal_experiment_report.md"
MODEL_PATH = config.MODELS_DIR / "trade_hold_reversal_gru_best.pth"
SCALER_PATH = config.MODELS_DIR / "trade_hold_reversal_gru_scaler.pkl"
CONFIG_PATH = config.MODELS_DIR / "trade_hold_reversal_gru_config.pkl"
QUALITY_OUTPUT_DIR = config.DATA_DIR / "processed" / "trade_hold_reversal_quality_experiments"
QUALITY_REPORT_PATH = BASE_DIR / "reports" / "trade_hold_reversal_quality_experiment_report.md"
QUALITY_MODEL_PATH = config.MODELS_DIR / "trade_hold_reversal_quality_gru_best.pth"
QUALITY_SCALER_PATH = config.MODELS_DIR / "trade_hold_reversal_quality_gru_scaler.pkl"
QUALITY_CONFIG_PATH = config.MODELS_DIR / "trade_hold_reversal_quality_gru_config.pkl"
BASE_CONFIG_PATH = config.MODELS_DIR / "trade_skip_reversal_gru_config.pkl"

HORIZON = 8
TP_THRESHOLD = 0.0008
SL_THRESHOLD = 0.0004
COST_PER_TRADE = 0.00003
THRESHOLDS = (0.50, 0.51, 0.52, 0.53, 0.55, 0.57)
CUSUM_GRID = (
    (96, 1.5),
    (96, 1.8),
    (96, 2.1),
    (192, 1.5),
    (192, 1.8),
    (192, 2.1),
    (288, 1.5),
    (288, 1.8),
    (288, 2.1),
)
LABEL_MODES = {
    "positive_horizon": "TRADE if CUSUM-reversal position is profitable at horizon close.",
    "tp_before_sl": "TRADE if TP is hit first; HOLD for SL or horizon.",
    "tp_sl_only": "TRADE if TP is hit first, HOLD if SL is hit first, ignore horizon-only cases.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research CUSUM-reversal TRADE/HOLD training variants.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-cusum", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--use-quality-features", action="store_true")
    return parser.parse_args()


def ts(value: str, index: pd.Index) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if getattr(index, "tz", None) is not None:
        return timestamp.tz_localize(index.tz)
    return timestamp


def base_feature_columns() -> list[str]:
    return list(joblib.load(BASE_CONFIG_PATH)["feature_columns"])


def reversal_decision(row: pd.Series) -> str | None:
    return rule_direction_from_row(row, "cusum_reversal")


def label_for_event(df: pd.DataFrame, entry_idx: int, label_mode: str) -> int | None:
    direction = reversal_decision(df.iloc[entry_idx])
    if direction is None:
        return None
    entry_price = float(df.iloc[entry_idx]["close"])

    if label_mode == "positive_horizon":
        exit_idx = min(entry_idx + HORIZON, len(df) - 1)
        if exit_idx <= entry_idx:
            return None
        exit_price = float(df.iloc[exit_idx]["close"])
        pnl = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        return int(pnl > 0)

    exit_idx, _, exit_reason = _resolve_exit(
        price_df=df,
        entry_idx=entry_idx,
        direction=direction,
        horizon=HORIZON,
        tp_threshold=TP_THRESHOLD,
        sl_threshold=SL_THRESHOLD,
    )
    if exit_idx <= entry_idx:
        return None
    if exit_reason == "TP":
        return 1
    if exit_reason == "SL":
        return 0
    if label_mode == "tp_before_sl":
        return 0
    if label_mode == "tp_sl_only":
        return None
    raise ValueError(f"Unknown label_mode: {label_mode}")


def build_trade_hold_frame(
    price_df: pd.DataFrame,
    feature_columns: list[str],
    volatility_window: int,
    threshold_mult: float,
    label_mode: str,
) -> pd.DataFrame:
    df = prepare_rule_frame(
        price_df,
        volatility_window=volatility_window,
        threshold_mult=threshold_mult,
        feature_columns=feature_columns,
    )
    decisions = []
    encoded = []
    labels = []
    for idx, row in enumerate(df.itertuples(index=False)):
        row_series = df.iloc[idx]
        if int(row_series.get("event", 0)) != 1:
            decisions.append(None)
            encoded.append(0)
            labels.append(None)
            continue
        decision = reversal_decision(row_series)
        decisions.append(decision)
        encoded.append(1 if decision == "BUY" else -1 if decision == "SELL" else 0)
        labels.append(label_for_event(df, idx, label_mode))

    result = df.copy()
    result["original_event_cusum_direction"] = result["event_cusum_direction"]
    result["event_cusum_direction"] = encoded
    result["rule_decision"] = decisions
    result["trade_success"] = labels
    return result


def train_trade_hold_model(
    labeled_df: pd.DataFrame,
    feature_columns: list[str],
    epochs: int,
    batch_size: int,
) -> dict:
    scaled, labels, train_indices, valid_indices, test_indices, scaler = prepare_trade_skip_arrays(
        labeled_df,
        horizon=HORIZON,
        feature_columns=feature_columns,
    )
    train_ds = TradeSkipSequenceDataset(scaled, labels, train_indices)
    valid_ds = TradeSkipSequenceDataset(scaled, labels, valid_indices)
    test_ds = TradeSkipSequenceDataset(scaled, labels, test_indices)
    if len(train_ds) == 0 or len(valid_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Not enough samples: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_type="gru",
        input_size=len(feature_columns),
        hidden_size=64,
        dropout=0.25,
        num_layers=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(labels, train_indices, device))
    best_state = None
    best_score = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y)
        metrics = evaluate_trade_skip_model(model, valid_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / max(len(train_ds), 1)
        history.append(metrics)
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "best_valid_score": best_score,
        "test_metrics": evaluate_trade_skip_model(model, test_loader, device),
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
    }


def raw_signals_from_model(
    labeled_df: pd.DataFrame,
    model: torch.nn.Module,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    feature_frame = labeled_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    scaled = scaler.transform(feature_frame).astype(np.float32)
    positions = np.arange(config.SEQUENCE_LENGTH - 1, len(labeled_df))
    probabilities = np.empty(len(positions), dtype=np.float32)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), 512):
            batch_positions = positions[start : start + 512]
            windows = np.stack([scaled[i - config.SEQUENCE_LENGTH + 1 : i + 1] for i in batch_positions])
            logits = model(torch.tensor(windows, dtype=torch.float32).to(device))
            probabilities[start : start + len(batch_positions)] = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    rows = []
    for idx, probability in zip(positions, probabilities):
        event = int(labeled_df.iloc[idx].get("event", 0))
        direction = int(labeled_df.iloc[idx].get("event_cusum_direction", 0))
        decision = "BUY" if direction == 1 else "SELL" if direction == -1 else "NO TRADE"
        rows.append(
            {
                "time": labeled_df.index[idx],
                "event": event,
                "event_cusum_direction": direction,
                "prediction": "TRADE" if probability >= 0.5 else "HOLD",
                "confidence": float(probability),
                "probability_trade": float(probability),
                "decision": decision if event == 1 else "NO TRADE",
                "close": labeled_df.iloc[idx]["close"],
            }
        )
    return pd.DataFrame(rows)


def threshold_signals(raw_signals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = raw_signals.copy()
    allowed = result["event"].astype(int).eq(1) & result["confidence"].astype(float).ge(threshold)
    result.loc[~allowed, "decision"] = "NO TRADE"
    result["prediction"] = np.where(allowed, "TRADE", "HOLD")
    return result


def apply_news_filter(signals: pd.DataFrame, prepared_df: pd.DataFrame, minutes: int = 60) -> pd.DataFrame:
    if signals.empty or "news_minutes_to_next" not in prepared_df.columns:
        return signals.copy()
    enriched = signals.merge(
        prepared_df[["news_minutes_to_next", "news_next_impact_score"]],
        left_on="time",
        right_index=True,
        how="left",
    )
    skip = enriched["news_minutes_to_next"].between(0, minutes, inclusive="both") & enriched["news_next_impact_score"].ge(2 / 3)
    result = enriched.copy()
    result.loc[skip, "decision"] = "NO TRADE"
    return result.drop(columns=["news_minutes_to_next", "news_next_impact_score"])


def evaluate_signal_set(
    strategy: str,
    raw_signals: pd.DataFrame,
    prepared_df: pd.DataFrame,
    price_df: pd.DataFrame,
    segment: str,
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    train_end = ts(config.TRAIN_END_DATE, prepared_df.index)
    valid_end = ts(config.VALID_END_DATE, prepared_df.index)
    if segment == "valid":
        mask = raw_signals["time"].between(train_end, valid_end, inclusive="left")
    elif segment == "test":
        mask = raw_signals["time"] >= valid_end
    else:
        raise ValueError(segment)

    rows = []
    segment_raw = raw_signals.loc[mask].copy()
    for threshold in thresholds:
        for news_filter in (False, True):
            signals = threshold_signals(segment_raw, threshold)
            if news_filter:
                signals = apply_news_filter(signals, prepared_df)
            trades = build_trades(
                signals,
                price_df,
                horizon=HORIZON,
                tp_threshold=TP_THRESHOLD,
                sl_threshold=SL_THRESHOLD,
                cost_per_trade=COST_PER_TRADE,
            )
            metrics = calculate_trade_metrics(trades)
            rows.append(
                {
                    "strategy": strategy,
                    "segment": segment,
                    "threshold": threshold,
                    "news_filter": news_filter,
                    "trades": int(metrics["Trades"]),
                    "total_pnl_pips": float(trades["PnL"].sum() * 10000) if not trades.empty else 0.0,
                    "winrate": float(metrics["Win Rate"]),
                    "profit_factor": float(metrics["Profit Factor"]),
                    "max_drawdown": float(metrics["Max Drawdown"]),
                    "avg_trade_pips": float(metrics["Average Trade"] * 10000),
                }
            )
    return pd.DataFrame(rows)


def rule_raw_signals(prepared_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in prepared_df.iterrows():
        if int(row.get("event", 0)) == 1:
            decision = reversal_decision(row) or "NO TRADE"
            direction = 1 if decision == "BUY" else -1 if decision == "SELL" else 0
        else:
            decision = "NO TRADE"
            direction = 0
        rows.append(
            {
                "time": idx,
                "event": int(row.get("event", 0)),
                "event_cusum_direction": direction,
                "prediction": "TRADE" if decision != "NO TRADE" else "HOLD",
                "confidence": 1.0 if decision != "NO TRADE" else 0.0,
                "probability_trade": 1.0 if decision != "NO TRADE" else 0.0,
                "decision": decision,
                "close": row["close"],
            }
        )
    return pd.DataFrame(rows)


def select_top_cusum(price_df: pd.DataFrame, feature_columns: list[str], top_n: int, output_dir: Path) -> pd.DataFrame:
    rows = []
    for window, mult in CUSUM_GRID:
        prepared = prepare_rule_frame(
            price_df,
            volatility_window=window,
            threshold_mult=mult,
            feature_columns=feature_columns,
        )
        raw = rule_raw_signals(prepared)
        valid = evaluate_signal_set(
            strategy=f"rule_w{window}_m{mult}",
            raw_signals=raw,
            prepared_df=prepared,
            price_df=price_df,
            segment="valid",
            thresholds=(0.5,),
        )
        row = valid.sort_values(["profit_factor", "total_pnl_pips"], ascending=False).iloc[0].to_dict()
        row["cusum_window"] = window
        row["cusum_mult"] = mult
        rows.append(row)
    scan = pd.DataFrame(rows).sort_values(["profit_factor", "total_pnl_pips"], ascending=False)
    scan.to_csv(output_dir / "rule_cusum_validation_scan.csv", index=False)
    return scan.head(top_n)


def save_best(best: dict, feature_columns: list[str], model_path: Path, scaler_path: Path, config_path: Path) -> dict:
    torch.save(best["model"].state_dict(), model_path)
    joblib.dump(best["scaler"], scaler_path)
    model_config = {
        "task": "trade_hold_reversal",
        "model_type": "gru",
        "input_size": len(feature_columns),
        "hidden_size": 64,
        "dropout": 0.25,
        "num_layers": 2,
        "horizon": HORIZON,
        "tp_threshold": TP_THRESHOLD,
        "sl_threshold": SL_THRESHOLD,
        "feature_columns": feature_columns,
        "quality_feature_columns": [column for column in REVERSAL_QUALITY_FEATURE_COLUMNS if column in feature_columns],
        "direction_rule": "cusum_reversal",
        "cusum_volatility_window": int(best["cusum_window"]),
        "cusum_threshold_mult": float(best["cusum_mult"]),
        "label_mode": best["label_mode"],
        "train_end_date": config.TRAIN_END_DATE,
        "valid_end_date": config.VALID_END_DATE,
        "selection": "selected by validation trading metrics; test used only for final comparison",
    }
    joblib.dump(model_config, config_path)
    return model_config


def report_text(
    rule_scan: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    trading: pd.DataFrame,
    model_config: dict,
    model_path: Path,
    scaler_path: Path,
    config_path: Path,
) -> str:
    valid_best = trading[trading["segment"].eq("valid")].sort_values(["profit_factor", "total_pnl_pips"], ascending=False).groupby("strategy").head(1)
    test_best = trading[trading["segment"].eq("test")].sort_values(["profit_factor", "total_pnl_pips"], ascending=False).groupby("strategy").head(1)
    return "\n".join(
        [
            "# Trade/Hold CUSUM Reversal Experiment",
            "",
            "## Idea",
            "CUSUM-reversal defines the trade direction. The neural model solves only TRADE vs HOLD.",
            f"Feature count: {model_config['input_size']}. Quality features: {len(model_config.get('quality_feature_columns', []))}.",
            "",
            "## Data split",
            f"- Train: before {config.TRAIN_END_DATE}",
            f"- Validation: {config.TRAIN_END_DATE} to {config.VALID_END_DATE}",
            f"- Test: from {config.VALID_END_DATE}",
            "",
            "## Label modes",
            *[f"- {name}: {description}" for name, description in LABEL_MODES.items()],
            "",
            "## Top rule-only CUSUM settings on validation",
            "```",
            rule_scan.head(10).to_string(index=False),
            "```",
            "",
            "## Candidate training summary",
            "```",
            candidate_summary.to_string(index=False),
            "```",
            "",
            "## Best validation row per trained strategy",
            "```",
            valid_best.to_string(index=False),
            "```",
            "",
            "## Test row for selected strategies",
            "```",
            test_best.to_string(index=False),
            "```",
            "",
            "## Selected model",
            f"- Model: `{model_path}`",
            f"- Scaler: `{scaler_path}`",
            f"- Config: `{config_path}`",
            f"- Label mode: {model_config['label_mode']}",
            f"- CUSUM window: {model_config['cusum_volatility_window']}",
            f"- CUSUM multiplier: {model_config['cusum_threshold_mult']}",
            "",
            "## Notes",
            "- Trading metrics use TP 8 pips, SL 4 pips, horizon 8 candles, cost 0.3 pip.",
            "- News filter is evaluated as a separate entry filter, not as a training feature.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.use_quality_features and args.output_dir == OUTPUT_DIR:
        args.output_dir = QUALITY_OUTPUT_DIR
    report_path = QUALITY_REPORT_PATH if args.use_quality_features else REPORT_PATH
    model_path = QUALITY_MODEL_PATH if args.use_quality_features else MODEL_PATH
    scaler_path = QUALITY_SCALER_PATH if args.use_quality_features else SCALER_PATH
    config_path = QUALITY_CONFIG_PATH if args.use_quality_features else CONFIG_PATH
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = base_feature_columns()
    if args.use_quality_features:
        feature_columns = feature_columns + [column for column in REVERSAL_QUALITY_FEATURE_COLUMNS if column not in feature_columns]
    print(
        "Torch device:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "| cuda_available=",
        torch.cuda.is_available(),
    )
    price_df, loaded = load_all_price_data(config.DATA_DIR)
    price_df = add_news_features(price_df)
    print(f"Loaded {len(price_df)} candles from {len(loaded)} files.")

    top_cusum = select_top_cusum(price_df, feature_columns, args.top_cusum, args.output_dir)
    print("Top CUSUM settings:")
    print(top_cusum[["cusum_window", "cusum_mult", "profit_factor", "total_pnl_pips", "trades", "news_filter"]].to_string(index=False))

    candidate_rows = []
    trading_frames = []
    best_record = None
    trained_records = []

    for _, cusum_row in top_cusum.iterrows():
        window = int(cusum_row["cusum_window"])
        mult = float(cusum_row["cusum_mult"])
        for label_mode in LABEL_MODES:
            strategy = f"hold_{label_mode}_w{window}_m{mult}"
            print(f"Training {strategy} for {args.epochs} epochs...")
            labeled = build_trade_hold_frame(price_df, feature_columns, window, mult, label_mode)
            result = train_trade_hold_model(labeled, feature_columns, epochs=args.epochs, batch_size=args.batch_size)
            pd.DataFrame(result["history"]).to_csv(args.output_dir / f"history_{strategy}.csv", index=False)
            raw = raw_signals_from_model(labeled, result["model"], result["scaler"], feature_columns)
            valid_trading = evaluate_signal_set(strategy, raw, labeled, price_df, "valid", THRESHOLDS)
            test_trading = evaluate_signal_set(strategy, raw, labeled, price_df, "test", THRESHOLDS)
            trading_frames.extend([valid_trading, test_trading])

            valid_best = valid_trading.sort_values(["profit_factor", "total_pnl_pips", "trades"], ascending=False).iloc[0]
            candidate_rows.append(
                {
                    "strategy": strategy,
                    "label_mode": label_mode,
                    "cusum_window": window,
                    "cusum_mult": mult,
                    "train_samples": result["train_samples"],
                    "valid_samples": result["valid_samples"],
                    "test_samples": result["test_samples"],
                    "best_valid_balanced_accuracy": float(result["best_valid_score"]),
                    "test_balanced_accuracy": float(result["test_metrics"]["balanced_accuracy"]),
                    "valid_best_threshold": float(valid_best["threshold"]),
                    "valid_best_news_filter": bool(valid_best["news_filter"]),
                    "valid_best_pf": float(valid_best["profit_factor"]),
                    "valid_best_pips": float(valid_best["total_pnl_pips"]),
                    "valid_best_trades": int(valid_best["trades"]),
                }
            )
            record = {
                "strategy": strategy,
                "label_mode": label_mode,
                "cusum_window": window,
                "cusum_mult": mult,
                "model": result["model"],
                "scaler": result["scaler"],
                "valid_best_pf": float(valid_best["profit_factor"]),
                "valid_best_pips": float(valid_best["total_pnl_pips"]),
                "valid_best_trades": int(valid_best["trades"]),
            }
            trained_records.append(record)

    candidate_summary = pd.DataFrame(candidate_rows).sort_values(["valid_best_pf", "valid_best_pips"], ascending=False)
    candidate_summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    trading = pd.concat(trading_frames, ignore_index=True)
    trading.to_csv(args.output_dir / "threshold_scan.csv", index=False)

    viable = [r for r in trained_records if r["valid_best_trades"] >= 100]
    if not viable:
        viable = trained_records
    best_record = sorted(viable, key=lambda r: (r["valid_best_pf"], r["valid_best_pips"], r["valid_best_trades"]), reverse=True)[0]
    model_config = save_best(best_record, feature_columns, model_path, scaler_path, config_path)
    (args.output_dir / "selected_model.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    report_path.write_text(
        report_text(top_cusum, candidate_summary, trading, model_config, model_path, scaler_path, config_path),
        encoding="utf-8",
    )
    print(f"Selected: {best_record['strategy']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
