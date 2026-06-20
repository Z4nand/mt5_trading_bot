from __future__ import annotations

import argparse
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
from src.models.trade_skip_training import TradeSkipSequenceDataset, class_weights, evaluate_trade_skip_model  # noqa: E402
from src.strategy.backtest import build_trades, calculate_trade_metrics  # noqa: E402
from scripts.trade_hold_reversal_experiments import (  # noqa: E402
    COST_PER_TRADE,
    HORIZON,
    SL_THRESHOLD,
    THRESHOLDS,
    TP_THRESHOLD,
    apply_news_filter,
    base_feature_columns,
    build_trade_hold_frame,
    raw_signals_from_model,
    threshold_signals,
)


OUTPUT_DIR = config.DATA_DIR / "processed" / "trade_hold_quality_walk_forward"
REPORT_PATH = BASE_DIR / "reports" / "trade_hold_quality_walk_forward_report.md"

CANDIDATES = (
    {"name": "positive_horizon_w192_m1.8", "label_mode": "positive_horizon", "window": 192, "mult": 1.8},
    {"name": "positive_horizon_w288_m2.1", "label_mode": "positive_horizon", "window": 288, "mult": 2.1},
)

FOLDS = (
    {"fold": "2021", "train_end": "2020-01-01", "valid_start": "2020-01-01", "valid_end": "2021-01-01", "test_start": "2021-01-01", "test_end": "2022-01-01"},
    {"fold": "2022", "train_end": "2021-01-01", "valid_start": "2021-01-01", "valid_end": "2022-01-01", "test_start": "2022-01-01", "test_end": "2023-01-01"},
    {"fold": "2023", "train_end": "2022-01-01", "valid_start": "2022-01-01", "valid_end": "2023-01-01", "test_start": "2023-01-01", "test_end": "2024-01-01"},
    {"fold": "2024", "train_end": "2023-01-01", "valid_start": "2023-01-01", "valid_end": "2024-01-01", "test_start": "2024-01-01", "test_end": "2025-01-01"},
    {"fold": "2025", "train_end": "2024-01-01", "valid_start": "2024-01-01", "valid_end": "2025-01-01", "test_start": "2025-01-01", "test_end": "2026-01-01"},
    {"fold": "2026_partial", "train_end": "2025-01-01", "valid_start": "2025-01-01", "valid_end": "2026-01-01", "test_start": "2026-01-01", "test_end": "2027-01-01"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward check for quality-feature TRADE/HOLD CUSUM reversal models.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-valid-trades", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def local_ts(value: str, index: pd.Index) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if getattr(index, "tz", None) is not None:
        return timestamp.tz_localize(index.tz)
    return timestamp


def split_indices(
    labeled_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    fold: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    labels_available = labeled_df["trade_success"].notna().to_numpy()
    sample_positions = np.flatnonzero(labels_available)
    sample_positions = sample_positions[sample_positions >= config.SEQUENCE_LENGTH - 1]

    train_end = local_ts(fold["train_end"], labeled_df.index)
    valid_start = local_ts(fold["valid_start"], labeled_df.index)
    valid_end = local_ts(fold["valid_end"], labeled_df.index)
    test_start = local_ts(fold["test_start"], labeled_df.index)
    test_end = local_ts(fold["test_end"], labeled_df.index)

    train_end_pos = labeled_df.index.searchsorted(train_end, side="left")
    valid_end_pos = labeled_df.index.searchsorted(valid_end, side="left")
    test_end_pos = labeled_df.index.searchsorted(test_end, side="left")

    index = labeled_df.index
    train_mask = index < train_end
    valid_mask = (index >= valid_start) & (index < valid_end)
    test_mask = (index >= test_start) & (index < test_end)

    train_ok = (sample_positions + HORIZON) < train_end_pos
    valid_ok = (sample_positions + HORIZON) < valid_end_pos
    test_ok = (sample_positions + HORIZON) < test_end_pos

    train_indices = sample_positions[np.asarray(train_mask)[sample_positions] & train_ok]
    valid_indices = sample_positions[np.asarray(valid_mask)[sample_positions] & valid_ok]
    test_indices = sample_positions[np.asarray(test_mask)[sample_positions] & test_ok]

    scaler = StandardScaler()
    scaler.fit(feature_frame.loc[train_mask])
    return train_indices, valid_indices, test_indices, scaler


def train_fold_model(
    labeled_df: pd.DataFrame,
    feature_columns: list[str],
    fold: dict,
    epochs: int,
    batch_size: int,
) -> dict:
    feature_frame = (
        labeled_df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    labels = labeled_df["trade_success"].fillna(-1).to_numpy(dtype=np.int64)
    train_indices, valid_indices, test_indices, scaler = split_indices(labeled_df, feature_frame, fold)
    scaled = scaler.transform(feature_frame).astype(np.float32)

    train_ds = TradeSkipSequenceDataset(scaled, labels, train_indices)
    valid_ds = TradeSkipSequenceDataset(scaled, labels, valid_indices)
    test_ds = TradeSkipSequenceDataset(scaled, labels, test_indices)
    if len(train_ds) == 0 or len(valid_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Not enough samples for {fold['fold']}: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

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
            best_score = float(metrics["balanced_accuracy"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "scaler": scaler,
        "history": pd.DataFrame(history),
        "best_valid_balanced_accuracy": best_score,
        "test_classification": evaluate_trade_skip_model(model, test_loader, device),
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
    }


def evaluate_window(
    raw_signals: pd.DataFrame,
    labeled_df: pd.DataFrame,
    price_df: pd.DataFrame,
    start: str,
    end: str,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    start_ts = local_ts(start, labeled_df.index)
    end_ts = local_ts(end, labeled_df.index)
    rows = []
    segment_raw = raw_signals[(raw_signals["time"] >= start_ts) & (raw_signals["time"] < end_ts)].copy()
    for threshold in thresholds:
        for news_filter in (False, True):
            signals = threshold_signals(segment_raw, threshold)
            if news_filter:
                signals = apply_news_filter(signals, labeled_df)
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


def select_validation_row(validation: pd.DataFrame, min_trades: int) -> pd.Series:
    eligible = validation[validation["trades"] >= min_trades].copy()
    if eligible.empty:
        eligible = validation.copy()
    eligible["score"] = (
        eligible["profit_factor"].replace(np.inf, 10).clip(lower=0, upper=10)
        + eligible["total_pnl_pips"].clip(lower=0) / 500.0
        - eligible["max_drawdown"].abs() * 5.0
    )
    return eligible.sort_values(["score", "profit_factor", "total_pnl_pips", "trades"], ascending=False).iloc[0]


def aggregate_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("candidate", as_index=False).agg(
        folds=("fold", "nunique"),
        positive_folds=("test_total_pnl_pips", lambda s: int((s > 0).sum())),
        total_test_pips=("test_total_pnl_pips", "sum"),
        median_test_pips=("test_total_pnl_pips", "median"),
        worst_test_pips=("test_total_pnl_pips", "min"),
        total_test_trades=("test_trades", "sum"),
        avg_test_pf=("test_profit_factor", "mean"),
        median_test_pf=("test_profit_factor", "median"),
        worst_test_pf=("test_profit_factor", "min"),
        max_abs_drawdown=("test_max_drawdown", "min"),
    )
    return grouped.sort_values(["positive_folds", "total_test_pips", "median_test_pf"], ascending=False)


def render_report(summary: pd.DataFrame, selected_rows: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Trade/Hold Quality Walk-Forward",
            "",
            "## Setup",
            "- Architecture: CUSUM reversal gives direction, GRU predicts TRADE/HOLD.",
            "- Features: base TradeSkip features + reversal quality features.",
            "- Label: positive_horizon.",
            "- Validation selects threshold/news filter; test year is unseen for that fold.",
            f"- Cost per trade: {COST_PER_TRADE * 10000:.2f} pips, TP: {TP_THRESHOLD * 10000:.1f} pips, SL: {SL_THRESHOLD * 10000:.1f} pips.",
            "",
            "## Summary",
            "```",
            summary.to_string(index=False),
            "```",
            "",
            "## Fold Results",
            "```",
            selected_rows.to_string(index=False),
            "```",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Torch device:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "| cuda_available=",
        torch.cuda.is_available(),
    )
    feature_columns = base_feature_columns() + [
        column for column in REVERSAL_QUALITY_FEATURE_COLUMNS if column not in base_feature_columns()
    ]
    price_df, loaded = load_all_price_data(config.DATA_DIR)
    price_df = add_news_features(price_df)
    print(f"Loaded {len(price_df)} candles from {len(loaded)} files.")

    selected_rows = []
    validation_frames = []
    test_frames = []

    for candidate in CANDIDATES:
        print(f"Preparing {candidate['name']}...")
        labeled = build_trade_hold_frame(
            price_df,
            feature_columns,
            volatility_window=int(candidate["window"]),
            threshold_mult=float(candidate["mult"]),
            label_mode=str(candidate["label_mode"]),
        )
        for fold in FOLDS:
            test_start = local_ts(fold["test_start"], labeled.index)
            if test_start >= labeled.index.max():
                continue
            print(f"Training {candidate['name']} fold {fold['fold']}...")
            result = train_fold_model(labeled, feature_columns, fold, epochs=args.epochs, batch_size=args.batch_size)
            result["history"].to_csv(args.output_dir / f"history_{candidate['name']}_{fold['fold']}.csv", index=False)
            raw = raw_signals_from_model(labeled, result["model"], result["scaler"], feature_columns)

            validation = evaluate_window(raw, labeled, price_df, fold["valid_start"], fold["valid_end"])
            validation["candidate"] = candidate["name"]
            validation["fold"] = fold["fold"]
            validation_frames.append(validation)
            selected = select_validation_row(validation, args.min_valid_trades)

            test = evaluate_window(raw, labeled, price_df, fold["test_start"], fold["test_end"], thresholds=(float(selected["threshold"]),))
            test = test[test["news_filter"].eq(bool(selected["news_filter"]))].copy()
            test["candidate"] = candidate["name"]
            test["fold"] = fold["fold"]
            test_frames.append(test)

            row = {
                "candidate": candidate["name"],
                "fold": fold["fold"],
                "train_end": fold["train_end"],
                "valid_window": f"{fold['valid_start']}..{fold['valid_end']}",
                "test_window": f"{fold['test_start']}..{fold['test_end']}",
                "selected_threshold": float(selected["threshold"]),
                "selected_news_filter": bool(selected["news_filter"]),
                "valid_trades": int(selected["trades"]),
                "valid_total_pnl_pips": float(selected["total_pnl_pips"]),
                "valid_profit_factor": float(selected["profit_factor"]),
                "test_trades": int(test.iloc[0]["trades"]) if not test.empty else 0,
                "test_total_pnl_pips": float(test.iloc[0]["total_pnl_pips"]) if not test.empty else 0.0,
                "test_winrate": float(test.iloc[0]["winrate"]) if not test.empty else 0.0,
                "test_profit_factor": float(test.iloc[0]["profit_factor"]) if not test.empty else 0.0,
                "test_max_drawdown": float(test.iloc[0]["max_drawdown"]) if not test.empty else 0.0,
                "test_avg_trade_pips": float(test.iloc[0]["avg_trade_pips"]) if not test.empty else 0.0,
                "train_samples": int(result["train_samples"]),
                "valid_samples": int(result["valid_samples"]),
                "test_samples": int(result["test_samples"]),
                "test_balanced_accuracy": float(result["test_classification"]["balanced_accuracy"]),
            }
            selected_rows.append(row)

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(args.output_dir / "selected_fold_results.csv", index=False)
    validation_all = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    test_all = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    validation_all.to_csv(args.output_dir / "validation_threshold_scan.csv", index=False)
    test_all.to_csv(args.output_dir / "test_selected_rows.csv", index=False)
    summary = aggregate_metrics(selected_df)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    REPORT_PATH.write_text(render_report(summary, selected_df), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
