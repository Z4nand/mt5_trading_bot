import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.trade_skip_regime_diagnostics import (
    DEFAULT_THRESHOLDS,
    _safe_classification_metrics,
    _save_outputs,
)
from scripts.event_direction_diagnostics import (
    _build_event_frame,
    _direction_for_row,
    _fit_regime_thresholds,
)
from scripts.trade_skip_walk_forward import _choose_threshold
from src import config
from src.connector.data_fetcher import load_all_price_data
from src.models.sequence_models import create_model
from src.models.trade_skip_training import (
    TradeSkipSequenceDataset,
    build_trade_skip_frame,
    class_weights,
    evaluate_trade_skip_model,
)
from src.strategy.backtest import _resolve_exit


def _localized_year_start(year: int, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(year=year, month=1, day=1)
    if index.tz is not None:
        timestamp = timestamp.tz_localize(index.tz)
    return timestamp


def _load_trade_skip_config(model_type: str) -> dict:
    config_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing TradeSkip config: {config_path}")
    return joblib.load(config_path)


def _build_trade_skip_frame_for_direction(
    price_df: pd.DataFrame,
    feature_columns: list[str],
    direction_rule: str,
    cusum_volatility_window: int,
    cusum_threshold_mult: float,
    min_edge_pips: float,
) -> pd.DataFrame:
    if (
        direction_rule == "cusum_momentum"
        and cusum_volatility_window == config.CUSUM_VOLATILITY_WINDOW
        and cusum_threshold_mult == config.CUSUM_THRESHOLD_MULT
    ):
        return build_trade_skip_frame(
            price_df,
            horizon=config.DEFAULT_HORIZON_CANDLES,
            tp_threshold=config.DEFAULT_TP_THRESHOLD,
            sl_threshold=config.DEFAULT_SL_THRESHOLD,
            feature_columns=feature_columns,
            min_edge=min_edge_pips * 0.0001,
        )

    frame = _build_event_frame(
        price_df,
        volatility_window=cusum_volatility_window,
        threshold_mult=cusum_threshold_mult,
    )
    thresholds = _fit_regime_thresholds(frame)
    labels = []
    rule_decisions = []
    encoded_directions = []
    min_edge = min_edge_pips * 0.0001

    for position in range(len(frame)):
        if int(frame.iloc[position]["event"]) != 1:
            labels.append(None)
            rule_decisions.append(None)
            encoded_directions.append(0)
            continue

        direction = _direction_for_row(frame.iloc[position], direction_rule, thresholds)
        rule_decisions.append(direction)
        if direction is None:
            labels.append(None)
            encoded_directions.append(0)
            continue

        encoded_directions.append(1 if direction == "BUY" else -1)
        entry_price = float(frame.iloc[position]["close"])
        exit_idx, exit_price, _ = _resolve_exit(
            price_df=frame,
            entry_idx=position,
            direction=direction,
            horizon=config.DEFAULT_HORIZON_CANDLES,
            tp_threshold=config.DEFAULT_TP_THRESHOLD,
            sl_threshold=config.DEFAULT_SL_THRESHOLD,
        )
        if exit_idx <= position:
            labels.append(None)
            continue
        pnl = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        labels.append(int(pnl > min_edge))

    result = frame.copy()
    result["trade_success"] = labels
    result["rule_decision"] = rule_decisions
    result["original_event_cusum_direction"] = result["event_cusum_direction"]
    result["event_cusum_direction"] = encoded_directions
    return result


def _period_positions(
    labeled: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
) -> np.ndarray:
    mask = labeled["trade_success"].notna() & (labeled.index >= start) & (labeled.index < end)
    positions = np.flatnonzero(mask.to_numpy())
    positions = positions[positions >= config.SEQUENCE_LENGTH - 1]
    end_pos = labeled.index.searchsorted(end, side="left")
    positions = positions[(positions + horizon) < end_pos]
    return positions


def _prepare_fold_arrays(
    labeled: pd.DataFrame,
    feature_columns: list[str],
    train_start_year: int,
    valid_year: int,
    test_year: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    train_start = _localized_year_start(train_start_year, labeled.index)
    valid_start = _localized_year_start(valid_year, labeled.index)
    test_start = _localized_year_start(test_year, labeled.index)
    test_end = _localized_year_start(test_year + 1, labeled.index)

    feature_frame = (
        labeled[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    labels = labeled["trade_success"].fillna(-1).to_numpy(dtype=np.int64)

    train_time_mask = (labeled.index >= train_start) & (labeled.index < valid_start)
    scaler = StandardScaler()
    scaler.fit(feature_frame.loc[train_time_mask])
    scaled = scaler.transform(feature_frame).astype(np.float32)

    train_indices = _period_positions(labeled, train_start, valid_start, horizon)
    valid_indices = _period_positions(labeled, valid_start, test_start, horizon)
    test_indices = _period_positions(labeled, test_start, test_end, horizon)
    return scaled, labels, train_indices, valid_indices, test_indices, scaler


def _predict_probabilities(
    model: torch.nn.Module,
    scaled: np.ndarray,
    sample_indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    probabilities = np.empty(len(sample_indices), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_indices), 512):
            positions = sample_indices[start : start + 512]
            windows = np.stack(
                [
                    scaled[position - config.SEQUENCE_LENGTH + 1 : position + 1]
                    for position in positions
                ]
            )
            logits = model(torch.tensor(windows, dtype=torch.float32).to(device))
            probabilities[start : start + len(positions)] = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return probabilities


def _diagnostics_for_positions(
    labeled: pd.DataFrame,
    sample_indices: np.ndarray,
    probabilities: np.ndarray,
    split: str,
    fold: int,
    threshold: float,
) -> pd.DataFrame:
    rows = labeled.iloc[sample_indices].copy()
    rows["fold"] = fold
    rows["fold_split"] = split
    rows["year"] = rows.index.year
    rows["prob_trade"] = probabilities
    rows["actual_trade"] = rows["trade_success"].astype(int)
    rows["threshold"] = threshold

    pnls = []
    exit_reasons = []
    for position in sample_indices:
        direction = "BUY" if int(labeled.iloc[position]["event_cusum_direction"]) == 1 else "SELL"
        entry_price = float(labeled.iloc[position]["close"])
        _, exit_price, exit_reason = _resolve_exit(
            price_df=labeled,
            entry_idx=position,
            direction=direction,
            horizon=config.DEFAULT_HORIZON_CANDLES,
            tp_threshold=config.DEFAULT_TP_THRESHOLD,
            sl_threshold=config.DEFAULT_SL_THRESHOLD,
        )
        pnl = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        pnls.append(pnl)
        exit_reasons.append(exit_reason)

    rows["pnl"] = pnls
    rows["exit_reason"] = exit_reasons
    return rows


def _train_fold(
    labeled: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    hidden_size: int,
    dropout: float,
    num_layers: int,
    train_start_year: int,
    valid_year: int,
    test_year: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    selection_metric: str,
    patience: int,
    thresholds: tuple[float, ...],
    threshold_objective: str,
    min_selected_trades: int,
    device: torch.device,
    fold: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    scaled, labels, train_indices, valid_indices, test_indices, _ = _prepare_fold_arrays(
        labeled=labeled,
        feature_columns=feature_columns,
        train_start_year=train_start_year,
        valid_year=valid_year,
        test_year=test_year,
        horizon=config.DEFAULT_HORIZON_CANDLES,
    )
    if len(train_indices) == 0 or len(valid_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"Not enough samples for fold {fold}: train={len(train_indices)}, "
            f"valid={len(valid_indices)}, test={len(test_indices)}"
        )

    train_ds = TradeSkipSequenceDataset(scaled, labels, train_indices)
    valid_ds = TradeSkipSequenceDataset(scaled, labels, valid_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = create_model(
        model_type=model_type,
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(labels, train_indices, device))

    best_state = None
    best_score = -np.inf
    epochs_without_improvement = 0
    history_rows = []
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
        metrics["fold"] = fold
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / max(len(train_ds), 1)
        history_rows.append(metrics)
        score = metrics[selection_metric]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_probabilities = _predict_probabilities(model, scaled, valid_indices, device)
    valid_diagnostics = _diagnostics_for_positions(
        labeled, valid_indices, valid_probabilities, split="valid", fold=fold, threshold=0.0
    )
    threshold, valid_threshold_metrics = _choose_threshold(
        valid_diagnostics,
        thresholds=thresholds,
        objective=threshold_objective,
        min_selected_trades=min_selected_trades,
    )

    train_probabilities = _predict_probabilities(model, scaled, train_indices, device)
    test_probabilities = _predict_probabilities(model, scaled, test_indices, device)
    train_diagnostics = _diagnostics_for_positions(
        labeled, train_indices, train_probabilities, split="train", fold=fold, threshold=threshold
    )
    test_diagnostics = _diagnostics_for_positions(
        labeled, test_indices, test_probabilities, split="test", fold=fold, threshold=threshold
    )

    train_metrics = _safe_classification_metrics(train_diagnostics, threshold)
    test_metrics = _safe_classification_metrics(test_diagnostics, threshold)
    row = {
        "fold": fold,
        "train_start_year": train_start_year,
        "train_end_year": valid_year - 1,
        "valid_year": valid_year,
        "test_year": test_year,
        "threshold": threshold,
        "best_valid_score": best_score,
        "epochs_ran": len(history_rows),
        "train_samples": len(train_indices),
        "valid_samples": len(valid_indices),
        "test_samples": len(test_indices),
    }
    row.update({f"valid_select_{key}": value for key, value in valid_threshold_metrics.items()})
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    diagnostics = pd.concat([train_diagnostics, valid_diagnostics, test_diagnostics], ignore_index=False)
    history = pd.DataFrame(history_rows)
    return row, diagnostics, history


def run_walk_forward_retraining(
    model_type: str,
    start_year: int | None,
    end_year: int | None,
    train_years: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    dropout: float,
    num_layers: int,
    selection_metric: str,
    patience: int,
    thresholds: tuple[float, ...],
    threshold_objective: str,
    min_selected_trades: int,
    min_edge_pips: float,
    direction_rule: str,
    cusum_volatility_window: int,
    cusum_threshold_mult: float,
    seed: int,
) -> dict[str, pd.DataFrame | dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_config = _load_trade_skip_config(model_type)
    feature_columns = model_config["feature_columns"]
    price_df, loaded_files = load_all_price_data(config.DATA_DIR)
    labeled = _build_trade_skip_frame_for_direction(
        price_df=price_df,
        feature_columns=feature_columns,
        direction_rule=direction_rule,
        cusum_volatility_window=cusum_volatility_window,
        cusum_threshold_mult=cusum_threshold_mult,
        min_edge_pips=min_edge_pips,
    )

    years = sorted(int(year) for year in pd.Index(labeled.index.year).unique())
    first_test_year = years[0] + train_years
    if start_year is not None:
        first_test_year = max(first_test_year, start_year)
    last_test_year = years[-1] if end_year is None else min(end_year, years[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    diagnostic_frames = []
    history_frames = []
    fold = 0
    for test_year in range(first_test_year, last_test_year + 1):
        valid_year = test_year - 1
        train_start_year = test_year - train_years
        fold += 1
        row, diagnostics, history = _train_fold(
            labeled=labeled,
            feature_columns=feature_columns,
            model_type=model_type,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            train_start_year=train_start_year,
            valid_year=valid_year,
            test_year=test_year,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            selection_metric=selection_metric,
            patience=patience,
            thresholds=thresholds,
            threshold_objective=threshold_objective,
            min_selected_trades=min_selected_trades,
            device=device,
            fold=fold,
        )
        fold_rows.append(row)
        diagnostic_frames.append(diagnostics)
        history_frames.append(history)
        print(
            f"fold={fold} test_year={test_year} threshold={row['threshold']:.3f} "
            f"test_pnl={row['test_sum_pnl_selected_pips']:.1f} "
            f"test_trades={row['test_selected_trades']}"
        )

    folds = pd.DataFrame(fold_rows)
    diagnostics_all = pd.concat(diagnostic_frames, ignore_index=False) if diagnostic_frames else pd.DataFrame()
    history_all = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    summary = {
        "loaded_files": len(loaded_files),
        "raw_rows": len(price_df),
        "prepared_rows": len(labeled),
        "model_type": model_type,
        "device": str(device),
        "train_years": train_years,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "num_layers": num_layers,
        "selection_metric": selection_metric,
        "threshold_objective": threshold_objective,
        "min_edge_pips": min_edge_pips,
        "direction_rule": direction_rule,
        "cusum_volatility_window": cusum_volatility_window,
        "cusum_threshold_mult": cusum_threshold_mult,
        "folds": len(folds),
        "total_test_pnl_selected_pips": float(folds["test_sum_pnl_selected_pips"].sum()) if not folds.empty else 0.0,
        "positive_test_folds": int((folds["test_sum_pnl_selected_pips"] > 0).sum()) if not folds.empty else 0,
        "seed": seed,
    }
    return {
        "summary": summary,
        "folds": folds,
        "fold_diagnostics": diagnostics_all,
        "training_history": history_all,
    }


def _print_report(outputs: dict[str, pd.DataFrame | dict]) -> None:
    print("\nSUMMARY")
    for key, value in outputs["summary"].items():
        print(f"{key}: {value}")

    folds = outputs["folds"]
    if folds.empty:
        return
    columns = [
        "test_year",
        "train_start_year",
        "train_end_year",
        "valid_year",
        "threshold",
        "best_valid_score",
        "epochs_ran",
        "valid_select_selected_trades",
        "valid_select_sum_pnl_selected_pips",
        "test_selected_trades",
        "test_selected_winrate_pct",
        "test_sum_pnl_selected_pips",
        "test_avg_pnl_selected_pips",
    ]
    print("\nFOLDS")
    print(folds[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward retraining diagnostics for TradeSkip models.")
    parser.add_argument("--model-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "balanced_accuracy", "f1", "precision", "recall"],
        default="balanced_accuracy",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--threshold-objective",
        choices=["sum_pnl", "avg_pnl", "winrate", "balanced_accuracy"],
        default="sum_pnl",
    )
    parser.add_argument("--min-selected-trades", type=int, default=50)
    parser.add_argument("--min-edge-pips", type=float, default=0.0)
    parser.add_argument("--direction-rule", default="cusum_momentum")
    parser.add_argument("--cusum-volatility-window", type=int, default=config.CUSUM_VOLATILITY_WINDOW)
    parser.add_argument("--cusum-threshold-mult", type=float, default=config.CUSUM_THRESHOLD_MULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR / "processed" / "walk_forward_retrain",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    outputs = run_walk_forward_retraining(
        model_type=args.model_type,
        start_year=args.start_year,
        end_year=args.end_year,
        train_years=args.train_years,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
        selection_metric=args.selection_metric,
        patience=args.patience,
        thresholds=tuple(args.thresholds),
        threshold_objective=args.threshold_objective,
        min_selected_trades=args.min_selected_trades,
        min_edge_pips=args.min_edge_pips,
        direction_rule=args.direction_rule,
        cusum_volatility_window=args.cusum_volatility_window,
        cusum_threshold_mult=args.cusum_threshold_mult,
        seed=args.seed,
    )
    _print_report(outputs)
    if not args.no_save:
        _save_outputs(outputs, args.output_dir)
        print(f"\nSaved CSV diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
