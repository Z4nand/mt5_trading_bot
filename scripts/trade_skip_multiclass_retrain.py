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
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.trade_skip_regime_diagnostics import DEFAULT_THRESHOLDS, _save_outputs
from scripts.trade_skip_walk_forward_retrain import _localized_year_start, _period_positions
from src import config
from src.connector.data_fetcher import load_all_price_data
from src.models.sequence_models import create_model
from src.models.trade_skip_training import TradeSkipSequenceDataset, build_trade_skip_frame
from src.strategy.backtest import _resolve_exit


def _load_feature_columns(model_type: str) -> list[str]:
    config_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_config.pkl"
    return joblib.load(config_path)["feature_columns"]


def _add_event_pnl(labeled: pd.DataFrame) -> pd.DataFrame:
    result = labeled.copy()
    pnls = []
    for idx in range(len(result)):
        if pd.isna(result.iloc[idx]["trade_success"]):
            pnls.append(np.nan)
            continue
        direction = "BUY" if int(result.iloc[idx]["event_cusum_direction"]) == 1 else "SELL"
        entry_price = float(result.iloc[idx]["close"])
        _, exit_price, _ = _resolve_exit(
            price_df=result,
            entry_idx=idx,
            direction=direction,
            horizon=config.DEFAULT_HORIZON_CANDLES,
            tp_threshold=config.DEFAULT_TP_THRESHOLD,
            sl_threshold=config.DEFAULT_SL_THRESHOLD,
        )
        pnl = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        pnls.append(pnl)
    result["event_pnl"] = pnls
    return result


def _add_multiclass_labels(labeled: pd.DataFrame, neutral_band_pips: float) -> pd.DataFrame:
    band = neutral_band_pips * 0.0001
    result = labeled.copy()
    result["trade_quality"] = np.nan
    result.loc[result["event_pnl"] <= -band, "trade_quality"] = 0
    result.loc[result["event_pnl"].between(-band, band, inclusive="neither"), "trade_quality"] = 1
    result.loc[result["event_pnl"] >= band, "trade_quality"] = 2
    return result


def _prepare_arrays(
    labeled: pd.DataFrame,
    feature_columns: list[str],
    train_start_year: int,
    valid_year: int,
    test_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_start = _localized_year_start(train_start_year, labeled.index)
    valid_start = _localized_year_start(valid_year, labeled.index)
    test_start = _localized_year_start(test_year, labeled.index)
    test_end = _localized_year_start(test_year + 1, labeled.index)

    features = (
        labeled[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    train_mask = (labeled.index >= train_start) & (labeled.index < valid_start)
    scaler = StandardScaler()
    scaler.fit(features.loc[train_mask])
    scaled = scaler.transform(features).astype(np.float32)

    labels = labeled["trade_quality"].fillna(-1).to_numpy(dtype=np.int64)
    position_frame = labeled.copy()
    position_frame["trade_success"] = position_frame["trade_quality"]
    train_indices = _period_positions(position_frame, train_start, valid_start, config.DEFAULT_HORIZON_CANDLES)
    valid_indices = _period_positions(position_frame, valid_start, test_start, config.DEFAULT_HORIZON_CANDLES)
    test_indices = _period_positions(position_frame, test_start, test_end, config.DEFAULT_HORIZON_CANDLES)
    return scaled, labels, train_indices, valid_indices, test_indices


def _class_weights(labels: np.ndarray, train_indices: np.ndarray, device: torch.device) -> torch.Tensor | None:
    y = labels[train_indices]
    y = y[y >= 0]
    counts = np.bincount(y, minlength=3).astype(np.float32)
    if np.any(counts == 0):
        return None
    weights = counts.sum() / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _evaluate_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())
    if not y_true:
        return {"balanced_accuracy": 0.0, "macro_f1": 0.0}
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _predict_good_prob(model: torch.nn.Module, scaled: np.ndarray, indices: np.ndarray, device: torch.device) -> np.ndarray:
    probabilities = np.empty(len(indices), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            positions = indices[start : start + 512]
            windows = np.stack([scaled[pos - config.SEQUENCE_LENGTH + 1 : pos + 1] for pos in positions])
            logits = model(torch.tensor(windows, dtype=torch.float32).to(device))
            probabilities[start : start + len(positions)] = torch.softmax(logits, dim=1)[:, 2].cpu().numpy()
    return probabilities


def _diagnostics(labeled: pd.DataFrame, indices: np.ndarray, prob_good: np.ndarray, fold: int, split: str) -> pd.DataFrame:
    frame = labeled.iloc[indices].copy()
    frame["fold"] = fold
    frame["fold_split"] = split
    frame["year"] = frame.index.year
    frame["prob_good"] = prob_good
    return frame


def _threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    selected = frame[frame["prob_good"] >= threshold]
    return {
        "events": len(frame),
        "selected_trades": len(selected),
        "selected_rate_pct": len(selected) / len(frame) * 100 if len(frame) else np.nan,
        "selected_good_rate_pct": (selected["trade_quality"].eq(2).mean() * 100) if len(selected) else np.nan,
        "selected_avg_pnl_pips": selected["event_pnl"].mean() * 10000 if len(selected) else np.nan,
        "selected_sum_pnl_pips": selected["event_pnl"].sum() * 10000 if len(selected) else 0.0,
    }


def _choose_threshold(frame: pd.DataFrame, thresholds: tuple[float, ...], min_selected_trades: int) -> tuple[float, dict]:
    best_threshold = thresholds[0]
    best_metrics = _threshold_metrics(frame, best_threshold)
    best_score = -np.inf
    for threshold in thresholds:
        metrics = _threshold_metrics(frame, threshold)
        score = metrics["selected_sum_pnl_pips"] if metrics["selected_trades"] >= min_selected_trades else -np.inf
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def _train_fold(
    labeled: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    train_start_year: int,
    valid_year: int,
    test_year: int,
    fold: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    dropout: float,
    num_layers: int,
    thresholds: tuple[float, ...],
    min_selected_trades: int,
    device: torch.device,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    scaled, labels, train_indices, valid_indices, test_indices = _prepare_arrays(
        labeled,
        feature_columns,
        train_start_year,
        valid_year,
        test_year,
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
        num_classes=3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(labels, train_indices, device))

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
        metrics = _evaluate_classifier(model, valid_loader, device)
        metrics.update({"fold": fold, "epoch": epoch, "train_loss": train_loss / max(len(train_ds), 1)})
        history_rows.append(metrics)
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_diag = _diagnostics(labeled, valid_indices, _predict_good_prob(model, scaled, valid_indices, device), fold, "valid")
    threshold, valid_threshold_metrics = _choose_threshold(valid_diag, thresholds, min_selected_trades)
    train_diag = _diagnostics(labeled, train_indices, _predict_good_prob(model, scaled, train_indices, device), fold, "train")
    test_diag = _diagnostics(labeled, test_indices, _predict_good_prob(model, scaled, test_indices, device), fold, "test")
    train_metrics = _threshold_metrics(train_diag, threshold)
    test_metrics = _threshold_metrics(test_diag, threshold)

    row = {
        "fold": fold,
        "train_start_year": train_start_year,
        "train_end_year": valid_year - 1,
        "valid_year": valid_year,
        "test_year": test_year,
        "threshold": threshold,
        "best_valid_balanced_accuracy": best_score,
        "epochs_ran": len(history_rows),
        "train_samples": len(train_indices),
        "valid_samples": len(valid_indices),
        "test_samples": len(test_indices),
    }
    row.update({f"valid_select_{key}": value for key, value in valid_threshold_metrics.items()})
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    diagnostics = pd.concat([train_diag, valid_diag, test_diag])
    return row, diagnostics, pd.DataFrame(history_rows)


def run_multiclass(
    neutral_band_pips: float,
    model_type: str,
    start_year: int,
    end_year: int,
    train_years: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    dropout: float,
    num_layers: int,
    thresholds: tuple[float, ...],
    min_selected_trades: int,
    seed: int,
) -> dict[str, pd.DataFrame | dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    feature_columns = _load_feature_columns(model_type)
    price_df, loaded_files = load_all_price_data(config.DATA_DIR)
    labeled = build_trade_skip_frame(price_df, feature_columns=feature_columns)
    labeled = _add_multiclass_labels(_add_event_pnl(labeled), neutral_band_pips)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows = []
    diagnostics = []
    histories = []
    fold = 0
    for test_year in range(start_year, end_year + 1):
        fold += 1
        row, fold_diag, history = _train_fold(
            labeled=labeled,
            feature_columns=feature_columns,
            model_type=model_type,
            train_start_year=test_year - train_years,
            valid_year=test_year - 1,
            test_year=test_year,
            fold=fold,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            thresholds=thresholds,
            min_selected_trades=min_selected_trades,
            device=device,
        )
        fold_rows.append(row)
        diagnostics.append(fold_diag)
        histories.append(history)
        print(
            f"fold={fold} test_year={test_year} threshold={row['threshold']:.3f} "
            f"test_pnl={row['test_selected_sum_pnl_pips']:.1f} "
            f"test_trades={row['test_selected_trades']}"
        )

    folds = pd.DataFrame(fold_rows)
    summary = {
        "loaded_files": len(loaded_files),
        "raw_rows": len(price_df),
        "neutral_band_pips": neutral_band_pips,
        "model_type": model_type,
        "device": str(device),
        "train_years": train_years,
        "epochs": epochs,
        "thresholds": ",".join(str(item) for item in thresholds),
        "folds": len(folds),
        "total_test_pnl_selected_pips": float(folds["test_selected_sum_pnl_pips"].sum()) if not folds.empty else 0.0,
        "positive_test_folds": int((folds["test_selected_sum_pnl_pips"] > 0).sum()) if not folds.empty else 0,
        "seed": seed,
    }
    return {
        "summary": summary,
        "folds": folds,
        "fold_diagnostics": pd.concat(diagnostics) if diagnostics else pd.DataFrame(),
        "training_history": pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(),
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
        "threshold",
        "valid_select_selected_trades",
        "valid_select_selected_sum_pnl_pips",
        "test_selected_trades",
        "test_selected_good_rate_pct",
        "test_selected_sum_pnl_pips",
        "test_selected_avg_pnl_pips",
    ]
    print("\nFOLDS")
    print(folds[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="3-class TradeSkip walk-forward retraining.")
    parser.add_argument("--neutral-band-pips", type=float, default=0.5)
    parser.add_argument("--model-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--min-selected-trades", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR / "processed" / "multiclass_retrain",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    outputs = run_multiclass(
        neutral_band_pips=args.neutral_band_pips,
        model_type=args.model_type,
        start_year=args.start_year,
        end_year=args.end_year,
        train_years=args.train_years,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
        thresholds=tuple(args.thresholds),
        min_selected_trades=args.min_selected_trades,
        seed=args.seed,
    )
    _print_report(outputs)
    if not args.no_save:
        _save_outputs(outputs, args.output_dir)
        print(f"\nSaved CSV diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
