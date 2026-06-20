import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.connector.data_fetcher import load_all_price_data
from src.models.sequence_models import load_model_with_config
from src.models.trade_skip_training import build_trade_skip_frame, prepare_trade_skip_arrays
from src.strategy.backtest import _resolve_exit


DEFAULT_THRESHOLDS = (0.45, 0.48, 0.50, 0.51, 0.52, 0.53, 0.55, 0.57, 0.60)


def _localized_timestamp(value: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if index.tz is not None:
        timestamp = timestamp.tz_localize(index.tz)
    return timestamp


def _split_name(index: pd.DatetimeIndex) -> np.ndarray:
    train_end = _localized_timestamp(config.TRAIN_END_DATE, index)
    valid_end = _localized_timestamp(config.VALID_END_DATE, index)
    return np.select(
        [
            index < train_end,
            (index >= train_end) & (index < valid_end),
            index >= valid_end,
        ],
        ["train_2015_2020", "valid_2021_2023", "test_2024_2026Q1"],
        default="unknown",
    )


def _max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    return float(((close - close.cummax()) / close.cummax()).min())


def _trade_skip_paths(model_type: str) -> tuple[Path, Path, Path]:
    model_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_best.pth"
    scaler_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_scaler.pkl"
    config_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_config.pkl"
    return model_path, scaler_path, config_path


def _add_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    train_mask = frame["split"].eq("train_2015_2020")
    train = frame.loc[train_mask]
    if train.empty:
        raise ValueError("Train split is empty, cannot fit regime thresholds.")

    vol_low = train["volatility_96"].quantile(0.33)
    vol_high = train["volatility_96"].quantile(0.67)
    trend_abs = train["ma_50_diff"].abs().quantile(0.60)

    result = frame.copy()
    result["vol_regime"] = np.select(
        [
            result["volatility_96"] <= vol_low,
            result["volatility_96"] >= vol_high,
        ],
        ["low_vol", "high_vol"],
        default="mid_vol",
    )
    result["trend_regime"] = np.where(
        result["ma_50_diff"] > trend_abs,
        "uptrend",
        np.where(result["ma_50_diff"] < -trend_abs, "downtrend", "range"),
    )
    result.attrs["regime_thresholds"] = {
        "volatility_96_low": float(vol_low),
        "volatility_96_high": float(vol_high),
        "abs_ma_50_diff_trend": float(trend_abs),
    }
    return result


def _split_overview(labeled: pd.DataFrame, train_indices: np.ndarray, valid_indices: np.ndarray, test_indices: np.ndarray) -> pd.DataFrame:
    frame = labeled.copy()
    frame["split"] = _split_name(frame.index)

    sample_counts = {
        "train_2015_2020": len(train_indices),
        "valid_2021_2023": len(valid_indices),
        "test_2024_2026Q1": len(test_indices),
    }

    rows = []
    for split, data in frame.groupby("split", sort=False):
        labels = data["trade_success"].dropna().astype(int)
        rows.append(
            {
                "split": split,
                "bars": len(data),
                "sample_count": sample_counts.get(split, 0),
                "start": data.index.min(),
                "end": data.index.max(),
                "close_return_pct": (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100,
                "max_drawdown_pct": _max_drawdown(data["close"]) * 100,
                "event_rate_pct": data["event"].mean() * 100,
                "events": int(data["event"].sum()),
                "trade_labels": int(labels.size),
                "trade_winrate_pct": labels.mean() * 100 if labels.size else np.nan,
                "volatility_20_bp": data["volatility_20"].mean() * 10000,
                "volatility_96_bp": data["volatility_96"].mean() * 10000,
                "range_mean_bp": data["range_pct"].mean() * 10000,
                "strong_range_pct": data["strong_range"].mean() * 100,
                "strong_body_pct": data["strong_body"].mean() * 100,
                "breakout_pct": data["breakout"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def _predict_trade_probabilities(
    labeled: pd.DataFrame,
    model_type: str,
) -> tuple[pd.DataFrame, dict]:
    model_path, scaler_path, config_path = _trade_skip_paths(model_type)
    if not model_path.exists() or not scaler_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Missing TradeSkip artifacts for model_type={model_type}")

    model_config = joblib.load(config_path)
    feature_columns = model_config["feature_columns"]
    scaler = joblib.load(scaler_path)
    model = load_model_with_config(model_path, config_path, device=torch.device("cpu"))

    feature_frame = (
        labeled[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    scaled = scaler.transform(feature_frame).astype(np.float32)

    sample_positions = np.flatnonzero(labeled["trade_success"].notna().to_numpy())
    sample_positions = sample_positions[sample_positions >= config.SEQUENCE_LENGTH - 1]

    probabilities = np.empty(len(sample_positions), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_positions), 512):
            positions = sample_positions[start : start + 512]
            windows = np.stack(
                [
                    scaled[position - config.SEQUENCE_LENGTH + 1 : position + 1]
                    for position in positions
                ]
            )
            logits = model(torch.tensor(windows, dtype=torch.float32))
            probabilities[start : start + len(positions)] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    diagnostics = labeled.iloc[sample_positions].copy()
    diagnostics["prob_trade"] = probabilities
    diagnostics["actual_trade"] = diagnostics["trade_success"].astype(int)
    diagnostics["split"] = _split_name(diagnostics.index)
    diagnostics["year"] = diagnostics.index.year

    pnls = []
    exit_reasons = []
    for position in sample_positions:
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

    diagnostics["pnl"] = pnls
    diagnostics["exit_reason"] = exit_reasons
    diagnostics = _add_regimes(diagnostics)
    model_info = {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "config_path": str(config_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        **diagnostics.attrs["regime_thresholds"],
    }
    return diagnostics, model_info


def _safe_classification_metrics(data: pd.DataFrame, threshold: float) -> dict:
    if data.empty:
        return {
            "events": 0,
            "base_winrate_pct": np.nan,
            "pred_trade_rate_pct": np.nan,
            "selected_trades": 0,
            "selected_winrate_pct": np.nan,
            "skipped_winrate_pct": np.nan,
            "avg_prob_trade": np.nan,
            "avg_pnl_all_pips": np.nan,
            "avg_pnl_selected_pips": np.nan,
            "sum_pnl_selected_pips": 0.0,
            "accuracy_pct": np.nan,
            "balanced_accuracy_pct": np.nan,
            "f1_pct": np.nan,
            "precision_pct": np.nan,
            "recall_pct": np.nan,
            "roc_auc_pct": np.nan,
        }

    result = data.copy()
    result["pred_trade"] = result["prob_trade"] >= threshold
    y_true = result["actual_trade"].to_numpy(dtype=int)
    y_pred = result["pred_trade"].to_numpy(dtype=int)
    selected = result[result["pred_trade"]]
    skipped = result[~result["pred_trade"]]

    return {
        "events": len(result),
        "base_winrate_pct": y_true.mean() * 100,
        "pred_trade_rate_pct": y_pred.mean() * 100,
        "selected_trades": len(selected),
        "selected_winrate_pct": selected["actual_trade"].mean() * 100 if not selected.empty else np.nan,
        "skipped_winrate_pct": skipped["actual_trade"].mean() * 100 if not skipped.empty else np.nan,
        "avg_prob_trade": result["prob_trade"].mean(),
        "avg_pnl_all_pips": result["pnl"].mean() * 10000,
        "avg_pnl_selected_pips": selected["pnl"].mean() * 10000 if not selected.empty else np.nan,
        "sum_pnl_selected_pips": selected["pnl"].sum() * 10000 if not selected.empty else 0.0,
        "accuracy_pct": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy_pct": balanced_accuracy_score(y_true, y_pred) * 100 if len(np.unique(y_true)) > 1 else np.nan,
        "f1_pct": f1_score(y_true, y_pred, zero_division=0) * 100,
        "precision_pct": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall_pct": recall_score(y_true, y_pred, zero_division=0) * 100,
        "roc_auc_pct": roc_auc_score(y_true, result["prob_trade"]) * 100 if len(np.unique(y_true)) > 1 else np.nan,
    }


def _metrics_by(diagnostics: pd.DataFrame, group_columns: list[str], threshold: float, min_events: int = 0) -> pd.DataFrame:
    rows = []
    for group_key, data in diagnostics.groupby(group_columns, sort=False):
        if len(data) < min_events:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        row.update(_safe_classification_metrics(data, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_scan(diagnostics: pd.DataFrame, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        for split, data in diagnostics.groupby("split", sort=False):
            row = {"threshold": threshold, "split": split}
            row.update(_safe_classification_metrics(data, threshold))
            rows.append(row)
    return pd.DataFrame(rows)


def _regime_distribution(diagnostics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        diagnostics.groupby(["split", "trend_regime", "vol_regime"], sort=False)
        .size()
        .rename("events")
        .reset_index()
    )
    grouped["split_events"] = grouped.groupby("split")["events"].transform("sum")
    grouped["event_share_pct"] = grouped["events"] / grouped["split_events"] * 100
    return grouped.drop(columns=["split_events"])


def run_diagnostics(model_type: str, threshold: float, thresholds: tuple[float, ...]) -> dict[str, pd.DataFrame | dict]:
    price_df, loaded_files = load_all_price_data(config.DATA_DIR)
    labeled = build_trade_skip_frame(
        price_df,
        horizon=config.DEFAULT_HORIZON_CANDLES,
        tp_threshold=config.DEFAULT_TP_THRESHOLD,
        sl_threshold=config.DEFAULT_SL_THRESHOLD,
    )
    _, _, train_indices, valid_indices, test_indices, _ = prepare_trade_skip_arrays(labeled)
    diagnostics, model_info = _predict_trade_probabilities(labeled, model_type=model_type)

    overview = _split_overview(labeled, train_indices, valid_indices, test_indices)
    by_split = _metrics_by(diagnostics, ["split"], threshold)
    by_year = _metrics_by(diagnostics, ["year"], threshold)
    by_regime = _metrics_by(diagnostics, ["split", "trend_regime", "vol_regime"], threshold, min_events=50)
    threshold_table = _threshold_scan(diagnostics, thresholds)
    regime_distribution = _regime_distribution(diagnostics)

    data_info = {
        "loaded_files": len(loaded_files),
        "raw_rows": len(price_df),
        "raw_start": str(price_df.index.min()),
        "raw_end": str(price_df.index.max()),
        "prepared_rows": len(labeled),
        "prepared_start": str(labeled.index.min()),
        "prepared_end": str(labeled.index.max()),
        "events": int(labeled["event"].sum()),
        "event_rate_pct": float(labeled["event"].mean() * 100),
        "trade_labels": int(labeled["trade_success"].notna().sum()),
    }
    return {
        "data_info": data_info,
        "model_info": model_info,
        "split_overview": overview,
        "by_split": by_split,
        "by_year": by_year,
        "by_regime": by_regime,
        "threshold_scan": threshold_table,
        "regime_distribution": regime_distribution,
    }


def _save_outputs(outputs: dict[str, pd.DataFrame | dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{name}.csv", index=False)
        else:
            pd.Series(value).to_frame("value").to_csv(output_dir / f"{name}.csv")


def _print_report(outputs: dict[str, pd.DataFrame | dict], threshold: float) -> None:
    print("DATA_INFO")
    for key, value in outputs["data_info"].items():
        print(f"{key}: {value}")

    print("\nMODEL_INFO")
    model_info = outputs["model_info"].copy()
    model_info.pop("feature_columns", None)
    for key, value in model_info.items():
        print(f"{key}: {value}")

    print(f"\nBY_SPLIT threshold={threshold}")
    print(outputs["by_split"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\nTHRESHOLD_SCAN")
    threshold_scan = outputs["threshold_scan"][
        [
            "threshold",
            "split",
            "selected_trades",
            "selected_winrate_pct",
            "avg_pnl_selected_pips",
            "sum_pnl_selected_pips",
        ]
    ]
    print(threshold_scan.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\nREGIME_DISTRIBUTION")
    print(outputs["regime_distribution"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose TradeSkip performance by split, year, market regime, and threshold.")
    parser.add_argument("--model-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--threshold", type=float, default=0.51)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_THRESHOLDS),
        help="Threshold values for split-level scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR / "processed" / "regime_diagnostics",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    outputs = run_diagnostics(args.model_type, args.threshold, tuple(args.thresholds))
    _print_report(outputs, args.threshold)
    if not args.no_save:
        _save_outputs(outputs, args.output_dir)
        print(f"\nSaved CSV diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
