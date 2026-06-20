import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.trade_skip_regime_diagnostics import (
    DEFAULT_THRESHOLDS,
    _metrics_by,
    _predict_trade_probabilities,
    _safe_classification_metrics,
    _save_outputs,
)
from src import config
from src.connector.data_fetcher import load_all_price_data
from src.models.trade_skip_training import build_trade_skip_frame


def _score_threshold(metrics: dict, objective: str) -> float:
    if objective == "sum_pnl":
        return float(metrics["sum_pnl_selected_pips"])
    if objective == "avg_pnl":
        value = metrics["avg_pnl_selected_pips"]
        return float(value) if pd.notna(value) else -np.inf
    if objective == "winrate":
        value = metrics["selected_winrate_pct"]
        return float(value) if pd.notna(value) else -np.inf
    if objective == "balanced_accuracy":
        value = metrics["balanced_accuracy_pct"]
        return float(value) if pd.notna(value) else -np.inf
    raise ValueError(f"Unsupported objective: {objective}")


def _choose_threshold(
    train_frame: pd.DataFrame,
    thresholds: tuple[float, ...],
    objective: str,
    min_selected_trades: int,
) -> tuple[float, dict]:
    best_threshold = thresholds[0]
    best_metrics = None
    best_score = -np.inf

    for threshold in thresholds:
        metrics = _safe_classification_metrics(train_frame, threshold)
        if metrics["selected_trades"] < min_selected_trades:
            score = -np.inf
        else:
            score = _score_threshold(metrics, objective)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None or best_score == -np.inf:
        best_threshold = thresholds[0]
        best_metrics = _safe_classification_metrics(train_frame, best_threshold)

    return best_threshold, best_metrics


def walk_forward_thresholds(
    diagnostics: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    train_years: int = 3,
    start_year: int | None = None,
    objective: str = "sum_pnl",
    min_train_events: int = 500,
    min_selected_trades: int = 100,
    expanding: bool = False,
) -> pd.DataFrame:
    years = sorted(int(year) for year in diagnostics["year"].unique())
    if not years:
        return pd.DataFrame()

    first_test_year = years[0] + train_years
    if start_year is not None:
        first_test_year = max(first_test_year, start_year)

    rows = []
    for test_year in years:
        if test_year < first_test_year:
            continue

        if expanding:
            train_mask = diagnostics["year"] < test_year
        else:
            train_mask = diagnostics["year"].between(test_year - train_years, test_year - 1)
        test_mask = diagnostics["year"].eq(test_year)

        train_frame = diagnostics.loc[train_mask]
        test_frame = diagnostics.loc[test_mask]
        if len(train_frame) < min_train_events or test_frame.empty:
            continue

        threshold, train_metrics = _choose_threshold(
            train_frame=train_frame,
            thresholds=thresholds,
            objective=objective,
            min_selected_trades=min_selected_trades,
        )
        test_metrics = _safe_classification_metrics(test_frame, threshold)

        row = {
            "test_year": test_year,
            "train_start_year": int(train_frame["year"].min()),
            "train_end_year": int(train_frame["year"].max()),
            "threshold": threshold,
            "objective": objective,
            "train_events": len(train_frame),
            "test_events": len(test_frame),
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def walk_forward_regime_breakdown(diagnostics: pd.DataFrame, walk_forward: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if walk_forward.empty:
        return pd.DataFrame()

    for item in walk_forward.itertuples(index=False):
        test_frame = diagnostics[diagnostics["year"].eq(item.test_year)]
        regime_metrics = _metrics_by(
            test_frame,
            ["trend_regime", "vol_regime"],
            threshold=float(item.threshold),
            min_events=20,
        )
        if regime_metrics.empty:
            continue
        regime_metrics.insert(0, "test_year", item.test_year)
        regime_metrics.insert(1, "threshold", item.threshold)
        rows.append(regime_metrics)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_walk_forward(
    model_type: str,
    thresholds: tuple[float, ...],
    train_years: int,
    start_year: int | None,
    objective: str,
    min_train_events: int,
    min_selected_trades: int,
    expanding: bool,
) -> dict[str, pd.DataFrame | dict]:
    price_df, loaded_files = load_all_price_data(config.DATA_DIR)
    labeled = build_trade_skip_frame(
        price_df,
        horizon=config.DEFAULT_HORIZON_CANDLES,
        tp_threshold=config.DEFAULT_TP_THRESHOLD,
        sl_threshold=config.DEFAULT_SL_THRESHOLD,
    )
    diagnostics, model_info = _predict_trade_probabilities(labeled, model_type=model_type)
    walk_forward = walk_forward_thresholds(
        diagnostics=diagnostics,
        thresholds=thresholds,
        train_years=train_years,
        start_year=start_year,
        objective=objective,
        min_train_events=min_train_events,
        min_selected_trades=min_selected_trades,
        expanding=expanding,
    )
    regime_breakdown = walk_forward_regime_breakdown(diagnostics, walk_forward)
    yearly_static_051 = _metrics_by(diagnostics, ["year"], threshold=0.51)
    yearly_static_052 = _metrics_by(diagnostics, ["year"], threshold=0.52)

    summary = {
        "loaded_files": len(loaded_files),
        "raw_rows": len(price_df),
        "diagnostic_events": len(diagnostics),
        "model_type": model_type,
        "train_years": train_years,
        "expanding": expanding,
        "objective": objective,
        "min_train_events": min_train_events,
        "min_selected_trades": min_selected_trades,
        "folds": len(walk_forward),
        **{key: value for key, value in model_info.items() if key != "feature_columns"},
    }

    return {
        "summary": summary,
        "walk_forward": walk_forward,
        "walk_forward_by_regime": regime_breakdown,
        "yearly_static_threshold_051": yearly_static_051,
        "yearly_static_threshold_052": yearly_static_052,
    }


def _print_report(outputs: dict[str, pd.DataFrame | dict]) -> None:
    print("SUMMARY")
    for key, value in outputs["summary"].items():
        print(f"{key}: {value}")

    walk_forward = outputs["walk_forward"]
    if walk_forward.empty:
        print("\nNo walk-forward folds were produced.")
        return

    columns = [
        "test_year",
        "train_start_year",
        "train_end_year",
        "threshold",
        "train_selected_trades",
        "train_selected_winrate_pct",
        "train_sum_pnl_selected_pips",
        "test_selected_trades",
        "test_selected_winrate_pct",
        "test_sum_pnl_selected_pips",
        "test_avg_pnl_selected_pips",
    ]
    print("\nWALK_FORWARD")
    print(walk_forward[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    total_test_pnl = walk_forward["test_sum_pnl_selected_pips"].sum()
    positive_folds = (walk_forward["test_sum_pnl_selected_pips"] > 0).sum()
    print("\nAGGREGATE")
    print(f"total_test_pnl_pips: {total_test_pnl:.3f}")
    print(f"positive_folds: {positive_folds}/{len(walk_forward)}")
    print(f"median_test_pnl_pips: {walk_forward['test_sum_pnl_selected_pips'].median():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward threshold diagnostics for saved TradeSkip models.")
    parser.add_argument("--model-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--expanding", action="store_true")
    parser.add_argument(
        "--objective",
        choices=["sum_pnl", "avg_pnl", "winrate", "balanced_accuracy"],
        default="sum_pnl",
    )
    parser.add_argument("--min-train-events", type=int, default=500)
    parser.add_argument("--min-selected-trades", type=int, default=100)
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR / "processed" / "walk_forward_diagnostics",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    outputs = run_walk_forward(
        model_type=args.model_type,
        thresholds=tuple(args.thresholds),
        train_years=args.train_years,
        start_year=args.start_year,
        objective=args.objective,
        min_train_events=args.min_train_events,
        min_selected_trades=args.min_selected_trades,
        expanding=args.expanding,
    )
    _print_report(outputs)
    if not args.no_save:
        _save_outputs(outputs, args.output_dir)
        print(f"\nSaved CSV diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
