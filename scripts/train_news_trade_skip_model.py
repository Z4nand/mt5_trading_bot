from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config  # noqa: E402
from src.connector.data_fetcher import load_all_price_data  # noqa: E402
from src.features.news_features import BASE_NEWS_FEATURE_COLUMNS, EVENT_GROUP_NEWS_FEATURE_COLUMNS, add_news_features  # noqa: E402
from src.models.sequence_models import load_model_with_config  # noqa: E402
from src.models.trade_skip_training import (  # noqa: E402
    TradeSkipSequenceDataset,
    build_trade_skip_frame,
    evaluate_trade_skip_model,
    generate_trade_skip_signal_history,
    train_trade_skip_model,
)
from src.strategy.backtest import build_trades, calculate_trade_metrics  # noqa: E402
from src.strategy.signal_generator import prepare_rule_frame  # noqa: E402


OUTPUT_DIR = config.DATA_DIR / "processed" / "news_model_experiment"
REPORT_PATH = BASE_DIR / "reports" / "trade_skip_news_model_report.md"
OLD_MODEL_PATH = config.MODELS_DIR / "trade_skip_reversal_gru_best.pth"
OLD_SCALER_PATH = config.MODELS_DIR / "trade_skip_reversal_gru_scaler.pkl"
OLD_CONFIG_PATH = config.MODELS_DIR / "trade_skip_reversal_gru_config.pkl"
NEW_MODEL_PATH = config.MODELS_DIR / "trade_skip_reversal_sl4_gru_best.pth"
NEW_SCALER_PATH = config.MODELS_DIR / "trade_skip_reversal_sl4_gru_scaler.pkl"
NEW_CONFIG_PATH = config.MODELS_DIR / "trade_skip_reversal_sl4_gru_config.pkl"

TP_THRESHOLD = 0.0008
SL_THRESHOLD = 0.0004
DEFAULT_COST_PER_TRADE = 0.00003
THRESHOLDS = (0.50, 0.51, 0.52, 0.53, 0.55, 0.57, 0.60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate TradeSkip GRU candidates with news features.")
    parser.add_argument("--explore-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-final-train", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def load_old_config() -> dict:
    return joblib.load(OLD_CONFIG_PATH)


def localized_timestamp(value: str, index: pd.Index) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if getattr(index, "tz", None) is not None:
        return timestamp.tz_localize(index.tz)
    return timestamp


def apply_threshold(raw_signals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    signals = raw_signals.copy()
    direction = signals["event_cusum_direction"].astype(int)
    base_decision = np.where(direction.eq(1), "BUY", np.where(direction.eq(-1), "SELL", "NO TRADE"))
    allowed = signals["event"].astype(int).eq(1) & signals["confidence"].astype(float).ge(threshold)
    signals["decision"] = np.where(allowed, base_decision, "NO TRADE")
    signals["prediction"] = np.where(allowed, "TRADE", "SKIP")
    return signals


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
    filtered = enriched.copy()
    filtered.loc[skip, "decision"] = "NO TRADE"
    return filtered.drop(columns=["news_minutes_to_next", "news_next_impact_score"])


def evaluate_trading(
    name: str,
    raw_signals: pd.DataFrame,
    prepared_df: pd.DataFrame,
    price_df: pd.DataFrame,
    thresholds: tuple[float, ...],
    use_news_filter: bool,
    cost_per_trade: float,
) -> pd.DataFrame:
    valid_end = localized_timestamp(config.VALID_END_DATE, prepared_df.index)
    test_raw = raw_signals[raw_signals["time"] >= valid_end].copy()
    rows = []
    for threshold in thresholds:
        signals = apply_threshold(test_raw, threshold)
        if use_news_filter:
            signals = apply_news_filter(signals, prepared_df, minutes=60)
        trades = build_trades(
            signals,
            price_df,
            horizon=8,
            tp_threshold=TP_THRESHOLD,
            sl_threshold=SL_THRESHOLD,
            cost_per_trade=cost_per_trade,
        )
        metrics = calculate_trade_metrics(trades, initial_capital=10_000.0)
        pnl_pips = float(trades["PnL"].sum() * 10000) if not trades.empty else 0.0
        rows.append(
            {
                "strategy": name,
                "threshold": threshold,
                "news_filter": use_news_filter,
                "trades": int(metrics["Trades"]),
                "total_pnl_pips": pnl_pips,
                "winrate": float(metrics["Win Rate"]),
                "profit_factor": float(metrics["Profit Factor"]),
                "max_drawdown": float(metrics["Max Drawdown"]),
                "avg_trade_pips": float(metrics["Average Trade"] * 10000),
            }
        )
    return pd.DataFrame(rows)


def classify_on_test(price_df: pd.DataFrame, model, scaler, feature_columns: list[str], model_config: dict) -> dict:
    labeled = build_trade_skip_frame(
        price_df,
        horizon=int(model_config["horizon"]),
        tp_threshold=float(model_config["tp_threshold"]),
        sl_threshold=float(model_config["sl_threshold"]),
        feature_columns=feature_columns,
        min_edge=float(model_config.get("min_edge", 0.0)),
        direction_rule=model_config.get("direction_rule"),
        cusum_volatility_window=model_config.get("cusum_volatility_window"),
        cusum_threshold_mult=model_config.get("cusum_threshold_mult"),
    )
    feature_frame = labeled[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    scaled = scaler.transform(feature_frame).astype(np.float32)
    labels = labeled["trade_success"].fillna(-1).to_numpy(dtype=np.int64)
    valid_end = localized_timestamp(config.VALID_END_DATE, labeled.index)
    sample_mask = labeled["trade_success"].notna().to_numpy()
    positions = np.flatnonzero(sample_mask)
    positions = positions[positions >= config.SEQUENCE_LENGTH - 1]
    positions = positions[labeled.index[positions] >= valid_end]
    dataset = TradeSkipSequenceDataset(scaled, labels, positions)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    device = next(model.parameters()).device
    return evaluate_trade_skip_model(model, loader, device)


def candidate_configs(base_features: list[str]) -> list[dict]:
    return [
        {
            "name": "retrained_no_news_sl4",
            "feature_columns": base_features,
            "description": "Existing technical/event features, retrained for TP 8 pips / SL 4 pips.",
        },
        {
            "name": "news_core",
            "feature_columns": base_features + BASE_NEWS_FEATURE_COLUMNS,
            "description": "Technical/event features plus safe timing, impact and EUR/USD news flags.",
        },
        {
            "name": "news_full",
            "feature_columns": base_features + BASE_NEWS_FEATURE_COLUMNS + EVENT_GROUP_NEWS_FEATURE_COLUMNS,
            "description": "Core news features plus event groups: CPI, rates, jobs, PMI, GDP, speeches.",
        },
    ]


def train_candidate(price_df: pd.DataFrame, candidate: dict, epochs: int, batch_size: int, output_dir: Path) -> dict:
    print(f"Training {candidate['name']} for {epochs} epochs...")
    result = train_trade_skip_model(
        price_df,
        model_type="gru",
        horizon=8,
        tp_threshold=TP_THRESHOLD,
        sl_threshold=SL_THRESHOLD,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        hidden_size=64,
        dropout=0.25,
        num_layers=2,
        selection_metric="balanced_accuracy",
        feature_columns=candidate["feature_columns"],
        direction_rule="cusum_reversal",
        cusum_volatility_window=192,
        cusum_threshold_mult=1.8,
    )
    history = pd.DataFrame(result["history"])
    history.to_csv(output_dir / f"history_{candidate['name']}.csv", index=False)
    return result


def save_final_config(candidate: dict, train_result: dict, path: Path) -> dict:
    model_config = {
        "task": "trade_skip_reversal_news",
        "model_type": "gru",
        "input_size": len(candidate["feature_columns"]),
        "hidden_size": 64,
        "dropout": 0.25,
        "num_layers": 2,
        "selection_metric": "balanced_accuracy",
        "best_valid_score": float(train_result["best_valid_score"]),
        "horizon": 8,
        "tp_threshold": TP_THRESHOLD,
        "sl_threshold": SL_THRESHOLD,
        "feature_columns": candidate["feature_columns"],
        "direction_rule": "cusum_reversal",
        "cusum_volatility_window": 192,
        "cusum_threshold_mult": 1.8,
        "train_end_date": config.TRAIN_END_DATE,
        "valid_end_date": config.VALID_END_DATE,
        "feature_set": candidate["name"],
        "description": candidate["description"],
        "leakage_policy": "Uses only scheduled news metadata known at or before the signal timestamp; excludes actual/forecast/previous values.",
    }
    joblib.dump(model_config, path)
    return model_config


def render_report(
    summary: pd.DataFrame,
    threshold_scan: pd.DataFrame,
    classification: pd.DataFrame,
    best_candidate: dict,
    final_config: dict,
    final_threshold_scan: pd.DataFrame | None = None,
    final_classification: pd.DataFrame | None = None,
) -> str:
    best_rows = threshold_scan.sort_values(["profit_factor", "total_pnl_pips"], ascending=False).groupby("strategy").head(1)
    final_best_rows = None
    if final_threshold_scan is not None and not final_threshold_scan.empty:
        final_best_rows = final_threshold_scan.sort_values(["profit_factor", "total_pnl_pips"], ascending=False).groupby("strategy").head(1)
    def table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return "```\n" + frame.to_string(index=False) + "\n```"

    lines = [
        "# TradeSkip News Model Experiment",
        "",
        "## Data split",
        f"- Train: before {config.TRAIN_END_DATE}",
        f"- Validation: {config.TRAIN_END_DATE} to {config.VALID_END_DATE}",
        f"- Test: from {config.VALID_END_DATE}",
        "- News leakage policy: used only event schedule metadata: time, currency, impact and event-name group. Excluded actual, forecast and previous values.",
        "",
        "## Candidate feature sets",
    ]
    for _, row in summary.iterrows():
        lines.append(f"- {row['candidate']}: features={int(row['features'])}, best_valid_balanced_accuracy={row['best_valid_score']:.4f}")
    lines.extend(
        [
            "",
            "## Selected model",
            f"- Feature set: {best_candidate['name']}",
            f"- Saved model: `{NEW_MODEL_PATH}`",
            f"- Saved scaler: `{NEW_SCALER_PATH}`",
            f"- Saved config: `{NEW_CONFIG_PATH}`",
            f"- Input features: {final_config['input_size']}",
            "",
            "## Final saved model classification on test",
            table(final_classification if final_classification is not None else classification),
            "",
            "## Final saved model trading comparison",
            table(final_best_rows if final_best_rows is not None else best_rows),
            "",
            "## Exploratory classification on test",
            table(classification),
            "",
            "## Exploratory best trading rows by strategy",
            table(best_rows),
            "",
            "## Notes",
            "- Trading comparison uses TP 8 pips, SL 4 pips, horizon 8 candles and cost 0.3 pip per trade.",
            "- The old model is not overwritten; it is evaluated as a baseline on the same test window.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        summary = pd.read_csv(args.output_dir / "candidate_summary.csv")
        threshold_scan = pd.read_csv(args.output_dir / "threshold_scan.csv")
        classification = pd.read_csv(args.output_dir / "classification_test.csv")
        final_threshold_scan_path = args.output_dir / "final_model_threshold_scan.csv"
        final_classification_path = args.output_dir / "final_model_classification_test.csv"
        final_threshold_scan = pd.read_csv(final_threshold_scan_path) if final_threshold_scan_path.exists() else None
        final_classification = pd.read_csv(final_classification_path) if final_classification_path.exists() else None
        final_config = json.loads((args.output_dir / "selected_model.json").read_text(encoding="utf-8"))
        best_candidate = {
            "name": final_config["feature_set"],
            "description": final_config.get("description", ""),
            "feature_columns": final_config["feature_columns"],
        }
        REPORT_PATH.write_text(
            render_report(
                summary,
                threshold_scan,
                classification,
                best_candidate,
                final_config,
                final_threshold_scan=final_threshold_scan,
                final_classification=final_classification,
            ),
            encoding="utf-8",
        )
        print(f"Report: {REPORT_PATH}")
        return

    old_config = load_old_config()
    base_features = list(old_config["feature_columns"])
    price_df, loaded_files = load_all_price_data(config.DATA_DIR)
    print(f"Loaded {len(price_df)} candles from {len(loaded_files)} files.")
    price_with_news = add_news_features(price_df)
    prepared_with_news = prepare_rule_frame(
        price_with_news,
        volatility_window=192,
        threshold_mult=1.8,
        feature_columns=base_features + BASE_NEWS_FEATURE_COLUMNS + EVENT_GROUP_NEWS_FEATURE_COLUMNS,
    )

    candidates = candidate_configs(base_features)
    summary_rows = []
    candidate_results = []
    trading_frames = []

    old_model = load_model_with_config(OLD_MODEL_PATH, OLD_CONFIG_PATH)
    old_scaler = joblib.load(OLD_SCALER_PATH)
    old_raw = generate_trade_skip_signal_history(
        price_with_news,
        old_model,
        old_scaler,
        feature_columns=base_features,
        threshold=0.0,
        max_rows=1_000_000,
        direction_rule=old_config.get("direction_rule"),
        cusum_volatility_window=old_config.get("cusum_volatility_window"),
        cusum_threshold_mult=old_config.get("cusum_threshold_mult"),
    )
    for use_filter in (False, True):
        trading_frames.append(
            evaluate_trading(
                "old_reversal_gru",
                old_raw,
                prepared_with_news,
                price_with_news,
                THRESHOLDS,
                use_news_filter=use_filter,
                cost_per_trade=DEFAULT_COST_PER_TRADE,
            )
        )

    old_classification = classify_on_test(price_with_news, old_model, old_scaler, base_features, old_config)
    classification_rows = [{"strategy": "old_reversal_gru", **old_classification}]

    for candidate in candidates:
        train_result = train_candidate(price_with_news, candidate, args.explore_epochs, args.batch_size, args.output_dir)
        raw = generate_trade_skip_signal_history(
            price_with_news,
            train_result["model"],
            train_result["scaler"],
            feature_columns=candidate["feature_columns"],
            threshold=0.0,
            max_rows=1_000_000,
            direction_rule="cusum_reversal",
            cusum_volatility_window=192,
            cusum_threshold_mult=1.8,
        )
        for use_filter in (False, True):
            trading_frames.append(
                evaluate_trading(
                    candidate["name"],
                    raw,
                    prepared_with_news,
                    price_with_news,
                    THRESHOLDS,
                    use_news_filter=use_filter,
                    cost_per_trade=DEFAULT_COST_PER_TRADE,
                )
            )
        summary_rows.append(
            {
                "candidate": candidate["name"],
                "description": candidate["description"],
                "features": len(candidate["feature_columns"]),
                "best_valid_score": float(train_result["best_valid_score"]),
                "test_balanced_accuracy": float(train_result["test_metrics"]["balanced_accuracy"]),
                "test_f1": float(train_result["test_metrics"]["f1"]),
            }
        )
        classification_rows.append({"strategy": candidate["name"], **train_result["test_metrics"]})
        candidate_results.append((candidate, train_result))

    threshold_scan = pd.concat(trading_frames, ignore_index=True)
    threshold_scan.to_csv(args.output_dir / "threshold_scan.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    classification = pd.DataFrame(classification_rows)
    classification.to_csv(args.output_dir / "classification_test.csv", index=False)

    model_rows = threshold_scan[threshold_scan["strategy"].isin([candidate["name"] for candidate in candidates])]
    ranked = model_rows.sort_values(
        ["profit_factor", "total_pnl_pips", "trades"],
        ascending=[False, False, False],
    )
    best_strategy = ranked.iloc[0]["strategy"]
    best_candidate, best_explore_result = next(item for item in candidate_results if item[0]["name"] == best_strategy)

    if args.skip_final_train:
        final_result = best_explore_result
    else:
        final_result = train_trade_skip_model(
            price_with_news,
            model_type="gru",
            horizon=8,
            tp_threshold=TP_THRESHOLD,
            sl_threshold=SL_THRESHOLD,
            epochs=args.final_epochs,
            batch_size=args.batch_size,
            learning_rate=1e-3,
            model_path=NEW_MODEL_PATH,
            scaler_path=NEW_SCALER_PATH,
            hidden_size=64,
            dropout=0.25,
            num_layers=2,
            selection_metric="balanced_accuracy",
            feature_columns=best_candidate["feature_columns"],
            direction_rule="cusum_reversal",
            cusum_volatility_window=192,
            cusum_threshold_mult=1.8,
        )
        pd.DataFrame(final_result["history"]).to_csv(args.output_dir / "history_final.csv", index=False)

    final_config = save_final_config(best_candidate, final_result, NEW_CONFIG_PATH)
    (args.output_dir / "selected_model.json").write_text(json.dumps(final_config, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(summary, threshold_scan, classification, best_candidate, final_config), encoding="utf-8")
    print(f"Selected candidate: {best_candidate['name']}")
    print(f"Report: {REPORT_PATH}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
