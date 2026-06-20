from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.connector.data_fetcher import load_all_price_data
from src.models.sequence_models import load_model_with_config
from src.models.trade_skip_training import generate_trade_skip_signal_history
from src.strategy.backtest import build_trades
from src.strategy.signal_generator import generate_rule_based_signal_history, load_scaler, prepare_rule_frame


NEWS_PATH = BASE_DIR / "data" / "economic_calendar" / "economic_calendar.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "news_impact_diagnostics"
DEFAULT_THRESHOLD = 0.53
DEFAULT_WINDOWS = (15, 30, 60, 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose strategy performance around economic news windows.")
    parser.add_argument("--news-path", type=Path, default=NEWS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--min-impact", choices=["low", "medium", "high"], default="medium")
    return parser.parse_args()


def to_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None:
        return timestamps.tz_localize("UTC")
    return timestamps.tz_convert("UTC")


def load_news(path: Path, min_impact: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Economic calendar not found: {path}")

    news = pd.read_csv(path)
    news["time_utc"] = pd.to_datetime(news["time_utc"], utc=True)
    news["impact"] = news["impact"].astype(str).str.lower()
    news["currency"] = news["currency"].astype(str).str.upper()
    news["country"] = news["country"].astype(str).str.upper()
    news["event_name"] = news["event_name"].astype(str)

    impact_rank = {"low": 1, "medium": 2, "high": 3}
    news["impact_rank"] = news["impact"].map(impact_rank).fillna(0).astype(int)
    news = news[news["currency"].isin(["EUR", "USD"])]
    news = news[news["impact_rank"] >= impact_rank[min_impact]]
    news = news.drop_duplicates(subset=["time_utc", "country", "impact", "actual", "forecast", "previous", "unit"])
    return news.sort_values("time_utc").reset_index(drop=True)


def apply_trade_skip_threshold(signals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if signals.empty:
        return signals
    updated = signals.copy()
    can_trade = updated["event"].eq(1) & updated["probability_trade"].ge(threshold)
    updated["decision"] = "NO TRADE"
    updated.loc[can_trade & updated["event_cusum_direction"].eq(1), "decision"] = "BUY"
    updated.loc[can_trade & updated["event_cusum_direction"].eq(-1), "decision"] = "SELL"
    updated["confidence"] = updated["probability_trade"]
    return updated


def trade_skip_reversal_paths() -> tuple[Path, Path, Path]:
    return (
        config.TRADE_SKIP_REVERSAL_GRU_MODEL_PATH,
        config.TRADE_SKIP_REVERSAL_GRU_SCALER_PATH,
        config.TRADE_SKIP_REVERSAL_GRU_CONFIG_PATH,
    )


def build_strategy_trades(price_df: pd.DataFrame, prepared_df: pd.DataFrame, threshold: float) -> dict[str, pd.DataFrame]:
    valid_end = pd.Timestamp(config.VALID_END_DATE, tz="UTC")
    test_rows = int((prepared_df.index >= valid_end).sum())
    max_rows = max(test_rows, 1)

    rule_signals = generate_rule_based_signal_history(price_df, max_rows=max_rows)
    rule_trades = build_trades(
        rule_signals,
        prepared_df,
        horizon=config.DEFAULT_HORIZON_CANDLES,
        tp_threshold=config.DEFAULT_TP_THRESHOLD,
        sl_threshold=config.DEFAULT_SL_THRESHOLD,
    )

    model_path, scaler_path, model_config_path = trade_skip_reversal_paths()
    model_config = joblib.load(model_config_path)
    model = load_model_with_config(model_path, model_config_path)
    scaler = load_scaler(scaler_path)
    raw_signals = generate_trade_skip_signal_history(
        price_df,
        model=model,
        scaler=scaler,
        feature_columns=model_config["feature_columns"],
        threshold=0.0,
        max_rows=max_rows,
        direction_rule=model_config["direction_rule"],
        cusum_volatility_window=model_config["cusum_volatility_window"],
        cusum_threshold_mult=model_config["cusum_threshold_mult"],
    )
    trade_skip_signals = apply_trade_skip_threshold(raw_signals, threshold)
    trade_skip_trades = build_trades(
        trade_skip_signals,
        prepared_df,
        horizon=config.DEFAULT_HORIZON_CANDLES,
        tp_threshold=config.DEFAULT_TP_THRESHOLD,
        sl_threshold=config.DEFAULT_SL_THRESHOLD,
    )

    return {
        "TradeSkip GRU Reversal": trade_skip_trades,
        "Rule CUSUM Reversal": rule_trades,
    }


def enrich_trades_with_news(trades: pd.DataFrame, news: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    result = trades.copy()
    entry_column = find_trade_entry_time_column(result)
    result["entry_time_utc"] = pd.to_datetime(result[entry_column], utc=True)
    news_times = news["time_utc"].to_numpy(dtype="datetime64[ns]")
    entry_times = result["entry_time_utc"].to_numpy(dtype="datetime64[ns]")
    prev_positions = np.searchsorted(news_times, entry_times, side="right") - 1
    next_positions = np.searchsorted(news_times, entry_times, side="left")

    prev_delta = np.full(len(result), np.nan)
    next_delta = np.full(len(result), np.nan)
    prev_impact = np.array(["none"] * len(result), dtype=object)
    next_impact = np.array(["none"] * len(result), dtype=object)
    prev_currency = np.array(["none"] * len(result), dtype=object)
    next_currency = np.array(["none"] * len(result), dtype=object)
    prev_event = np.array([""] * len(result), dtype=object)
    next_event = np.array([""] * len(result), dtype=object)

    valid_prev = prev_positions >= 0
    if valid_prev.any():
        prev_news = news.iloc[prev_positions[valid_prev]].reset_index(drop=True)
        prev_delta[valid_prev] = (
            pd.Series(entry_times[valid_prev]).dt.tz_localize("UTC") - prev_news["time_utc"]
        ).dt.total_seconds() / 60.0
        prev_impact[valid_prev] = prev_news["impact"].to_numpy()
        prev_currency[valid_prev] = prev_news["currency"].to_numpy()
        prev_event[valid_prev] = prev_news["event_name"].to_numpy()

    valid_next = next_positions < len(news)
    if valid_next.any():
        next_news = news.iloc[next_positions[valid_next]].reset_index(drop=True)
        next_delta[valid_next] = (
            next_news["time_utc"] - pd.Series(entry_times[valid_next]).dt.tz_localize("UTC")
        ).dt.total_seconds() / 60.0
        next_impact[valid_next] = next_news["impact"].to_numpy()
        next_currency[valid_next] = next_news["currency"].to_numpy()
        next_event[valid_next] = next_news["event_name"].to_numpy()

    result["minutes_since_news"] = prev_delta
    result["minutes_to_news"] = next_delta
    result["prev_news_impact"] = prev_impact
    result["next_news_impact"] = next_impact
    result["prev_news_currency"] = prev_currency
    result["next_news_currency"] = next_currency
    result["prev_news_event"] = prev_event
    result["next_news_event"] = next_event

    for window in windows:
        result[f"after_news_{window}m"] = result["minutes_since_news"].between(0, window, inclusive="both")
        result[f"before_news_{window}m"] = result["minutes_to_news"].between(0, window, inclusive="both")
        result[f"around_news_{window}m"] = result[f"after_news_{window}m"] | result[f"before_news_{window}m"]

    result["pnl_pips"] = result["PnL"].astype(float) * 10000.0
    return result


def find_trade_entry_time_column(trades: pd.DataFrame) -> str:
    for column in trades.columns:
        if column == "Дата":
            return column
    if "Exit time" in trades.columns:
        excluded = {"Exit time"}
    else:
        excluded = set()
    best_column = None
    best_score = -1.0
    for column in trades.columns:
        if column in excluded:
            continue
        parsed = pd.to_datetime(trades[column], errors="coerce", utc=True)
        score = float(parsed.notna().mean())
        if score > best_score:
            best_column = column
            best_score = score
    if best_column is None or best_score < 0.8:
        raise ValueError("Could not identify trade entry timestamp column.")
    return best_column


def metric_row(strategy: str, segment: str, trades: pd.DataFrame) -> dict:
    pnl = trades["pnl_pips"].astype(float) if not trades.empty else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0
    return {
        "strategy": strategy,
        "segment": segment,
        "trades": int(len(trades)),
        "total_pnl_pips": float(pnl.sum()) if len(pnl) else 0.0,
        "avg_pnl_pips": float(pnl.mean()) if len(pnl) else np.nan,
        "median_pnl_pips": float(pnl.median()) if len(pnl) else np.nan,
        "winrate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "profit_factor": float(profit_factor),
        "gross_profit_pips": float(gross_profit),
        "gross_loss_pips": float(gross_loss),
    }


def summarize_segments(strategy: str, trades: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = [metric_row(strategy, "all_trades", trades)]
    for window in windows:
        around = trades[f"around_news_{window}m"] if not trades.empty else pd.Series(dtype=bool)
        before = trades[f"before_news_{window}m"] if not trades.empty else pd.Series(dtype=bool)
        after = trades[f"after_news_{window}m"] if not trades.empty else pd.Series(dtype=bool)
        rows.append(metric_row(strategy, f"no_news_pm_{window}m", trades[~around]))
        rows.append(metric_row(strategy, f"around_news_pm_{window}m", trades[around]))
        rows.append(metric_row(strategy, f"before_news_{window}m", trades[before]))
        rows.append(metric_row(strategy, f"after_news_{window}m", trades[after]))
    return pd.DataFrame(rows)


def summarize_by_currency(strategy: str, trades: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []
    for window in windows:
        for direction, flag_prefix, currency_col in [
            ("before", f"before_news_{window}m", "next_news_currency"),
            ("after", f"after_news_{window}m", "prev_news_currency"),
        ]:
            for currency in ["USD", "EUR"]:
                mask = trades[flag_prefix] & trades[currency_col].eq(currency) if not trades.empty else pd.Series(dtype=bool)
                rows.append(metric_row(strategy, f"{direction}_{currency}_{window}m", trades[mask]))
    return pd.DataFrame(rows)


def run_diagnostics(threshold: float, windows: list[int], min_impact: str) -> dict[str, pd.DataFrame]:
    price_df, _ = load_all_price_data(config.DATA_DIR)
    price_df = price_df.copy()
    price_df.index = to_utc_index(price_df.index)
    prepared_df = prepare_rule_frame(price_df)
    news = load_news(NEWS_PATH, min_impact=min_impact)

    strategies = build_strategy_trades(price_df, prepared_df, threshold=threshold)
    enriched_frames = []
    summary_frames = []
    currency_frames = []
    for strategy, trades in strategies.items():
        enriched = enrich_trades_with_news(trades, news, windows)
        enriched["strategy"] = strategy
        enriched_frames.append(enriched)
        summary_frames.append(summarize_segments(strategy, enriched, windows))
        currency_frames.append(summarize_by_currency(strategy, enriched, windows))

    return {
        "summary": pd.concat(summary_frames, ignore_index=True),
        "by_currency": pd.concat(currency_frames, ignore_index=True),
        "trades": pd.concat(enriched_frames, ignore_index=True),
        "news_info": pd.DataFrame(
            [
                {
                    "news_rows": len(news),
                    "news_start": news["time_utc"].min(),
                    "news_end": news["time_utc"].max(),
                    "min_impact": min_impact,
                    "threshold": threshold,
                    "windows": ",".join(map(str, windows)),
                }
            ]
        ),
    }


def save_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def print_report(outputs: dict[str, pd.DataFrame]) -> None:
    summary = outputs["summary"]
    print("News diagnostics")
    print(outputs["news_info"].to_string(index=False))
    for strategy in summary["strategy"].unique():
        print(f"\n{strategy}")
        cols = ["segment", "trades", "total_pnl_pips", "avg_pnl_pips", "winrate", "profit_factor"]
        print(summary[summary["strategy"].eq(strategy)][cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    outputs = run_diagnostics(threshold=args.threshold, windows=args.windows, min_impact=args.min_impact)
    save_outputs(outputs, args.output_dir)
    print_report(outputs)
    print(f"\nSaved diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
