from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.news_impact_diagnostics import (  # noqa: E402
    DEFAULT_THRESHOLD,
    OUTPUT_DIR as DIAGNOSTICS_DIR,
    build_strategy_trades,
    enrich_trades_with_news,
    load_news,
    metric_row,
    to_utc_index,
)
from src import config  # noqa: E402
from src.connector.data_fetcher import load_all_price_data  # noqa: E402
from src.strategy.signal_generator import prepare_rule_frame  # noqa: E402


OUTPUT_DIR = BASE_DIR / "data" / "processed" / "news_filter_experiments"
NEWS_PATH = BASE_DIR / "data" / "economic_calendar" / "economic_calendar.csv"
DEFAULT_WINDOWS = (15, 30, 60, 120)


FILTERS = [
    {
        "filter": "base_no_filter",
        "description": "No news filter",
        "rules": [],
    },
    {
        "filter": "skip_before_medium_high_15m",
        "description": "Skip entries 15m before EUR/USD medium/high news",
        "rules": [{"side": "before", "impact_min": "medium", "currency": "ANY", "minutes": 15}],
    },
    {
        "filter": "skip_before_medium_high_30m",
        "description": "Skip entries 30m before EUR/USD medium/high news",
        "rules": [{"side": "before", "impact_min": "medium", "currency": "ANY", "minutes": 30}],
    },
    {
        "filter": "skip_before_medium_high_60m",
        "description": "Skip entries 60m before EUR/USD medium/high news",
        "rules": [{"side": "before", "impact_min": "medium", "currency": "ANY", "minutes": 60}],
    },
    {
        "filter": "skip_before_high_30m",
        "description": "Skip entries 30m before EUR/USD high news only",
        "rules": [{"side": "before", "impact_min": "high", "currency": "ANY", "minutes": 30}],
    },
    {
        "filter": "skip_before_high_60m",
        "description": "Skip entries 60m before EUR/USD high news only",
        "rules": [{"side": "before", "impact_min": "high", "currency": "ANY", "minutes": 60}],
    },
    {
        "filter": "skip_eur60_usd30_medium_high",
        "description": "Skip 60m before EUR medium/high and 30m before USD medium/high",
        "rules": [
            {"side": "before", "impact_min": "medium", "currency": "EUR", "minutes": 60},
            {"side": "before", "impact_min": "medium", "currency": "USD", "minutes": 30},
        ],
    },
    {
        "filter": "skip_eur120_usd30_medium_high",
        "description": "Skip 120m before EUR medium/high and 30m before USD medium/high",
        "rules": [
            {"side": "before", "impact_min": "medium", "currency": "EUR", "minutes": 120},
            {"side": "before", "impact_min": "medium", "currency": "USD", "minutes": 30},
        ],
    },
    {
        "filter": "skip_eur60_only_medium_high",
        "description": "Skip 60m before EUR medium/high news only",
        "rules": [{"side": "before", "impact_min": "medium", "currency": "EUR", "minutes": 60}],
    },
    {
        "filter": "skip_eur120_only_medium_high",
        "description": "Skip 120m before EUR medium/high news only",
        "rules": [{"side": "before", "impact_min": "medium", "currency": "EUR", "minutes": 120}],
    },
]

IMPACT_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate economic-news entry filters.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--news-path", type=Path, default=NEWS_PATH)
    return parser.parse_args()


def rule_mask(trades: pd.DataFrame, rule: dict) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=bool)

    side = rule["side"]
    minutes = float(rule["minutes"])
    currency = rule["currency"]
    impact_min = IMPACT_RANK[rule["impact_min"]]
    if side == "before":
        delta = trades["minutes_to_news"]
        impact = trades["next_news_impact"].map(IMPACT_RANK).fillna(0)
        news_currency = trades["next_news_currency"]
    elif side == "after":
        delta = trades["minutes_since_news"]
        impact = trades["prev_news_impact"].map(IMPACT_RANK).fillna(0)
        news_currency = trades["prev_news_currency"]
    else:
        raise ValueError(f"Unsupported rule side: {side}")

    mask = delta.between(0, minutes, inclusive="both") & impact.ge(impact_min)
    if currency != "ANY":
        mask &= news_currency.eq(currency)
    return mask


def apply_filter(trades: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    if trades.empty or not rules:
        return trades.copy()
    skip = pd.Series(False, index=trades.index)
    for rule in rules:
        skip |= rule_mask(trades, rule)
    filtered = trades.loc[~skip].copy()
    filtered["filtered_out"] = False
    return filtered


def max_drawdown_pips(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    equity = trades["pnl_pips"].astype(float).cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def extended_metric_row(strategy: str, filter_name: str, description: str, trades: pd.DataFrame, base_trades: int) -> dict:
    row = metric_row(strategy, filter_name, trades)
    row["description"] = description
    row["removed_trades"] = int(base_trades - len(trades))
    row["removed_pct"] = float((base_trades - len(trades)) / base_trades) if base_trades else 0.0
    row["max_drawdown_pips"] = max_drawdown_pips(trades)
    return row


def summarize_by_year(strategy: str, filter_name: str, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["year"] = pd.to_datetime(frame["entry_time_utc"], utc=True).dt.year
    rows = []
    for year, group in frame.groupby("year"):
        row = metric_row(strategy, str(year), group)
        row["filter"] = filter_name
        row["year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiments(threshold: float, news_path: Path) -> dict[str, pd.DataFrame]:
    price_df, _ = load_all_price_data(config.DATA_DIR)
    price_df = price_df.copy()
    price_df.index = to_utc_index(price_df.index)
    prepared_df = prepare_rule_frame(price_df)
    strategies = build_strategy_trades(price_df, prepared_df, threshold=threshold)
    news = load_news(news_path, min_impact="medium")

    summary_rows = []
    year_frames = []
    filtered_trade_frames = []

    for strategy, trades in strategies.items():
        enriched = enrich_trades_with_news(trades, news, list(DEFAULT_WINDOWS))
        base_count = len(enriched)
        for candidate in FILTERS:
            filtered = apply_filter(enriched, candidate["rules"])
            filtered["strategy"] = strategy
            filtered["filter"] = candidate["filter"]
            filtered_trade_frames.append(filtered)
            summary_rows.append(
                extended_metric_row(
                    strategy=strategy,
                    filter_name=candidate["filter"],
                    description=candidate["description"],
                    trades=filtered,
                    base_trades=base_count,
                )
            )
            by_year = summarize_by_year(strategy, candidate["filter"], filtered)
            if not by_year.empty:
                year_frames.append(by_year)

    summary = pd.DataFrame(summary_rows)
    year = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()
    trades_out = pd.concat(filtered_trade_frames, ignore_index=True) if filtered_trade_frames else pd.DataFrame()
    if not year.empty:
        yearly_quality = (
            year.groupby(["strategy", "filter"])
            .agg(
                positive_years=("total_pnl_pips", lambda value: int((value > 0).sum())),
                years=("year", "nunique"),
                worst_year_pips=("total_pnl_pips", "min"),
                median_year_pips=("total_pnl_pips", "median"),
            )
            .reset_index()
        )
        summary = summary.merge(yearly_quality, left_on=["strategy", "segment"], right_on=["strategy", "filter"], how="left")
        summary = summary.drop(columns=["filter"])
    return {
        "summary": summary,
        "by_year": year,
        "filtered_trades": trades_out,
    }


def save_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def print_report(summary: pd.DataFrame) -> None:
    cols = [
        "strategy",
        "segment",
        "trades",
        "removed_trades",
        "total_pnl_pips",
        "avg_pnl_pips",
        "winrate",
        "profit_factor",
        "max_drawdown_pips",
        "positive_years",
        "worst_year_pips",
    ]
    for strategy in summary["strategy"].unique():
        print(f"\n{strategy}")
        table = summary[summary["strategy"].eq(strategy)].copy()
        table = table.sort_values(["profit_factor", "total_pnl_pips"], ascending=False)
        print(table[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    outputs = run_experiments(threshold=args.threshold, news_path=args.news_path)
    save_outputs(outputs, args.output_dir)
    print_report(outputs["summary"])
    print(f"\nSaved filter experiments to: {args.output_dir}")


if __name__ == "__main__":
    main()
