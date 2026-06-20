from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
DEFAULT_NEWS_PATH = config.DATA_DIR / "economic_calendar" / "economic_calendar.csv"
DEFAULT_CURRENCIES = ("EUR", "USD")
DEFAULT_MAX_MINUTES = 24 * 60

EVENT_GROUP_PATTERNS = {
    "cpi": r"\b(?:cpi|consumer price|inflation)\b",
    "rate": r"\b(?:rate decision|interest rate|ecb|fomc|fed|monetary policy)\b",
    "jobs": r"\b(?:nonfarm|nfp|payroll|unemployment|jobless|employment)\b",
    "pmi": r"\b(?:pmi|ism)\b",
    "gdp": r"\b(?:gdp|gross domestic)\b",
    "speech": r"\b(?:speech|speaks|testifies|press conference)\b",
}

BASE_NEWS_FEATURE_COLUMNS = [
    "news_minutes_to_next",
    "news_minutes_since_last",
    "news_next_impact_score",
    "news_last_impact_score",
    "news_next_is_high",
    "news_last_is_high",
    "news_next_is_eur",
    "news_next_is_usd",
    "news_last_is_eur",
    "news_last_is_usd",
    "news_before_30m",
    "news_before_60m",
    "news_before_120m",
    "news_after_30m",
    "news_after_60m",
]

EVENT_GROUP_NEWS_FEATURE_COLUMNS = [
    *[f"news_next_is_{group}" for group in EVENT_GROUP_PATTERNS],
    *[f"news_last_is_{group}" for group in EVENT_GROUP_PATTERNS],
]

NEWS_FEATURE_COLUMNS = BASE_NEWS_FEATURE_COLUMNS + EVENT_GROUP_NEWS_FEATURE_COLUMNS


def _impact_rank(value: object) -> int:
    return IMPACT_RANK.get(str(value).strip().lower(), 0)


def _to_utc_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    timestamp_index = pd.to_datetime(index)
    if timestamp_index.tz is None:
        return timestamp_index.tz_localize("UTC")
    return timestamp_index.tz_convert("UTC")


def load_economic_calendar(
    path: Path = DEFAULT_NEWS_PATH,
    currencies: tuple[str, ...] = DEFAULT_CURRENCIES,
    min_impact: str = "medium",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Economic calendar not found: {path}")

    calendar = pd.read_csv(path)
    required = {"time_utc", "currency", "event_name", "impact"}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"Economic calendar is missing columns: {sorted(missing)}")

    result = calendar.copy()
    result["time_utc"] = pd.to_datetime(result["time_utc"], utc=True, errors="coerce")
    result["currency"] = result["currency"].astype(str).str.upper()
    result["impact"] = result["impact"].astype(str).str.lower()
    result["impact_rank"] = result["impact"].map(_impact_rank).fillna(0).astype(int)

    min_rank = _impact_rank(min_impact)
    result = result[
        result["time_utc"].notna()
        & result["currency"].isin(currencies)
        & result["impact_rank"].ge(min_rank)
    ].copy()

    if result.empty:
        return result

    event_text = result["event_name"].fillna("").astype(str).str.lower()
    for group, pattern in EVENT_GROUP_PATTERNS.items():
        result[f"is_{group}"] = event_text.str.contains(pattern, regex=True).astype(int)

    safe_columns = ["time_utc", "currency", "event_name", "impact", "impact_rank", *[f"is_{group}" for group in EVENT_GROUP_PATTERNS]]
    return result[safe_columns].sort_values("time_utc")


def aggregate_calendar_by_time(calendar: pd.DataFrame) -> pd.DataFrame:
    if calendar.empty:
        return pd.DataFrame(columns=["time_utc", "impact_score", "is_high", "is_eur", "is_usd", *[f"is_{group}" for group in EVENT_GROUP_PATTERNS]])

    data = calendar.copy()
    data["impact_score"] = data["impact_rank"] / 3.0
    data["is_high"] = data["impact_rank"].ge(3).astype(int)
    data["is_eur"] = data["currency"].eq("EUR").astype(int)
    data["is_usd"] = data["currency"].eq("USD").astype(int)
    event_text = data["event_name"].fillna("").astype(str).str.lower()
    for group, pattern in EVENT_GROUP_PATTERNS.items():
        if f"is_{group}" not in data.columns:
            data[f"is_{group}"] = event_text.str.contains(pattern, regex=True).astype(int)

    agg_columns = {
        "impact_score": "max",
        "is_high": "max",
        "is_eur": "max",
        "is_usd": "max",
    }
    for group in EVENT_GROUP_PATTERNS:
        agg_columns[f"is_{group}"] = "max"

    return data.groupby("time_utc", as_index=False).agg(agg_columns).sort_values("time_utc")


def add_news_features(
    price_df: pd.DataFrame,
    calendar: pd.DataFrame | None = None,
    calendar_path: Path = DEFAULT_NEWS_PATH,
    currencies: tuple[str, ...] = DEFAULT_CURRENCIES,
    min_impact: str = "medium",
    max_minutes: int = DEFAULT_MAX_MINUTES,
) -> pd.DataFrame:
    """Add scheduled-news features known at the candle timestamp.

    Only event schedule metadata is used: time, currency, impact and event name group.
    Outcome fields such as actual, forecast and previous are intentionally ignored.
    """
    result = price_df.copy()
    if calendar is None:
        calendar = load_economic_calendar(calendar_path, currencies=currencies, min_impact=min_impact)
    events = aggregate_calendar_by_time(calendar)

    if events.empty:
        for column in NEWS_FEATURE_COLUMNS:
            result[column] = 0.0
        result["news_minutes_to_next"] = float(max_minutes)
        result["news_minutes_since_last"] = float(max_minutes)
        return result

    index_utc = _to_utc_datetime_index(result.index)
    candle_ns = index_utc.view("int64")
    event_times = pd.DatetimeIndex(events["time_utc"])
    event_ns = event_times.view("int64")

    next_pos = np.searchsorted(event_ns, candle_ns, side="left")
    last_pos = np.searchsorted(event_ns, candle_ns, side="right") - 1
    has_next = next_pos < len(events)
    has_last = last_pos >= 0

    minutes_to_next = np.full(len(result), float(max_minutes), dtype=np.float32)
    minutes_since_last = np.full(len(result), float(max_minutes), dtype=np.float32)
    minutes_to_next[has_next] = ((event_ns[next_pos[has_next]] - candle_ns[has_next]) / 60_000_000_000).astype(np.float32)
    minutes_since_last[has_last] = ((candle_ns[has_last] - event_ns[last_pos[has_last]]) / 60_000_000_000).astype(np.float32)
    minutes_to_next = np.clip(minutes_to_next, 0, max_minutes)
    minutes_since_last = np.clip(minutes_since_last, 0, max_minutes)

    result["news_minutes_to_next"] = minutes_to_next
    result["news_minutes_since_last"] = minutes_since_last

    source_columns = ["impact_score", "is_high", "is_eur", "is_usd", *[f"is_{group}" for group in EVENT_GROUP_PATTERNS]]
    for column in source_columns:
        values = events[column].to_numpy(dtype=np.float32)
        next_values = np.zeros(len(result), dtype=np.float32)
        last_values = np.zeros(len(result), dtype=np.float32)
        next_values[has_next] = values[next_pos[has_next]]
        last_values[has_last] = values[last_pos[has_last]]
        result[f"news_next_{column}"] = next_values
        result[f"news_last_{column}"] = last_values

    result = result.rename(
        columns={
            "news_next_impact_score": "news_next_impact_score",
            "news_last_impact_score": "news_last_impact_score",
            "news_next_is_high": "news_next_is_high",
            "news_last_is_high": "news_last_is_high",
            "news_next_is_eur": "news_next_is_eur",
            "news_next_is_usd": "news_next_is_usd",
            "news_last_is_eur": "news_last_is_eur",
            "news_last_is_usd": "news_last_is_usd",
        }
    )
    result["news_before_30m"] = (result["news_minutes_to_next"] <= 30).astype(int)
    result["news_before_60m"] = (result["news_minutes_to_next"] <= 60).astype(int)
    result["news_before_120m"] = (result["news_minutes_to_next"] <= 120).astype(int)
    result["news_after_30m"] = (result["news_minutes_since_last"] <= 30).astype(int)
    result["news_after_60m"] = (result["news_minutes_since_last"] <= 60).astype(int)

    for column in NEWS_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0

    return result
