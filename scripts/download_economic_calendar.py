from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "economic_calendar"
RAW_DIR = OUTPUT_DIR / "raw"
PROCESSED_PATH = OUTPUT_DIR / "economic_calendar.csv"

API_BASE_URL = "https://api.tradingeconomics.com/calendar/country"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1/calendar/economic"
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_COUNTRIES = (
    "US",
    "EU",
    "DE",
    "FR",
    "IT",
    "ES",
)
COUNTRY_CURRENCY = {
    "united states": "USD",
    "euro area": "EUR",
    "germany": "EUR",
    "france": "EUR",
    "italy": "EUR",
    "spain": "EUR",
    "us": "USD",
    "eu": "EUR",
    "de": "EUR",
    "fr": "EUR",
    "it": "EUR",
    "es": "EUR",
}
IMPACT_LABELS = {
    1: "low",
    2: "medium",
    3: "high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize an economic calendar for EURUSD diagnostics."
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date in YYYY-MM-DD format.")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--source",
        choices=["finnhub", "tradingeconomics"],
        default="finnhub",
        help="Calendar API source. Finnhub is the default because it has a free key tier.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=list(DEFAULT_COUNTRIES),
        help=(
            "Country filters. For Finnhub use country codes like US EU DE FR IT ES. "
            "For Trading Economics use country names."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key. If omitted, reads FINNHUB_API_KEY for Finnhub, or "
            "TRADING_ECONOMICS_API_KEY/TRADINGECONOMICS_API_KEY/TE_API_KEY for Trading Economics."
        ),
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=31,
        help="Download window size in days. Smaller chunks are safer for API limits.",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Pause between API calls in seconds.")
    parser.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory for raw JSON responses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_PATH,
        help="Normalized CSV output path.",
    )
    return parser.parse_args()


def get_api_key(source: str, explicit_key: str | None = None) -> str:
    load_dotenv(BASE_DIR / ".env")
    if source == "finnhub":
        api_key = explicit_key or os.getenv("FINNHUB_API_KEY")
    else:
        api_key = (
            explicit_key
            or os.getenv("TRADING_ECONOMICS_API_KEY")
            or os.getenv("TRADINGECONOMICS_API_KEY")
            or os.getenv("TE_API_KEY")
        )
    if not api_key:
        raise RuntimeError(
            f"{source} API key is required. Add the key to .env or pass --api-key."
        )
    return api_key


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("End date must be greater than or equal to start date.")
    if chunk_days < 1:
        raise ValueError("--chunk-days must be positive.")

    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def build_url(countries: list[str], start: date, end: date) -> str:
    encoded_countries = ",".join(quote(country.strip()) for country in countries if country.strip())
    return f"{API_BASE_URL}/{encoded_countries}/{start.isoformat()}/{end.isoformat()}"


def fetch_calendar_chunk(
    countries: list[str],
    start: date,
    end: date,
    api_key: str,
    timeout: float,
) -> list[dict]:
    url = build_url(countries, start, end)
    response = requests.get(url, params={"c": api_key, "f": "json"}, timeout=timeout)
    if response.status_code == 410:
        raise RuntimeError(
            "Trading Economics rejected the request with HTTP 410. Check that your API key "
            "has Calendar API access; guest/demo credentials do not cover this endpoint reliably."
        )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("Message"):
        raise RuntimeError(f"Trading Economics API error: {payload['Message']}")
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected API response type: {type(payload).__name__}")
    return payload


def fetch_finnhub_chunk(
    start: date,
    end: date,
    api_key: str,
    timeout: float,
) -> list[dict]:
    response = requests.get(
        FINNHUB_BASE_URL,
        params={"from": start.isoformat(), "to": end.isoformat(), "token": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Finnhub API error: {payload['error']}")
    events = payload.get("economicCalendar")
    if not isinstance(events, list):
        raise RuntimeError("Unexpected Finnhub response: missing economicCalendar list.")
    return events


def impact_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    try:
        return IMPACT_LABELS.get(int(value), "unknown")
    except (TypeError, ValueError):
        normalized = str(value).strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return "unknown"


def country_to_currency(country: object) -> str:
    normalized = str(country or "").strip().lower()
    return COUNTRY_CURRENCY.get(normalized, "")


def column_or_empty(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([""] * len(frame), index=frame.index)


def normalize_events(events: list[dict]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=[
                "time_utc",
                "country",
                "currency",
                "event_name",
                "category",
                "impact",
                "impact_score",
                "actual",
                "forecast",
                "previous",
                "te_forecast",
                "reference",
                "reference_date",
                "source",
                "source_url",
                "url",
                "calendar_id",
                "last_update_utc",
                "revised",
                "unit",
                "ticker",
                "source_name",
            ]
        )

    raw = pd.DataFrame(events)
    if "Date" not in raw.columns:
        raise RuntimeError("Trading Economics response does not contain the Date field.")

    date_series = pd.to_datetime(raw["Date"], errors="coerce", utc=True, format="mixed")

    last_update = pd.to_datetime(column_or_empty(raw, "LastUpdate"), errors="coerce", utc=True, format="mixed")

    country = column_or_empty(raw, "Country")
    importance = column_or_empty(raw, "Importance")
    result = pd.DataFrame(
        {
            "time_utc": date_series.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "country": country,
            "currency": country.map(country_to_currency),
            "event_name": column_or_empty(raw, "Event"),
            "category": column_or_empty(raw, "Category"),
            "impact": importance.map(impact_label),
            "impact_score": pd.to_numeric(importance, errors="coerce"),
            "actual": column_or_empty(raw, "Actual"),
            "forecast": column_or_empty(raw, "Forecast"),
            "previous": column_or_empty(raw, "Previous"),
            "te_forecast": column_or_empty(raw, "TEForecast"),
            "reference": column_or_empty(raw, "Reference"),
            "reference_date": column_or_empty(raw, "ReferenceDate"),
            "source": "tradingeconomics",
            "source_url": column_or_empty(raw, "SourceURL"),
            "url": column_or_empty(raw, "URL"),
            "calendar_id": column_or_empty(raw, "CalendarId"),
            "last_update_utc": last_update.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "revised": column_or_empty(raw, "Revised"),
            "unit": column_or_empty(raw, "Unit"),
            "ticker": column_or_empty(raw, "Ticker"),
            "source_name": column_or_empty(raw, "Source"),
        }
    )
    result = result.dropna(subset=["time_utc"])
    result = result[result["currency"].isin(["EUR", "USD"])]
    result = result.drop_duplicates(subset=["time_utc", "country", "event_name", "category"])
    return result.sort_values(["time_utc", "currency", "impact_score", "event_name"]).reset_index(drop=True)


def normalize_finnhub_events(events: list[dict], countries: set[str]) -> pd.DataFrame:
    columns = [
        "time_utc",
        "country",
        "currency",
        "event_name",
        "category",
        "impact",
        "impact_score",
        "actual",
        "forecast",
        "previous",
        "unit",
        "source",
    ]
    if not events:
        return pd.DataFrame(columns=columns)

    raw = pd.DataFrame(events)
    if "time" not in raw.columns:
        raise RuntimeError("Finnhub response does not contain the time field.")

    country = column_or_empty(raw, "country").astype(str).str.upper()
    date_series = pd.to_datetime(raw["time"], errors="coerce", utc=True, format="mixed")
    impact = column_or_empty(raw, "impact").astype(str).str.lower()
    impact_score_map = {"low": 1, "medium": 2, "high": 3}

    result = pd.DataFrame(
        {
            "time_utc": date_series.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "country": country,
            "currency": country.map(country_to_currency),
            "event_name": column_or_empty(raw, "event"),
            "category": "",
            "impact": impact.where(impact.isin(["low", "medium", "high"]), "unknown"),
            "impact_score": impact.map(impact_score_map),
            "actual": column_or_empty(raw, "actual"),
            "forecast": column_or_empty(raw, "estimate"),
            "previous": column_or_empty(raw, "prev"),
            "unit": column_or_empty(raw, "unit"),
            "source": "finnhub",
        }
    )
    result = result.dropna(subset=["time_utc"])
    result = result[result["country"].isin(countries)]
    result = result[result["currency"].isin(["EUR", "USD"])]
    result = result.drop_duplicates(subset=["time_utc", "country", "event_name"])
    return result.sort_values(["time_utc", "currency", "impact_score", "event_name"]).reset_index(drop=True)


def save_raw(events: list[dict], raw_dir: Path, start: date, end: date) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"tradingeconomics_{start.isoformat()}_{end.isoformat()}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(events, file, ensure_ascii=False, indent=2)


def download_tradingeconomics_calendar(
    start: date,
    end: date,
    countries: list[str],
    api_key: str,
    chunk_days: int,
    sleep_seconds: float,
    timeout: float,
    raw_dir: Path,
) -> pd.DataFrame:
    all_events = []
    chunks = iter_date_chunks(start, end, chunk_days)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"[{index}/{len(chunks)}] Downloading {chunk_start} -> {chunk_end}...")
        events = fetch_calendar_chunk(countries, chunk_start, chunk_end, api_key, timeout)
        save_raw(events, raw_dir, chunk_start, chunk_end)
        all_events.extend(events)
        if sleep_seconds > 0 and index < len(chunks):
            time.sleep(sleep_seconds)
    return normalize_events(all_events)


def download_finnhub_calendar(
    start: date,
    end: date,
    countries: list[str],
    api_key: str,
    chunk_days: int,
    sleep_seconds: float,
    timeout: float,
    raw_dir: Path,
) -> pd.DataFrame:
    all_events = []
    chunks = iter_date_chunks(start, end, chunk_days)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"[{index}/{len(chunks)}] Downloading {chunk_start} -> {chunk_end}...")
        events = fetch_finnhub_chunk(chunk_start, chunk_end, api_key, timeout)
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"finnhub_{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(events, file, ensure_ascii=False, indent=2)
        all_events.extend(events)
        if sleep_seconds > 0 and index < len(chunks):
            time.sleep(sleep_seconds)
    return normalize_finnhub_events(all_events, {country.upper() for country in countries})


def main() -> None:
    args = parse_args()
    api_key = get_api_key(args.source, args.api_key)
    start = parse_date(args.start)
    end = parse_date(args.end)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "finnhub":
        normalized = download_finnhub_calendar(
            start=start,
            end=end,
            countries=args.countries,
            api_key=api_key,
            chunk_days=args.chunk_days,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
            raw_dir=args.raw_dir,
        )
    else:
        normalized = download_tradingeconomics_calendar(
            start=start,
            end=end,
            countries=args.countries,
            api_key=api_key,
            chunk_days=args.chunk_days,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
            raw_dir=args.raw_dir,
        )
    normalized.to_csv(output_path, index=False, encoding="utf-8")

    impact_counts = normalized["impact"].value_counts(dropna=False).to_dict()
    currency_counts = normalized["currency"].value_counts(dropna=False).to_dict()
    print(f"Saved {len(normalized)} normalized events to {output_path}")
    print(f"Impact counts: {impact_counts}")
    print(f"Currency counts: {currency_counts}")
    print(f"Downloaded at UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
