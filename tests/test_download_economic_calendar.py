from datetime import date

import pandas as pd
import pytest

from scripts.download_economic_calendar import (
    build_url,
    impact_label,
    iter_date_chunks,
    normalize_finnhub_events,
    normalize_events,
)


def test_impact_label_normalizes_trading_economics_scores():
    assert impact_label(1) == "low"
    assert impact_label("2") == "medium"
    assert impact_label(3.0) == "high"
    assert impact_label("High") == "high"
    assert impact_label(None) == "unknown"


def test_iter_date_chunks_covers_range_without_overlap():
    chunks = iter_date_chunks(date(2024, 1, 1), date(2024, 1, 5), chunk_days=2)

    assert chunks == [
        (date(2024, 1, 1), date(2024, 1, 2)),
        (date(2024, 1, 3), date(2024, 1, 4)),
        (date(2024, 1, 5), date(2024, 1, 5)),
    ]


def test_iter_date_chunks_rejects_inverted_range():
    with pytest.raises(ValueError):
        iter_date_chunks(date(2024, 1, 5), date(2024, 1, 1), chunk_days=2)


def test_build_url_encodes_country_names_and_dates():
    url = build_url(["united states", "euro area"], date(2024, 1, 1), date(2024, 1, 31))

    assert url.endswith("/united%20states,euro%20area/2024-01-01/2024-01-31")


def test_normalize_events_keeps_eurusd_and_impact_fields():
    rows = [
        {
            "CalendarId": "1",
            "Date": "2024-01-05T13:30:00",
            "Country": "United States",
            "Event": "Non Farm Payrolls",
            "Category": "Labour",
            "Importance": 3,
            "Actual": "216K",
            "Forecast": "170K",
            "Previous": "173K",
            "LastUpdate": "2024-01-05T13:31:00",
        },
        {
            "CalendarId": "2",
            "Date": "2024-01-05T10:00:00+01:00",
            "Country": "Germany",
            "Event": "CPI",
            "Category": "Prices",
            "Importance": 2,
        },
        {
            "CalendarId": "3",
            "Date": "2024-01-05T01:00:00",
            "Country": "Japan",
            "Event": "Unrelated",
            "Category": "Other",
            "Importance": 1,
        },
    ]

    result = normalize_events(rows)

    assert len(result) == 2
    assert set(result["currency"]) == {"EUR", "USD"}
    assert set(result["impact"]) == {"high", "medium"}
    assert "impact_score" in result.columns
    assert pd.to_datetime(result["time_utc"], utc=True).notna().all()


def test_normalize_finnhub_events_maps_fields_and_filters_country_codes():
    rows = [
        {
            "actual": 216,
            "country": "US",
            "estimate": 170,
            "event": "Non Farm Payrolls",
            "impact": "high",
            "prev": 173,
            "time": "2024-01-05 13:30:00",
            "unit": "K",
        },
        {
            "actual": 44.4,
            "country": "EU",
            "estimate": 44.2,
            "event": "HCOB Manufacturing PMI Final",
            "impact": "medium",
            "prev": 44.2,
            "time": "2024-01-02 09:00:00",
            "unit": "",
        },
        {
            "actual": 50,
            "country": "JP",
            "estimate": 49,
            "event": "Filtered Out",
            "impact": "low",
            "prev": 48,
            "time": "2024-01-02 09:00:00",
            "unit": "",
        },
    ]

    result = normalize_finnhub_events(rows, {"US", "EU"})

    assert len(result) == 2
    assert set(result["currency"]) == {"EUR", "USD"}
    assert set(result["impact"]) == {"high", "medium"}
    assert set(result["forecast"]) == {170, 44.2}
