from pathlib import Path

import pandas as pd

from src.features.news_features import NEWS_FEATURE_COLUMNS, add_news_features, load_economic_calendar


def test_load_economic_calendar_ignores_outcome_columns(tmp_path: Path):
    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame(
        [
            {
                "time_utc": "2024-01-01 10:00:00+0000",
                "country": "US",
                "currency": "USD",
                "event_name": "CPI",
                "category": "",
                "impact": "high",
                "impact_score": 3,
                "actual": 9.9,
                "forecast": 1.0,
                "previous": 0.5,
                "unit": "%",
                "source": "test",
            }
        ]
    ).to_csv(calendar_path, index=False)

    loaded = load_economic_calendar(calendar_path)

    assert {"actual", "forecast", "previous"}.isdisjoint(loaded.columns)
    assert loaded.iloc[0]["impact_rank"] == 3


def test_add_news_features_uses_schedule_without_outcome_leakage():
    price = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=pd.to_datetime(
            [
                "2024-01-01 09:45:00+00:00",
                "2024-01-01 10:00:00+00:00",
                "2024-01-01 10:15:00+00:00",
            ]
        ),
    )
    calendar = pd.DataFrame(
        [
            {
                "time_utc": pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
                "currency": "USD",
                "event_name": "US CPI",
                "impact": "high",
                "impact_rank": 3,
            }
        ]
    )

    featured = add_news_features(price, calendar=calendar)

    assert set(NEWS_FEATURE_COLUMNS).issubset(featured.columns)
    assert featured.loc[pd.Timestamp("2024-01-01 09:45:00+00:00"), "news_minutes_to_next"] == 15
    assert featured.loc[pd.Timestamp("2024-01-01 09:45:00+00:00"), "news_minutes_since_last"] == 1440
    assert featured.loc[pd.Timestamp("2024-01-01 10:15:00+00:00"), "news_minutes_since_last"] == 15
    assert featured.loc[pd.Timestamp("2024-01-01 10:15:00+00:00"), "news_last_is_cpi"] == 1
    assert "actual" not in featured.columns
    assert "forecast" not in featured.columns
    assert "previous" not in featured.columns
