from pathlib import Path

import pandas as pd
import pytest

from src.connector import data_fetcher
from src.connector.data_fetcher import load_all_price_data, load_price_data


def test_load_price_data_normalizes_columns_and_sorts(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "Date": ["2024-01-01 00:15:00", "2024-01-01 00:00:00", "2024-01-01 00:00:00"],
            "Open": [1.2, 1.1, 9.9],
            "High": [1.3, 1.2, 9.9],
            "Low": [1.1, 1.0, 9.9],
            "Close": [1.25, 1.15, 9.9],
            "tick_volume": [20, 10, 99],
        }
    ).to_csv(csv_path, index=False)

    result = load_price_data(csv_path)

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.is_monotonic_increasing
    assert len(result) == 2
    assert result.iloc[0]["open"] == 1.1
    assert result.iloc[0]["volume"] == 10


def test_load_price_data_adds_zero_volume_when_missing(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"],
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
        }
    ).to_csv(csv_path, index=False)

    result = load_price_data(csv_path)

    assert result.iloc[0]["volume"] == 0.0


def test_load_all_price_data_skips_bad_csv_and_deduplicates(tmp_path: Path):
    good_a = tmp_path / "a.csv"
    good_b = tmp_path / "b.csv"
    bad = tmp_path / "bad.csv"

    pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"],
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
            "volume": [10],
        }
    ).to_csv(good_a, index=False)
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:15:00"],
            "open": [2.1, 2.2],
            "high": [2.2, 2.3],
            "low": [2.0, 2.1],
            "close": [2.15, 2.25],
            "volume": [20, 30],
        }
    ).to_csv(good_b, index=False)
    pd.DataFrame({"x": ["not a price file"]}).to_csv(bad, index=False)

    result, loaded = load_all_price_data(tmp_path)

    assert loaded == ["a.csv", "b.csv"]
    assert len(result) == 2
    assert result.loc[pd.Timestamp("2024-01-01 00:00:00"), "open"] == 2.1


def test_load_all_price_data_raises_when_no_csv_can_be_loaded(tmp_path: Path):
    (tmp_path / "bad.csv").write_text("x\nnot a date\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV"):
        load_all_price_data(tmp_path)


def test_list_csv_files_uses_raw_dir_when_called_with_data_root(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data"
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir()
    raw_csv = raw_dir / "prices.csv"
    processed_csv = processed_dir / "diagnostics.csv"
    raw_csv.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
    processed_csv.write_text("not,price,data\n", encoding="utf-8")
    monkeypatch.setattr(data_fetcher, "DATA_DIR", data_root)

    assert data_fetcher.list_csv_files(data_root) == [raw_csv]
