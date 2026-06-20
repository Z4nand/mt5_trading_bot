import numpy as np
import pandas as pd

from scripts.trade_skip_regime_diagnostics import (
    _add_regimes,
    _metrics_by,
    _save_outputs,
    _split_name,
    _threshold_scan,
)


def _diagnostic_frame() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2020-01-01 00:00:00+00:00",
            "2020-01-01 00:15:00+00:00",
            "2022-01-01 00:00:00+00:00",
            "2024-01-01 00:00:00+00:00",
            "2025-01-01 00:00:00+00:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "split": _split_name(index),
            "volatility_96": [0.1, 0.2, 0.15, 0.3, 0.05],
            "ma_50_diff": [-0.2, 0.0, 0.3, 0.4, -0.5],
            "prob_trade": [0.40, 0.60, 0.55, 0.70, 0.20],
            "actual_trade": [0, 1, 0, 1, 0],
            "pnl": [-0.0008, 0.0008, -0.0002, 0.0008, -0.0008],
        },
        index=index,
    )
    return _add_regimes(frame)


def test_split_name_handles_utc_index_boundaries():
    index = pd.to_datetime(
        [
            "2020-12-31 23:45:00+00:00",
            "2021-01-01 00:00:00+00:00",
            "2024-01-01 00:00:00+00:00",
        ]
    )

    assert _split_name(index).tolist() == [
        "train_2015_2020",
        "valid_2021_2023",
        "test_2024_2026Q1",
    ]


def test_add_regimes_uses_train_thresholds_and_attaches_metadata():
    result = _diagnostic_frame()

    assert set(result["vol_regime"]) <= {"low_vol", "mid_vol", "high_vol"}
    assert set(result["trend_regime"]) <= {"downtrend", "range", "uptrend"}
    assert result.attrs["regime_thresholds"]["volatility_96_low"] > 0
    assert result.attrs["regime_thresholds"]["abs_ma_50_diff_trend"] >= 0


def test_metrics_by_calculates_selected_trade_metrics():
    frame = _diagnostic_frame()

    result = _metrics_by(frame, ["split"], threshold=0.50)
    train = result[result["split"].eq("train_2015_2020")].iloc[0]

    assert train["events"] == 2
    assert train["selected_trades"] == 1
    assert train["selected_winrate_pct"] == 100.0
    assert np.isclose(train["avg_pnl_selected_pips"], 8.0)


def test_threshold_scan_returns_one_row_per_split_and_threshold():
    frame = _diagnostic_frame()

    result = _threshold_scan(frame, thresholds=(0.50, 0.65))

    assert set(result["threshold"]) == {0.50, 0.65}
    assert set(result["split"]) == {"train_2015_2020", "valid_2021_2023", "test_2024_2026Q1"}
    assert len(result) == 6


def test_save_outputs_writes_dataframe_and_dict_csv(tmp_path):
    outputs = {
        "table": pd.DataFrame({"a": [1], "b": [2]}),
        "info": {"x": "y"},
    }

    _save_outputs(outputs, tmp_path)

    assert (tmp_path / "table.csv").exists()
    assert (tmp_path / "info.csv").exists()
    assert pd.read_csv(tmp_path / "table.csv").iloc[0]["a"] == 1
