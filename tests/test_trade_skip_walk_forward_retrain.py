import numpy as np
import pandas as pd

from scripts.trade_skip_walk_forward_retrain import (
    _build_trade_skip_frame_for_direction,
    _localized_year_start,
    _period_positions,
)


def test_localized_year_start_preserves_index_timezone():
    index = pd.to_datetime(["2024-01-01 00:00:00+00:00"])

    result = _localized_year_start(2024, index)

    assert result.tzinfo is not None
    assert str(result) == "2024-01-01 00:00:00+00:00"


def test_period_positions_filters_labels_sequence_and_horizon():
    index = pd.date_range("2024-01-01", periods=110, freq="15min", tz="UTC")
    frame = pd.DataFrame({"trade_success": [np.nan] * 110}, index=index)
    frame.loc[index[95:109], "trade_success"] = 1

    positions = _period_positions(frame, index[0], index[-1], horizon=4)

    assert positions.min() >= 95
    assert positions.max() <= 104


def test_build_trade_skip_frame_for_direction_can_reverse_cusum(monkeypatch):
    index = pd.date_range("2020-01-01", periods=2, freq="15min", tz="UTC")
    event_frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.001],
            "low": [1.0, 0.999],
            "close": [1.0, 1.001],
            "event": [1, 0],
            "event_cusum_direction": [1, 0],
            "returns": [0.0, 0.001],
            "volatility_96": [0.001, 0.001],
            "ma_50_diff": [0.0, 0.0],
            "breakout_up": [False, False],
            "breakout_down": [False, False],
            "near_support": [False, False],
            "near_resistance": [False, False],
        },
        index=index,
    )
    monkeypatch.setattr(
        "scripts.trade_skip_walk_forward_retrain._build_event_frame",
        lambda price_df, volatility_window, threshold_mult: event_frame,
    )
    monkeypatch.setattr(
        "scripts.trade_skip_walk_forward_retrain._fit_regime_thresholds",
        lambda frame: {"trend_abs": 1.0, "vol_high": 1.0},
    )

    result = _build_trade_skip_frame_for_direction(
        price_df=event_frame,
        feature_columns=["returns", "event_cusum_direction"],
        direction_rule="cusum_reversal",
        cusum_volatility_window=192,
        cusum_threshold_mult=1.8,
        min_edge_pips=0.0,
    )

    assert result.iloc[0]["original_event_cusum_direction"] == 1
    assert result.iloc[0]["event_cusum_direction"] == -1
    assert result.iloc[0]["rule_decision"] == "SELL"
