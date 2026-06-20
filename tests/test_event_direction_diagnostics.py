import pandas as pd

from scripts.event_direction_diagnostics import _direction_for_row
from src.strategy.signal_generator import first_hit_label_at, rule_direction_from_row


def test_cusum_momentum_and_reversal_rules():
    row = pd.Series(
        {
            "event_cusum_direction": 1,
            "breakout_up": False,
            "breakout_down": False,
            "near_support": False,
            "near_resistance": False,
            "ma_50_diff": 0.0,
            "volatility_96": 0.0,
        }
    )

    assert _direction_for_row(row, "cusum_momentum", {"trend_abs": 1, "vol_high": 1}) == "BUY"
    assert _direction_for_row(row, "cusum_reversal", {"trend_abs": 1, "vol_high": 1}) == "SELL"


def test_level_reversal_requires_unambiguous_level():
    row = pd.Series(
        {
            "event_cusum_direction": 0,
            "breakout_up": False,
            "breakout_down": False,
            "near_support": True,
            "near_resistance": False,
            "ma_50_diff": 0.0,
            "volatility_96": 0.0,
        }
    )

    assert _direction_for_row(row, "level_reversal", {"trend_abs": 1, "vol_high": 1}) == "BUY"
    row["near_resistance"] = True
    assert _direction_for_row(row, "level_reversal", {"trend_abs": 1, "vol_high": 1}) is None


def test_active_rule_direction_defaults_to_cusum_reversal():
    row = pd.Series({"event_cusum_direction": 1})

    assert rule_direction_from_row(row) == "SELL"


def test_first_hit_label_uses_fixed_price_delta():
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000],
            "high": [1.1000, 1.1005],
            "low": [1.1000, 1.0998],
            "close": [1.1000, 1.1002],
        }
    )

    assert first_hit_label_at(frame, entry_idx=0, threshold=0.0005, horizon=1) == "UP"
