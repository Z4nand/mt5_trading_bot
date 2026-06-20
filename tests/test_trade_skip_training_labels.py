import pandas as pd

from src.models.trade_skip_training import trade_success_label_at


def test_trade_success_label_respects_min_edge():
    frame = pd.DataFrame(
        {
            "open": [1.0000, 1.0000],
            "high": [1.0000, 1.0003],
            "low": [1.0000, 0.9999],
            "close": [1.0000, 1.0001],
            "event_cusum_direction": [1, 0],
        }
    )

    assert trade_success_label_at(frame, 0, horizon=1, tp_threshold=None, sl_threshold=None) == 1
    assert trade_success_label_at(frame, 0, horizon=1, tp_threshold=None, sl_threshold=None, min_edge=0.0002) == 0
