import numpy as np
import pandas as pd

from src.ui.streamlit_app import trade_skip_model_metrics


def test_trade_skip_metrics_use_current_tp_sl_and_cost():
    index = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    prepared = pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1000],
            "high": [1.1000, 1.1004, 1.1000],
            "low": [1.1000, 1.0990, 1.1000],
            "close": [1.1000, 1.0995, 1.1000],
        },
        index=index,
    )
    signals = pd.DataFrame(
        [
            {
                "time": index[0],
                "event": 1,
                "event_cusum_direction": 1,
                "decision": "BUY",
                "confidence": 0.6,
            }
        ]
    )

    metrics, cm = trade_skip_model_metrics(
        signals,
        prepared,
        horizon=1,
        tp_threshold=0.0008,
        sl_threshold=0.0004,
        cost_per_trade=0.00003,
    )

    assert metrics["Actual profitable rate"] == 0.0
    assert metrics["Predicted trade rate"] == 1.0
    assert metrics["F1 TRADE"] == 0.0
    assert metrics["F1 macro"] == 0.0
    assert np.array_equal(cm, np.array([[0, 1], [0, 0]]))
