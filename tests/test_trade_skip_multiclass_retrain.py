import numpy as np
import pandas as pd

from scripts.trade_skip_multiclass_retrain import _add_multiclass_labels, _threshold_metrics


def test_add_multiclass_labels_uses_neutral_band():
    frame = pd.DataFrame({"event_pnl": [-0.0002, -0.00001, 0.0, 0.00001, 0.0002]})

    result = _add_multiclass_labels(frame, neutral_band_pips=0.5)

    assert result["trade_quality"].tolist() == [0, 1, 1, 1, 2]


def test_threshold_metrics_selects_good_probability_and_scores_pnl():
    frame = pd.DataFrame(
        {
            "prob_good": [0.40, 0.55, 0.70],
            "trade_quality": [0, 2, 2],
            "event_pnl": [-0.0008, 0.0002, 0.0008],
        }
    )

    metrics = _threshold_metrics(frame, threshold=0.50)

    assert metrics["selected_trades"] == 2
    assert metrics["selected_good_rate_pct"] == 100.0
    assert np.isclose(metrics["selected_sum_pnl_pips"], 10.0)
