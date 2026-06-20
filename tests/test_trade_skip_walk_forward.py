import pandas as pd

from scripts.trade_skip_walk_forward import _choose_threshold, walk_forward_thresholds


def _frame() -> pd.DataFrame:
    rows = []
    for year in [2020, 2021, 2022, 2023]:
        for idx, probability in enumerate([0.40, 0.51, 0.53, 0.60]):
            actual = int(probability >= 0.53)
            pnl = 0.0008 if actual else -0.0008
            rows.append(
                {
                    "year": year,
                    "prob_trade": probability,
                    "actual_trade": actual,
                    "pnl": pnl,
                    "split": "train_2015_2020" if year == 2020 else "valid_2021_2023",
                }
            )
    return pd.DataFrame(rows)


def test_choose_threshold_respects_min_selected_trades():
    threshold, metrics = _choose_threshold(
        train_frame=_frame(),
        thresholds=(0.50, 0.53, 0.61),
        objective="sum_pnl",
        min_selected_trades=2,
    )

    assert threshold == 0.53
    assert metrics["selected_trades"] == 8
    assert metrics["sum_pnl_selected_pips"] > 0


def test_walk_forward_thresholds_uses_only_past_years():
    result = walk_forward_thresholds(
        diagnostics=_frame(),
        thresholds=(0.50, 0.53),
        train_years=2,
        objective="sum_pnl",
        min_train_events=1,
        min_selected_trades=1,
    )

    assert result["test_year"].tolist() == [2022, 2023]
    assert result["train_start_year"].tolist() == [2020, 2021]
    assert result["train_end_year"].tolist() == [2021, 2022]
    assert set(result["threshold"]) == {0.53}
    assert (result["test_sum_pnl_selected_pips"] > 0).all()
