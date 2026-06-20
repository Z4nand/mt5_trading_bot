import pandas as pd

from src.features.reversal_quality import REVERSAL_QUALITY_FEATURE_COLUMNS
from src.strategy.signal_generator import prepare_rule_frame


def test_prepare_rule_frame_adds_reversal_quality_features():
    index = pd.date_range("2024-01-01", periods=320, freq="15min", tz="UTC")
    close = pd.Series([1.10 + i * 0.00002 for i in range(len(index))], index=index)
    price = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "volume": 100,
        },
        index=index,
    )

    frame = prepare_rule_frame(
        price,
        volatility_window=96,
        threshold_mult=1.8,
        feature_columns=REVERSAL_QUALITY_FEATURE_COLUMNS,
    )

    assert set(REVERSAL_QUALITY_FEATURE_COLUMNS).issubset(frame.columns)
    assert frame[REVERSAL_QUALITY_FEATURE_COLUMNS].notna().all().all()
