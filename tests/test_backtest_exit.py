import pandas as pd

from src.strategy.backtest import _resolve_exit, build_trades, calculate_risk_equity_curve


def test_resolve_exit_uses_fixed_price_delta_for_buy_tp():
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000],
            "high": [1.1000, 1.1008],
            "low": [1.1000, 1.0998],
            "close": [1.1000, 1.1005],
        }
    )

    exit_idx, exit_price, reason = _resolve_exit(
        frame,
        entry_idx=0,
        direction="BUY",
        horizon=1,
        tp_threshold=0.0008,
        sl_threshold=0.0004,
    )

    assert exit_idx == 1
    assert exit_price == 1.1008
    assert reason == "TP"


def test_resolve_exit_uses_fixed_price_delta_for_sell_sl():
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000],
            "high": [1.1000, 1.1004],
            "low": [1.1000, 1.0999],
            "close": [1.1000, 1.1002],
        }
    )

    exit_idx, exit_price, reason = _resolve_exit(
        frame,
        entry_idx=0,
        direction="SELL",
        horizon=1,
        tp_threshold=0.0008,
        sl_threshold=0.0004,
    )

    assert exit_idx == 1
    assert exit_price == 1.1004
    assert reason == "SL"


def test_build_trades_reports_net_pips_after_cost():
    index = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    price = pd.DataFrame(
        {
            "open": [1.1000, 1.1000],
            "high": [1.1000, 1.1008],
            "low": [1.1000, 1.0998],
            "close": [1.1000, 1.1005],
        },
        index=index,
    )
    signals = pd.DataFrame(
        [{"time": index[0], "decision": "BUY", "confidence": 0.6}]
    )

    trades = build_trades(signals, price, horizon=1, tp_threshold=0.0008, sl_threshold=0.0004, cost_per_trade=0.00003)

    assert round(float(trades.iloc[0]["Gross PnL"]) * 10000, 1) == 8.0
    assert round(float(trades.iloc[0]["PnL"]) * 10000, 1) == 7.7


def test_empty_risk_equity_curve_uses_date_column():
    equity = calculate_risk_equity_curve(pd.DataFrame(), initial_capital=1000.0)

    assert list(equity.columns) == ["Дата", "Equity"]
    assert float(equity.iloc[0]["Equity"]) == 1000.0
