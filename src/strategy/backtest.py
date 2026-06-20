import numpy as np
import pandas as pd

from src.config import DEFAULT_SL_THRESHOLD, DEFAULT_TP_THRESHOLD, INSTRUMENT


def _resolve_exit(
    price_df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    horizon: int,
    tp_threshold: float | None,
    sl_threshold: float | None,
) -> tuple[int, float, str]:
    entry_price = float(price_df.iloc[entry_idx]["close"])
    horizon_exit_idx = min(entry_idx + horizon, len(price_df) - 1)

    if tp_threshold is None and sl_threshold is None:
        return horizon_exit_idx, float(price_df.iloc[horizon_exit_idx]["close"]), "HORIZON"

    for idx in range(entry_idx + 1, horizon_exit_idx + 1):
        candle = price_df.iloc[idx]
        high = float(candle["high"])
        low = float(candle["low"])

        if direction == "BUY":
            tp_price = entry_price + tp_threshold if tp_threshold is not None else None
            sl_price = entry_price - sl_threshold if sl_threshold is not None else None
            tp_hit = tp_price is not None and high >= tp_price
            sl_hit = sl_price is not None and low <= sl_price
            if sl_hit:
                return idx, sl_price, "SL"
            if tp_hit:
                return idx, tp_price, "TP"
        else:
            tp_price = entry_price - tp_threshold if tp_threshold is not None else None
            sl_price = entry_price + sl_threshold if sl_threshold is not None else None
            tp_hit = tp_price is not None and low <= tp_price
            sl_hit = sl_price is not None and high >= sl_price
            if sl_hit:
                return idx, sl_price, "SL"
            if tp_hit:
                return idx, tp_price, "TP"

    return horizon_exit_idx, float(price_df.iloc[horizon_exit_idx]["close"]), "HORIZON"


def build_trades(
    signals: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int = 4,
    tp_threshold: float | None = DEFAULT_TP_THRESHOLD,
    sl_threshold: float | None = DEFAULT_SL_THRESHOLD,
    cost_per_trade: float = 0.0,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    executable = signals[signals["decision"].isin(["BUY", "SELL"])].copy()
    if executable.empty:
        return pd.DataFrame()

    price_df = price_df.sort_index()
    tp_threshold = tp_threshold if tp_threshold and tp_threshold > 0 else None
    sl_threshold = sl_threshold if sl_threshold and sl_threshold > 0 else None
    rows = []
    for _, signal in executable.iterrows():
        entry_time = signal["time"]
        if entry_time not in price_df.index:
            entry_idx = price_df.index.get_indexer([entry_time], method="nearest")[0]
        else:
            entry_idx = price_df.index.get_loc(entry_time)
            if not isinstance(entry_idx, int):
                entry_idx = int(entry_idx[0])

        entry_price = float(price_df.iloc[entry_idx]["close"])
        direction = signal["decision"]
        exit_idx, exit_price, exit_reason = _resolve_exit(
            price_df=price_df,
            entry_idx=entry_idx,
            direction=direction,
            horizon=horizon,
            tp_threshold=tp_threshold,
            sl_threshold=sl_threshold,
        )
        gross_pnl = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        cost_per_trade = max(float(cost_per_trade), 0.0)
        pnl = gross_pnl - cost_per_trade

        rows.append(
            {
                "Дата": entry_time,
                "Инструмент": INSTRUMENT,
                "Сигнал": direction,
                "confidence": round(float(signal["confidence"]), 4),
                "Цена входа": round(entry_price, 5),
                "Цена выхода": round(exit_price, 5),
                "Результат": "WIN" if pnl > 0 else "LOSS",
                "Exit time": price_df.index[exit_idx],
                "Exit reason": exit_reason,
                "Gross PnL": round(gross_pnl, 5),
                "Cost": round(cost_per_trade, 5),
                "PnL": round(pnl, 5),
            }
        )

    return pd.DataFrame(rows)


def calculate_equity_curve(trades: pd.DataFrame, initial_capital: float = 10_000.0, lot_multiplier: float = 10000.0):
    if trades.empty:
        return pd.DataFrame({"Дата": [pd.Timestamp.now()], "Equity": [initial_capital]})
    equity = initial_capital + (trades["PnL"].astype(float) * lot_multiplier).cumsum()
    return pd.DataFrame({"Дата": trades["Дата"], "Equity": equity})


def calculate_trade_metrics(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> dict:
    if trades.empty:
        return {
            "total_return": 0.0,
            "winrate": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "trades": 0,
            "Total Return": 0.0,
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
            "Max Drawdown": 0.0,
            "Average Trade": 0.0,
            "Trades": 0,
            "Current Capital": initial_capital,
        }

    pnl = trades["PnL"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = initial_capital + (pnl * 10000.0).cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    total_return = (equity.iloc[-1] - initial_capital) / initial_capital

    return {
        "total_return": total_return,
        "winrate": len(wins) / len(trades),
        "max_drawdown": drawdown.min(),
        "profit_factor": profit_factor,
        "avg_trade": pnl.mean(),
        "trades": len(trades),
        "Total Return": total_return,
        "Win Rate": len(wins) / len(trades),
        "Profit Factor": profit_factor,
        "Max Drawdown": drawdown.min(),
        "Average Trade": pnl.mean(),
        "Trades": len(trades),
        "Current Capital": float(equity.iloc[-1]),
    }


def apply_risk_sizing(
    trades: pd.DataFrame,
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 1.0,
    sl_threshold: float | None = DEFAULT_SL_THRESHOLD,
    pip_value_per_lot: float = 10.0,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    sl_pips = abs(float(sl_threshold or DEFAULT_SL_THRESHOLD) * 10000.0)
    if sl_pips <= 0:
        sl_pips = abs(float(DEFAULT_SL_THRESHOLD) * 10000.0)
    risk_fraction = max(float(risk_per_trade_pct), 0.0) / 100.0
    pip_value_per_lot = max(float(pip_value_per_lot), 1e-9)

    result = trades.copy()
    equity = float(initial_capital)
    equities = []
    risk_amounts = []
    lot_sizes = []
    pnl_values = []
    pnl_pips_values = []

    for pnl in result["PnL"].astype(float):
        pnl_pips = pnl * 10000.0
        risk_amount = equity * risk_fraction
        lot_size = risk_amount / (sl_pips * pip_value_per_lot) if risk_amount > 0 else 0.0
        pnl_money = pnl_pips * pip_value_per_lot * lot_size
        equity += pnl_money

        pnl_pips_values.append(pnl_pips)
        risk_amounts.append(risk_amount)
        lot_sizes.append(lot_size)
        pnl_values.append(pnl_money)
        equities.append(equity)

    result["PnL pips"] = pnl_pips_values
    result["Risk Amount"] = risk_amounts
    result["Lot Size"] = lot_sizes
    result["PnL Money"] = pnl_values
    result["Equity"] = equities
    return result


def calculate_risk_trade_metrics(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> dict:
    if trades.empty or "PnL Money" not in trades:
        return {
            "total_return": 0.0,
            "winrate": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "trades": 0,
            "Total Return": 0.0,
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
            "Max Drawdown": 0.0,
            "Average Trade": 0.0,
            "Trades": 0,
            "Current Capital": initial_capital,
            "Total PnL Money": 0.0,
            "Average Trade Money": 0.0,
            "Max Drawdown Money": 0.0,
            "Average Lot Size": 0.0,
        }

    pnl = trades["PnL Money"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = trades["Equity"].astype(float)
    peak = equity.cummax()
    drawdown_money = equity - peak
    drawdown_pct = drawdown_money / peak
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    current_capital = float(equity.iloc[-1])
    total_return = (current_capital - initial_capital) / initial_capital if initial_capital else 0.0

    return {
        "total_return": total_return,
        "winrate": len(wins) / len(trades),
        "max_drawdown": float(drawdown_pct.min()),
        "profit_factor": profit_factor,
        "avg_trade": float(pnl.mean()),
        "trades": len(trades),
        "Total Return": total_return,
        "Win Rate": len(wins) / len(trades),
        "Profit Factor": profit_factor,
        "Max Drawdown": float(drawdown_pct.min()),
        "Average Trade": float(pnl.mean()),
        "Trades": len(trades),
        "Current Capital": current_capital,
        "Total PnL Money": float(pnl.sum()),
        "Average Trade Money": float(pnl.mean()),
        "Max Drawdown Money": float(drawdown_money.min()),
        "Average Lot Size": float(trades["Lot Size"].mean()) if "Lot Size" in trades else 0.0,
    }


def calculate_risk_equity_curve(trades: pd.DataFrame, initial_capital: float = 10_000.0):
    if trades.empty or "Equity" not in trades:
        return pd.DataFrame({"Дата": [pd.Timestamp.now()], "Equity": [initial_capital]})
    time_column = "Дата" if "Дата" in trades.columns else trades.columns[0]
    return pd.DataFrame({"Дата": trades[time_column], "Equity": trades["Equity"].astype(float)})
