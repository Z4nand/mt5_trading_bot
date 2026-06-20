import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import config
from src.connector.data_fetcher import load_all_price_data
from src.features.event_detector import (
    add_cusum_events,
    add_event_features,
)
from src.features.feature_pipeline import generate_features
from src.strategy.backtest import _resolve_exit


DEFAULT_VOL_WINDOWS = (48, 96, 192)
DEFAULT_THRESHOLD_MULTS = (1.8, 2.2, 2.8, 3.4, 4.0)
DEFAULT_DIRECTION_RULES = (
    "cusum_momentum",
    "cusum_reversal",
    "breakout_momentum",
    "breakout_reversal",
    "level_reversal",
    "regime_mixed",
)


def _build_event_frame(price_df: pd.DataFrame, volatility_window: int, threshold_mult: float) -> pd.DataFrame:
    frame = generate_features(price_df)
    frame = add_event_features(frame)
    frame = add_cusum_events(
        frame,
        volatility_window=volatility_window,
        threshold_mult=threshold_mult,
    )
    frame["near_resistance"] = (
        (frame["dist_to_prev_res"].abs() < config.NEAR_LEVEL_THRESHOLD)
        | (frame["dist_to_prev_res_96"].abs() < config.NEAR_LEVEL_THRESHOLD)
    )
    frame["near_support"] = (
        (frame["dist_to_prev_sup"].abs() < config.NEAR_LEVEL_THRESHOLD)
        | (frame["dist_to_prev_sup_96"].abs() < config.NEAR_LEVEL_THRESHOLD)
    )
    frame["near_level"] = frame["near_resistance"] | frame["near_support"]
    frame["strong_range"] = frame["range_ratio_20"] > config.STRONG_RANGE_MULTIPLIER
    frame["strong_body"] = frame["body_ratio_20"] > config.STRONG_BODY_MULTIPLIER
    frame["strong_candle"] = frame["strong_range"] | frame["strong_body"]
    frame["breakout_up"] = (frame["close"] > frame["prev_resistance_50"]) | (frame["close"] > frame["prev_resistance_96"])
    frame["breakout_down"] = (frame["close"] < frame["prev_support_50"]) | (frame["close"] < frame["prev_support_96"])
    frame["breakout"] = frame["breakout_up"] | frame["breakout_down"]
    frame["event"] = (
        (frame["event_cusum"] == 1)
        & (frame["near_level"] | frame["strong_candle"] | frame["breakout"])
    ).astype(int)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _fit_regime_thresholds(frame: pd.DataFrame) -> dict:
    train_end = pd.Timestamp(config.TRAIN_END_DATE)
    if frame.index.tz is not None:
        train_end = train_end.tz_localize(frame.index.tz)
    train = frame[frame.index < train_end]
    return {
        "vol_high": float(train["volatility_96"].quantile(0.67)),
        "trend_abs": float(train["ma_50_diff"].abs().quantile(0.60)),
    }


def _direction_for_row(row: pd.Series, rule: str, thresholds: dict) -> str | None:
    cusum_direction = int(row["event_cusum_direction"])
    if rule == "cusum_momentum":
        if cusum_direction == 1:
            return "BUY"
        if cusum_direction == -1:
            return "SELL"
        return None

    if rule == "cusum_reversal":
        if cusum_direction == 1:
            return "SELL"
        if cusum_direction == -1:
            return "BUY"
        return None

    breakout_up = bool(row["breakout_up"])
    breakout_down = bool(row["breakout_down"])
    near_support = bool(row["near_support"])
    near_resistance = bool(row["near_resistance"])

    if rule == "breakout_momentum":
        if breakout_up and not breakout_down:
            return "BUY"
        if breakout_down and not breakout_up:
            return "SELL"
        return None

    if rule == "breakout_reversal":
        if breakout_up and not breakout_down:
            return "SELL"
        if breakout_down and not breakout_up:
            return "BUY"
        return None

    if rule == "level_reversal":
        if near_support and not near_resistance:
            return "BUY"
        if near_resistance and not near_support:
            return "SELL"
        return None

    if rule == "regime_mixed":
        is_range = abs(float(row["ma_50_diff"])) <= thresholds["trend_abs"]
        is_low_or_mid_vol = float(row["volatility_96"]) < thresholds["vol_high"]
        if is_range and is_low_or_mid_vol:
            return _direction_for_row(row, "level_reversal", thresholds)
        if breakout_up or breakout_down:
            return _direction_for_row(row, "breakout_momentum", thresholds)
        return _direction_for_row(row, "cusum_momentum", thresholds)

    raise ValueError(f"Unsupported direction rule: {rule}")


def _trade_rows(frame: pd.DataFrame, rule: str, thresholds: dict) -> pd.DataFrame:
    event_positions = np.flatnonzero(frame["event"].to_numpy(dtype=bool))
    rows = []
    for position in event_positions:
        row = frame.iloc[position]
        direction = _direction_for_row(row, rule, thresholds)
        if direction is None:
            continue
        entry_price = float(row["close"])
        _, exit_price, exit_reason = _resolve_exit(
            price_df=frame,
            entry_idx=position,
            direction=direction,
            horizon=config.DEFAULT_HORIZON_CANDLES,
            tp_threshold=config.DEFAULT_TP_THRESHOLD,
            sl_threshold=config.DEFAULT_SL_THRESHOLD,
        )
        pnl = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        rows.append(
            {
                "time": frame.index[position],
                "year": frame.index[position].year,
                "direction_rule": rule,
                "direction": direction,
                "pnl": pnl,
                "pnl_pips": pnl * 10000,
                "win": pnl > 0,
                "exit_reason": exit_reason,
                "event_cusum_direction": int(row["event_cusum_direction"]),
                "near_support": bool(row["near_support"]),
                "near_resistance": bool(row["near_resistance"]),
                "breakout_up": bool(row["breakout_up"]),
                "breakout_down": bool(row["breakout_down"]),
                "volatility_96": float(row["volatility_96"]),
                "ma_50_diff": float(row["ma_50_diff"]),
            }
        )
    return pd.DataFrame(rows)


def _summarize(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(group_columns, dropna=False)
        .agg(
            trades=("pnl", "size"),
            wins=("win", "sum"),
            winrate_pct=("win", lambda item: float(item.mean() * 100)),
            avg_pnl_pips=("pnl_pips", "mean"),
            total_pnl_pips=("pnl_pips", "sum"),
            median_pnl_pips=("pnl_pips", "median"),
        )
        .reset_index()
    )


def run_direction_diagnostics(
    volatility_windows: tuple[int, ...],
    threshold_mults: tuple[float, ...],
    direction_rules: tuple[str, ...],
) -> dict[str, pd.DataFrame | dict]:
    price_df, loaded_files = load_all_price_data(config.DATA_DIR)
    all_trades = []
    config_rows = []

    for volatility_window in volatility_windows:
        for threshold_mult in threshold_mults:
            frame = _build_event_frame(price_df, volatility_window, threshold_mult)
            regime_thresholds = _fit_regime_thresholds(frame)
            config_rows.append(
                {
                    "volatility_window": volatility_window,
                    "threshold_mult": threshold_mult,
                    "events": int(frame["event"].sum()),
                    "event_rate_pct": float(frame["event"].mean() * 100),
                    **regime_thresholds,
                }
            )
            for rule in direction_rules:
                trades = _trade_rows(frame, rule, regime_thresholds)
                if trades.empty:
                    continue
                trades.insert(0, "volatility_window", volatility_window)
                trades.insert(1, "threshold_mult", threshold_mult)
                all_trades.append(trades)
                print(
                    f"window={volatility_window} mult={threshold_mult:.2f} "
                    f"rule={rule} trades={len(trades)} pnl={trades['pnl_pips'].sum():.1f}"
                )

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    by_config_rule = _summarize(trades_all, ["volatility_window", "threshold_mult", "direction_rule"])
    by_year = _summarize(trades_all, ["volatility_window", "threshold_mult", "direction_rule", "year"])
    if not by_year.empty:
        stability = (
            by_year.assign(positive_year=by_year["total_pnl_pips"] > 0)
            .groupby(["volatility_window", "threshold_mult", "direction_rule"])
            .agg(
                positive_years=("positive_year", "sum"),
                years=("year", "size"),
                worst_year_pnl_pips=("total_pnl_pips", "min"),
                best_year_pnl_pips=("total_pnl_pips", "max"),
                median_year_pnl_pips=("total_pnl_pips", "median"),
            )
            .reset_index()
        )
        by_config_rule = by_config_rule.merge(
            stability,
            on=["volatility_window", "threshold_mult", "direction_rule"],
            how="left",
        )
    by_config_rule = by_config_rule.sort_values(
        ["total_pnl_pips", "positive_years", "avg_pnl_pips"],
        ascending=[False, False, False],
    )
    summary = {
        "loaded_files": len(loaded_files),
        "raw_rows": len(price_df),
        "volatility_windows": ",".join(str(item) for item in volatility_windows),
        "threshold_mults": ",".join(str(item) for item in threshold_mults),
        "direction_rules": ",".join(direction_rules),
        "variants": len(by_config_rule),
        "best_variant": by_config_rule.iloc[0].to_dict() if not by_config_rule.empty else None,
    }
    return {
        "summary": summary,
        "event_configs": pd.DataFrame(config_rows),
        "direction_trades": trades_all,
        "by_config_rule": by_config_rule,
        "by_year": by_year,
    }


def _print_report(outputs: dict[str, pd.DataFrame | dict], top_n: int) -> None:
    print("\nSUMMARY")
    for key, value in outputs["summary"].items():
        print(f"{key}: {value}")
    top = outputs["by_config_rule"].head(top_n)
    columns = [
        "volatility_window",
        "threshold_mult",
        "direction_rule",
        "trades",
        "winrate_pct",
        "avg_pnl_pips",
        "total_pnl_pips",
        "positive_years",
        "years",
        "worst_year_pnl_pips",
    ]
    print(f"\nTOP {top_n}")
    print(top[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate alternative event direction rules and CUSUM parameters.")
    parser.add_argument("--volatility-windows", type=int, nargs="*", default=list(DEFAULT_VOL_WINDOWS))
    parser.add_argument("--threshold-mults", type=float, nargs="*", default=list(DEFAULT_THRESHOLD_MULTS))
    parser.add_argument("--direction-rules", nargs="*", default=list(DEFAULT_DIRECTION_RULES))
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR / "processed" / "event_direction_diagnostics",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    outputs = run_direction_diagnostics(
        volatility_windows=tuple(args.volatility_windows),
        threshold_mults=tuple(args.threshold_mults),
        direction_rules=tuple(args.direction_rules),
    )
    _print_report(outputs, args.top_n)
    if not args.no_save:
        _save_outputs(outputs, args.output_dir)
        print(f"\nSaved CSV diagnostics to: {args.output_dir}")


def _save_outputs(outputs: dict[str, pd.DataFrame | dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{name}.csv", index=False)
        else:
            pd.Series(value).to_frame("value").to_csv(output_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
