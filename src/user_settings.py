import json
from pathlib import Path
from typing import Any

from src import config


USER_SETTINGS_DIR = config.BASE_DIR / "config" / "user_settings"
USER_SETTINGS_PATH = USER_SETTINGS_DIR / "settings.json"

DEFAULT_USER_SETTINGS: dict[str, Any] = {
    "robot_enabled": True,
    "selected_strategy": "TradeSkip GRU Reversal",
    "selected_model_type": config.MODEL_TYPE,
    "threshold": 0.51,
    "horizon": config.DEFAULT_HORIZON_CANDLES,
    "fixed_exit_enabled": True,
    "tp_pips": 8.0,
    "sl_pips": 4.0,
    "tp_threshold": 0.0008,
    "sl_threshold": 0.0004,
    "settings_tp_pips": 8.0,
    "settings_sl_pips": 4.0,
    "news_filter_enabled": True,
    "news_filter_minutes": 60,
    "include_costs": True,
    "spread_pips": 0.2,
    "slippage_pips": 0.1,
    "commission_pips": 0.0,
    "initial_balance": 1_000.0,
    "risk_per_trade_pct": 1.0,
    "pip_value_per_lot": 10.0,
}


def _coerce_like(default: Any, value: Any) -> Any:
    try:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError):
        return default
    return value


def normalize_user_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    normalized = DEFAULT_USER_SETTINGS.copy()
    if not isinstance(settings, dict):
        return normalized
    for key, default_value in DEFAULT_USER_SETTINGS.items():
        if key in settings:
            normalized[key] = _coerce_like(default_value, settings[key])

    if normalized["fixed_exit_enabled"]:
        normalized["tp_threshold"] = normalized["tp_pips"] / 10000.0
        normalized["sl_threshold"] = normalized["sl_pips"] / 10000.0
    return normalized


def load_user_settings(path: Path = USER_SETTINGS_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return DEFAULT_USER_SETTINGS.copy()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_USER_SETTINGS.copy()
    return normalize_user_settings(raw)


def save_user_settings(settings: dict[str, Any], path: Path = USER_SETTINGS_PATH) -> dict[str, Any]:
    normalized = normalize_user_settings(settings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return normalized
