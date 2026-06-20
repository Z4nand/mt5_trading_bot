from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

MODEL_TYPE = "gru"  # "gru" or "lstm"

GRU_EVENT_MODEL_PATH = MODELS_DIR / "gru_event_direction_best.pth"
GRU_EVENT_SCALER_PATH = MODELS_DIR / "gru_event_direction_scaler.pkl"
LSTM_EVENT_MODEL_PATH = MODELS_DIR / "lstm_event_direction_best.pth"
LSTM_EVENT_SCALER_PATH = MODELS_DIR / "lstm_event_direction_scaler.pkl"

GRU_FULL_MODEL_PATH = MODELS_DIR / "gru_full_direction_best.pth"
GRU_FULL_SCALER_PATH = MODELS_DIR / "gru_full_direction_scaler.pkl"
LSTM_FULL_MODEL_PATH = MODELS_DIR / "lstm_full_direction_best.pth"
LSTM_FULL_SCALER_PATH = MODELS_DIR / "lstm_full_direction_scaler.pkl"

EVENT_MODEL_PATHS = {
    "gru": GRU_EVENT_MODEL_PATH,
    "lstm": LSTM_EVENT_MODEL_PATH,
}
EVENT_SCALER_PATHS = {
    "gru": GRU_EVENT_SCALER_PATH,
    "lstm": LSTM_EVENT_SCALER_PATH,
}
FULL_MODEL_PATHS = {
    "gru": GRU_FULL_MODEL_PATH,
    "lstm": LSTM_FULL_MODEL_PATH,
}
FULL_SCALER_PATHS = {
    "gru": GRU_FULL_SCALER_PATH,
    "lstm": LSTM_FULL_SCALER_PATH,
}

MODEL_PATH = EVENT_MODEL_PATHS[MODEL_TYPE]
SCALER_PATH = EVENT_SCALER_PATHS[MODEL_TYPE]

# параметры запуска модели ==================
GRU_EVENT_CONFIG_PATH = MODELS_DIR / "gru_event_direction_config.pkl"
LSTM_EVENT_CONFIG_PATH = MODELS_DIR / "lstm_event_direction_config.pkl"

EVENT_CONFIG_PATHS = {
    "gru": GRU_EVENT_CONFIG_PATH,
    "lstm": LSTM_EVENT_CONFIG_PATH,
}

GRU_FULL_CONFIG_PATH = MODELS_DIR / "gru_full_direction_config.pkl"
LSTM_FULL_CONFIG_PATH = MODELS_DIR / "lstm_full_direction_config.pkl"

FULL_CONFIG_PATHS = {
    "gru": GRU_FULL_CONFIG_PATH,
    "lstm": LSTM_FULL_CONFIG_PATH,
}
#=============
INSTRUMENT = "EURUSD"

TRAIN_END_DATE = "2021-01-01"
VALID_END_DATE = "2024-01-01"

FEATURE_COLUMNS = [
    "returns",
    "log_returns",
    "momentum_4",
    "momentum_8",
    "momentum_16",
    "momentum_32",
    "momentum_96",
    "volatility_20",
    "volatility_96",
    "range_pct",
    "body_pct",
    "body_abs_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_position",
    "ma_10_diff",
    "ma_20_diff",
    "ma_50_diff",
    "ema_12_diff",
    "ema_26_diff",
    "rsi_14_norm",
    "volume_change",
    "volume_zscore_96",
    "time_sin",
    "time_cos",
    "dow_sin",
    "dow_cos",
    "dist_to_res",
    "dist_to_sup",
    "break_res",
    "break_sup",
    "dist_to_prev_res",
    "dist_to_prev_sup",
    "dist_to_prev_res_96",
    "dist_to_prev_sup_96",
    "range_ratio_20",
    "body_ratio_20",
    "near_resistance",
    "near_support",
    "strong_range",
    "strong_body",
    "breakout_up",
    "breakout_down",
    "event_cusum_direction",
]

INPUT_SIZE = len(FEATURE_COLUMNS)
HIDDEN_SIZE = 96
NUM_LAYERS = 2
DROPOUT = 0.25
BIDIRECTIONAL = True
NUM_CLASSES = 2
SEQUENCE_LENGTH = 96

DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_HORIZON_CANDLES = 8
DEFAULT_RISK_PER_TRADE = 1.0
DEFAULT_LABEL_THRESHOLD = 0.0005
DEFAULT_TP_THRESHOLD = 0.0008
DEFAULT_SL_THRESHOLD = 0.0004

# Parameters copied from the notebook event detector.
CUSUM_VOLATILITY_WINDOW = 96
CUSUM_THRESHOLD_MULT = 2.8
NEAR_LEVEL_THRESHOLD = 0.00045
STRONG_RANGE_MULTIPLIER = 1.8
STRONG_BODY_MULTIPLIER = 1.6

ACTIVE_RULE_DIRECTION = "cusum_reversal"
ACTIVE_RULE_CUSUM_VOLATILITY_WINDOW = 192
ACTIVE_RULE_CUSUM_THRESHOLD_MULT = 1.8

TRADE_SKIP_REVERSAL_GRU_MODEL_PATH = MODELS_DIR / "trade_skip_reversal_gru_best.pth"
TRADE_SKIP_REVERSAL_GRU_SCALER_PATH = MODELS_DIR / "trade_skip_reversal_gru_scaler.pkl"
TRADE_SKIP_REVERSAL_GRU_CONFIG_PATH = MODELS_DIR / "trade_skip_reversal_gru_config.pkl"
