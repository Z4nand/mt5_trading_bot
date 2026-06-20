from pathlib import Path

import torch
import torch.nn as nn

from src.config import BIDIRECTIONAL, DROPOUT, HIDDEN_SIZE, INPUT_SIZE, MODEL_TYPE, NUM_CLASSES, NUM_LAYERS


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        bidirectional: bool = BIDIRECTIONAL,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.head(out)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        bidirectional: bool = BIDIRECTIONAL,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out)


def infer_model_type(model_path: Path | None, fallback: str = MODEL_TYPE) -> str:
    if model_path is not None and "lstm" in model_path.name.lower():
        return "lstm"
    if model_path is not None and "gru" in model_path.name.lower():
        return "gru"
    return fallback.lower()

def create_model(
    model_type: str = MODEL_TYPE,
    input_size: int = INPUT_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    dropout: float = DROPOUT,
    num_layers: int = NUM_LAYERS,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    
    if model_type == "gru":
        return GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    if model_type == "lstm":
        return LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Use 'gru' or 'lstm'.")


def load_model(model_path: Path, device: torch.device | None = None, model_type: str | None = None) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Файл весов не найден: {model_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type=model_type or infer_model_type(model_path)).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_model_with_config(model_path, config_path, device=None):
    import joblib

    model_config = joblib.load(config_path)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        model_type=model_config["model_type"],
        input_size=model_config.get("input_size", len(model_config.get("feature_columns", [])) or INPUT_SIZE),
        hidden_size=model_config["hidden_size"],
        dropout=model_config["dropout"],
        num_layers=model_config["num_layers"],
        num_classes=model_config.get("num_classes", NUM_CLASSES),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model
