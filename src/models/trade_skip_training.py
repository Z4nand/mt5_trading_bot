from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src import config
from src.features.event_detector import detect_events
from src.features.feature_pipeline import generate_features, missing_feature_columns
from src.models.sequence_models import create_model
from src.strategy.signal_generator import prepare_rule_frame, rule_direction_from_row
from src.strategy.backtest import _resolve_exit


class TradeSkipSequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_indices: np.ndarray,
        sequence_length: int = config.SEQUENCE_LENGTH,
    ):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.sample_indices = sample_indices.astype(np.int64)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        end_idx = self.sample_indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        x = self.features[start_idx : end_idx + 1]
        y = self.labels[end_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def event_direction_to_decision(direction: int) -> str | None:
    if direction == 1:
        return "BUY"
    if direction == -1:
        return "SELL"
    return None


def trade_success_label_at(
    df: pd.DataFrame,
    entry_idx: int,
    horizon: int,
    tp_threshold: float,
    sl_threshold: float,
    min_edge: float = 0.0,
) -> int | None:
    direction = event_direction_to_decision(int(df.iloc[entry_idx].get("event_cusum_direction", 0)))
    if direction is None:
        return None

    entry_price = float(df.iloc[entry_idx]["close"])
    exit_idx, exit_price, _ = _resolve_exit(
        price_df=df,
        entry_idx=entry_idx,
        direction=direction,
        horizon=horizon,
        tp_threshold=tp_threshold,
        sl_threshold=sl_threshold,
    )
    if exit_idx <= entry_idx:
        return None

    pnl = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
    return int(pnl > min_edge)


def build_trade_skip_frame(
    price_df: pd.DataFrame,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    tp_threshold: float = config.DEFAULT_TP_THRESHOLD,
    sl_threshold: float = config.DEFAULT_SL_THRESHOLD,
    feature_columns: list[str] | None = None,
    min_edge: float = 0.0,
    direction_rule: str | None = None,
    cusum_volatility_window: int | None = None,
    cusum_threshold_mult: float | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or config.FEATURE_COLUMNS
    if direction_rule is None:
        df = detect_events(generate_features(price_df))
    else:
        df = prepare_rule_frame(
            price_df,
            volatility_window=cusum_volatility_window or config.ACTIVE_RULE_CUSUM_VOLATILITY_WINDOW,
            threshold_mult=cusum_threshold_mult or config.ACTIVE_RULE_CUSUM_THRESHOLD_MULT,
            feature_columns=feature_columns,
        )
        encoded_directions = []
        rule_decisions = []
        for _, row in df.iterrows():
            if int(row.get("event", 0)) != 1:
                encoded_directions.append(0)
                rule_decisions.append(None)
                continue
            decision = rule_direction_from_row(row, direction_rule)
            rule_decisions.append(decision)
            encoded_directions.append(1 if decision == "BUY" else -1 if decision == "SELL" else 0)
        df = df.copy()
        df["original_event_cusum_direction"] = df["event_cusum_direction"]
        df["event_cusum_direction"] = encoded_directions
        df["rule_decision"] = rule_decisions
    missing = missing_feature_columns(df, feature_columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    labels = []
    for idx in range(len(df)):
        if int(df.iloc[idx].get("event", 0)) != 1:
            labels.append(None)
        else:
            labels.append(
                trade_success_label_at(
                    df,
                    entry_idx=idx,
                    horizon=horizon,
                    tp_threshold=tp_threshold,
                    sl_threshold=sl_threshold,
                    min_edge=min_edge,
                )
            )

    df = df.copy()
    df["trade_success"] = labels
    if "rule_decision" not in df:
        df["rule_decision"] = df["event_cusum_direction"].map({1: "BUY", -1: "SELL"})
    return df


def prepare_trade_skip_arrays(
    labeled_df: pd.DataFrame,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    feature_columns = feature_columns or config.FEATURE_COLUMNS
    feature_frame = (
        labeled_df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    labels = labeled_df["trade_success"].fillna(-1).to_numpy(dtype=np.int64)

    index_tz = labeled_df.index.tz
    train_end = pd.Timestamp(config.TRAIN_END_DATE)
    valid_end = pd.Timestamp(config.VALID_END_DATE)
    if index_tz is not None:
        train_end = train_end.tz_localize(index_tz)
        valid_end = valid_end.tz_localize(index_tz)

    train_time_mask = labeled_df.index < train_end
    valid_time_mask = (labeled_df.index >= train_end) & (labeled_df.index < valid_end)
    test_time_mask = labeled_df.index >= valid_end

    scaler = StandardScaler()
    scaler.fit(feature_frame.loc[train_time_mask])
    scaled = scaler.transform(feature_frame).astype(np.float32)

    sample_mask = labeled_df["trade_success"].notna()
    sample_positions = np.flatnonzero(sample_mask.to_numpy())
    sample_positions = sample_positions[sample_positions >= config.SEQUENCE_LENGTH - 1]

    train_time_mask = np.asarray(train_time_mask)
    valid_time_mask = np.asarray(valid_time_mask)
    test_time_mask = np.asarray(test_time_mask)

    train_end_pos = labeled_df.index.searchsorted(train_end, side="left")
    valid_end_pos = labeled_df.index.searchsorted(valid_end, side="left")
    has_future_in_train = (sample_positions + horizon) < train_end_pos
    has_future_in_valid = (sample_positions + horizon) < valid_end_pos

    train_indices = sample_positions[train_time_mask[sample_positions] & has_future_in_train]
    valid_indices = sample_positions[valid_time_mask[sample_positions] & has_future_in_valid]
    test_indices = sample_positions[test_time_mask[sample_positions]]
    return scaled, labels, train_indices, valid_indices, test_indices, scaler


def class_weights(labels: np.ndarray, train_indices: np.ndarray, device: torch.device) -> torch.Tensor | None:
    y = labels[train_indices]
    y = y[y >= 0]
    counts = np.bincount(y, minlength=2).astype(np.float32)
    if np.any(counts == 0):
        return None
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_trade_skip_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true = []
    y_pred = []
    probabilities = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probability_trade = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            pred = (probability_trade >= 0.5).astype(int)
            probabilities.extend(probability_trade.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    if not y_true:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "count": 0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidence = np.maximum(probabilities, 1 - np.array(probabilities))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "count": int(len(y_true)),
        "avg_confidence": float(np.mean(confidence)),
    }


def train_trade_skip_model(
    price_df: pd.DataFrame,
    model_type: str = "gru",
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    tp_threshold: float = config.DEFAULT_TP_THRESHOLD,
    sl_threshold: float = config.DEFAULT_SL_THRESHOLD,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    hidden_size: int = 64,
    dropout: float = 0.25,
    num_layers: int = 2,
    selection_metric: str = "balanced_accuracy",
    feature_columns: list[str] | None = None,
    min_edge: float = 0.0,
    direction_rule: str | None = None,
    cusum_volatility_window: int | None = None,
    cusum_threshold_mult: float | None = None,
) -> dict:
    feature_columns = feature_columns or config.FEATURE_COLUMNS
    labeled_df = build_trade_skip_frame(
        price_df,
        horizon=horizon,
        tp_threshold=tp_threshold,
        sl_threshold=sl_threshold,
        feature_columns=feature_columns,
        min_edge=min_edge,
        direction_rule=direction_rule,
        cusum_volatility_window=cusum_volatility_window,
        cusum_threshold_mult=cusum_threshold_mult,
    )
    scaled, labels, train_indices, valid_indices, test_indices, scaler = prepare_trade_skip_arrays(
        labeled_df,
        horizon=horizon,
        feature_columns=feature_columns,
    )

    train_ds = TradeSkipSequenceDataset(scaled, labels, train_indices)
    valid_ds = TradeSkipSequenceDataset(scaled, labels, valid_indices)
    test_ds = TradeSkipSequenceDataset(scaled, labels, test_indices)
    if len(train_ds) == 0 or len(valid_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Not enough samples: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_type=model_type,
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(labels, train_indices, device))

    metric_aliases = {
        "accuracy": "accuracy",
        "acc": "accuracy",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "balanced_acc": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
    }
    selection_metric = metric_aliases.get(selection_metric, selection_metric)
    if selection_metric not in {"accuracy", "f1", "balanced_accuracy", "precision", "recall"}:
        raise ValueError("Unsupported selection_metric")

    best_state = None
    best_score = -1.0
    patience = 10
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y)

        metrics = evaluate_trade_skip_model(model, valid_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / max(len(train_ds), 1)
        history.append(metrics)

        if metrics[selection_metric] > best_score:
            best_score = metrics[selection_metric]
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_trade_skip_model(model, test_loader, device)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "test_metrics": test_metrics,
        "samples": len(labeled_df),
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
        "selection_metric": selection_metric,
        "best_valid_score": best_score,
        "feature_columns": feature_columns,
        "min_edge": min_edge,
        "direction_rule": direction_rule,
        "cusum_volatility_window": cusum_volatility_window,
        "cusum_threshold_mult": cusum_threshold_mult,
    }


def generate_trade_skip_signal_history(
    price_df: pd.DataFrame,
    model: torch.nn.Module,
    scaler: StandardScaler,
    feature_columns: list[str],
    threshold: float = 0.55,
    max_rows: int = 500,
    direction_rule: str | None = None,
    cusum_volatility_window: int | None = None,
    cusum_threshold_mult: float | None = None,
) -> pd.DataFrame:
    if direction_rule is None:
        df = detect_events(generate_features(price_df))
    else:
        df = prepare_rule_frame(
            price_df,
            volatility_window=cusum_volatility_window or config.ACTIVE_RULE_CUSUM_VOLATILITY_WINDOW,
            threshold_mult=cusum_threshold_mult or config.ACTIVE_RULE_CUSUM_THRESHOLD_MULT,
            feature_columns=feature_columns,
        )
        encoded_directions = []
        for _, row in df.iterrows():
            if int(row.get("event", 0)) != 1:
                encoded_directions.append(0)
                continue
            decision = rule_direction_from_row(row, direction_rule)
            encoded_directions.append(1 if decision == "BUY" else -1 if decision == "SELL" else 0)
        df = df.copy()
        df["original_event_cusum_direction"] = df["event_cusum_direction"]
        df["event_cusum_direction"] = encoded_directions
    feature_frame = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    scaled = scaler.transform(feature_frame)
    start = max(config.SEQUENCE_LENGTH - 1, len(df) - max_rows)
    positions = np.arange(start, len(df))
    device = next(model.parameters()).device
    probabilities = np.empty(len(positions), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(positions), 512):
            batch_positions = positions[batch_start : batch_start + 512]
            windows = np.stack([scaled[i - config.SEQUENCE_LENGTH + 1 : i + 1] for i in batch_positions])
            x = torch.tensor(windows, dtype=torch.float32).to(device)
            logits = model(x)
            probabilities[batch_start : batch_start + len(batch_positions)] = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    rows = []
    for i, probability_trade in zip(positions, probabilities):
        event = int(df.iloc[i].get("event", 0))
        direction_value = int(df.iloc[i].get("event_cusum_direction", 0))
        rule_decision = event_direction_to_decision(direction_value)
        decision = rule_decision if event == 1 and rule_decision is not None and probability_trade >= threshold else "NO TRADE"
        rows.append(
            {
                "time": df.index[i],
                "event": event,
                "event_cusum_direction": direction_value,
                "prediction": "TRADE" if probability_trade >= threshold else "SKIP",
                "probability_trade": float(probability_trade),
                "confidence": float(probability_trade),
                "decision": decision,
                "close": df.iloc[i]["close"],
            }
        )

    return pd.DataFrame(rows)
