from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import optuna

from src import config
from src.features.event_detector import detect_events
from src.features.feature_pipeline import generate_features, missing_feature_columns
from src.models.sequence_models import create_model
from src.strategy.signal_generator import first_hit_label_at


class DirectionSequenceDataset(Dataset):
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


def build_labeled_frame(
    price_df: pd.DataFrame,
    label_threshold: float = config.DEFAULT_LABEL_THRESHOLD,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or config.FEATURE_COLUMNS
    df = detect_events(generate_features(price_df))
    missing = missing_feature_columns(df, feature_columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    labels = []
    for idx in range(len(df)):
        labels.append(first_hit_label_at(df, idx, threshold=label_threshold, horizon=horizon))

    df = df.copy()
    df["direction_label"] = labels
    df["target"] = df["direction_label"].map({"DOWN": 0, "UP": 1})
    return df


def prepare_sequence_arrays(
    labeled_df: pd.DataFrame,
    event_only: bool = True,
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

    labels = labeled_df["target"].fillna(-1).to_numpy(dtype=np.int64)

    index_tz = labeled_df.index.tz

    train_end = pd.Timestamp(config.TRAIN_END_DATE)
    valid_end = pd.Timestamp(config.VALID_END_DATE)

    if index_tz is not None:
        train_end = train_end.tz_localize(index_tz)
        valid_end = valid_end.tz_localize(index_tz)

    train_time_mask = labeled_df.index < train_end

    valid_time_mask = (
        (labeled_df.index >= train_end)
        &
        (labeled_df.index < valid_end)
    )

    test_time_mask = labeled_df.index >= valid_end

    scaler = StandardScaler()
    scaler.fit(feature_frame.loc[train_time_mask])

    scaled = scaler.transform(feature_frame).astype(np.float32)

    sample_mask = labeled_df["target"].notna()

    if event_only:
        sample_mask &= labeled_df["event"].eq(1)

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


def train_direction_model(
    price_df: pd.DataFrame,
    model_type: str = "gru",
    event_only: bool = True,
    label_threshold: float = config.DEFAULT_LABEL_THRESHOLD,
    horizon: int = config.DEFAULT_HORIZON_CANDLES,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    hidden_size: int = 128,
    dropout: float = 0.2,
    num_layers: int = 2,
    selection_metric: str = "balanced_accuracy",
    feature_columns: list[str] | None = None,
    trial=None,
) -> dict:
    feature_columns = feature_columns or config.FEATURE_COLUMNS
    labeled_df = build_labeled_frame(
        price_df,
        label_threshold=label_threshold,
        horizon=horizon,
        feature_columns=feature_columns,
    )
    scaled, labels, train_indices, valid_indices, test_indices, scaler = prepare_sequence_arrays(
        labeled_df,
        event_only=event_only,
        horizon=horizon,
        feature_columns=feature_columns,
    )


    train_ds = DirectionSequenceDataset(scaled, labels, train_indices)
    valid_ds = DirectionSequenceDataset(scaled, labels, valid_indices)
    test_ds = DirectionSequenceDataset(scaled, labels, test_indices)

    if len(train_ds) == 0 or len(valid_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"Not enough labeled samples: "
            f"train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}. "
            "Check split dates, SEQUENCE_LENGTH, label_threshold, or data size."
        )
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
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    metric_aliases = {
        "accuracy": "accuracy",
        "acc": "accuracy",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "balanced_acc": "balanced_accuracy",
    }
    selection_metric = metric_aliases.get(selection_metric, selection_metric)
    if selection_metric not in {"accuracy", "f1", "balanced_accuracy"}:
        raise ValueError("selection_metric must be one of: accuracy, f1, balanced_accuracy")

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

        metrics = evaluate_torch_model(model, valid_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / max(len(train_ds), 1)
        history.append(metrics)

        if trial is not None:
            trial.report(metrics[selection_metric], epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
            
        #вывод метрик обучения
        # print(
        #     f"Epoch [{epoch}/{epochs}] | "
        #     f"train_loss: {metrics['train_loss']:.6f} | "
        #     f"val_acc: {metrics['accuracy']:.4f} | "
        #     f"val_f1: {metrics['f1']:.4f} | "
        #     f"balanced_acc: {metrics['balanced_accuracy']:.4f} | "
        #     f"avg_conf: {metrics.get('avg_confidence', 0):.4f}"
        # )

        if metrics[selection_metric] > best_score:
            best_score = metrics[selection_metric]
            epochs_without_improvement = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            # print(f"  -> Найдена лучшая модель. F1 = {best_f1:.4f}")

        else:
            epochs_without_improvement += 1

            # print(
            #     f"  -> Без улучшения: "
            #     f"{epochs_without_improvement}/{patience}"
            # )

            if epochs_without_improvement >= patience:
                # print("\nEarly stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_torch_model(model, test_loader, device)

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
        "event_only": event_only,
        "model_type": model_type,
        "selection_metric": selection_metric,
        "best_valid_score": best_score,
        "feature_columns": feature_columns,
    }


def evaluate_torch_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true = []
    y_pred = []
    probabilities = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probability_up = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            pred = (probability_up >= 0.5).astype(int)
            probabilities.extend(probability_up.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    if not y_true:
        return {"accuracy": 0.0, "f1": 0.0, "balanced_accuracy": 0.0, "count": 0}
    return classification_metrics(np.array(y_true), np.array(y_pred), np.array(probabilities))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probability_up: np.ndarray | None = None) -> dict:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "count": int(len(y_true)),
    }
    if probability_up is not None:
        confidence = np.maximum(probability_up, 1 - probability_up)
        result["avg_confidence"] = float(np.mean(confidence))
    return result
