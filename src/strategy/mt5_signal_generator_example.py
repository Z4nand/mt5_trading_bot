import torch


def predict_next_candle_direction(recent_df, model, scaler, feature_columns, sequence_length, device):
    if len(recent_df) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} rows for prediction.")

    model_input_df = recent_df[feature_columns].copy()
    scaled_recent = scaler.transform(model_input_df)
    scaled_recent = torch.FloatTensor(scaled_recent)
    sequence = scaled_recent[-sequence_length:].unsqueeze(0).to(device)

    class_names = {
        0: "SELL",
        1: "HOLD",
        2: "BUY",
    }

    model.eval()
    with torch.no_grad():
        logits = model(sequence)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return {
        "probabilities": probabilities.cpu().numpy()[0],
        "predicted_class": predicted_class,
        "signal": class_names[predicted_class],
    }


if __name__ == "__main__":
    print("Import predict_next_candle_direction and pass trained model inputs explicitly.")
