# Experiment index

## Full-data direction GRU

- Notebook: `notebooks/01_test_training.ipynb`
- Follow-up notebook: `notebooks/08_test_full_data.ipynb`
- Saved model files:
  - `data/models/gru_full_direction_best.pth`
  - `data/models/gru_full_direction_scaler.pkl`
  - `data/models/gru_full_direction_config.pkl`
- First saved: 2026-05-08 20:45:28

This is the experiment trained on the full dataset rather than only event/CUSUM rows.

## Walk-forward full-data retrain checks

- `data/processed/walk_forward_retrain_full_3epoch/summary.csv`: -277.9708 test pips, 5/9 positive folds
- `data/processed/walk_forward_retrain_full_3epoch_edge05/summary.csv`: -800.7509 test pips, 2/9 positive folds
- `data/processed/walk_forward_retrain_full_3epoch_edge1/summary.csv`: -556.8954 test pips, 4/9 positive folds

These runs did not reproduce a strong walk-forward result.

## Best found walk-forward result

- Directory: `data/processed/walk_forward_retrain_cusum_reversal_192_18/`
- Summary: `data/processed/walk_forward_retrain_cusum_reversal_192_18/summary.csv`
- Model setup: GRU, 3 train years, 3 epochs, CUSUM reversal direction
- Parameters: `cusum_volatility_window=192`, `cusum_threshold_mult=1.8`
- Result: +3443.9606 selected test pips, 7/9 positive test folds

This is the strongest persisted walk-forward experiment I found in the repository.
