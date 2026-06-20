# TradeSkip News Model Experiment

## Data split
- Train: before 2021-01-01
- Validation: 2021-01-01 to 2024-01-01
- Test: from 2024-01-01
- News leakage policy: used only event schedule metadata: time, currency, impact and event-name group. Excluded actual, forecast and previous values.

## Candidate feature sets
- retrained_no_news_sl4: features=20, best_valid_balanced_accuracy=0.5335
- news_core: features=35, best_valid_balanced_accuracy=0.5339
- news_full: features=47, best_valid_balanced_accuracy=0.5268

## Selected model
- Feature set: retrained_no_news_sl4
- Saved model: `C:\Projects\robot\data\models\trade_skip_reversal_sl4_gru_best.pth`
- Saved scaler: `C:\Projects\robot\data\models\trade_skip_reversal_sl4_gru_scaler.pkl`
- Saved config: `C:\Projects\robot\data\models\trade_skip_reversal_sl4_gru_config.pkl`
- Input features: 20

## Final saved model classification on test
```
          strategy  accuracy  balanced_accuracy       f1  precision   recall  count  avg_confidence
  old_reversal_gru   0.49755           0.497152 0.518508   0.502544 0.535521   6938        0.514426
final_selected_gru   0.50173           0.526500 0.476609   0.387398 0.619197   6938        0.531534
```

## Final saved model trading comparison
```
          strategy  threshold  news_filter  trades  total_pnl_pips  winrate  profit_factor  max_drawdown  avg_trade_pips
  old_reversal_gru       0.53         True     688           252.5 0.441860       1.153123     -0.009103        0.367006
final_selected_gru       0.51         True    2372           487.8 0.409781       1.079903     -0.011959        0.205649
```

## Exploratory classification on test
```
             strategy  accuracy  balanced_accuracy       f1  precision   recall  count  avg_confidence
     old_reversal_gru  0.497550           0.497152 0.518508   0.502544 0.535521   6938        0.514426
retrained_no_news_sl4  0.572932           0.536067 0.405855   0.413906 0.398112   6938        0.533583
            news_core  0.537907           0.526678 0.434568   0.393862 0.484658   6938        0.530555
            news_full  0.477948           0.519347 0.486241   0.380213 0.674272   6938        0.556825
```

## Exploratory best trading rows by strategy
```
             strategy  threshold  news_filter  trades  total_pnl_pips  winrate  profit_factor  max_drawdown  avg_trade_pips
     old_reversal_gru       0.53         True     688           252.5 0.441860       1.153123     -0.009103        0.367006
retrained_no_news_sl4       0.50         True    1966           404.0 0.415565       1.081606     -0.008456        0.205493
            news_core       0.52         True    1383           258.0 0.423717       1.078007     -0.018648        0.186551
            news_full       0.52         True    2520           449.6 0.405556       1.068384     -0.018355        0.178413
```

## Notes
- Trading comparison uses TP 8 pips, SL 4 pips, horizon 8 candles and cost 0.3 pip per trade.
- The old model is not overwritten; it is evaluated as a baseline on the same test window.
