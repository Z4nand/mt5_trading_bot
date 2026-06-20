# Trade/Hold CUSUM Reversal Experiment

## Idea
CUSUM-reversal defines the trade direction. The neural model solves only TRADE vs HOLD.

## Data split
- Train: before 2021-01-01
- Validation: 2021-01-01 to 2024-01-01
- Test: from 2024-01-01

## Label modes
- positive_horizon: TRADE if CUSUM-reversal position is profitable at horizon close.
- tp_before_sl: TRADE if TP is hit first; HOLD for SL or horizon.
- tp_sl_only: TRADE if TP is hit first, HOLD if SL is hit first, ignore horizon-only cases.

## Top rule-only CUSUM settings on validation
```
      strategy segment  threshold  news_filter  trades  total_pnl_pips  winrate  profit_factor  max_drawdown  avg_trade_pips  cusum_window  cusum_mult
rule_w288_m2.1   valid        0.5         True    4929          -707.3 0.365389       0.950728     -0.083483       -0.143498           288         2.1
 rule_w96_m2.1   valid        0.5         True    5263          -795.9 0.365951       0.947909     -0.093732       -0.151226            96         2.1
rule_w192_m1.8   valid        0.5         True    6023         -1008.9 0.364602       0.942629     -0.123054       -0.167508           192         1.8
```

## Candidate training summary
```
                       strategy       label_mode  cusum_window  cusum_mult  train_samples  valid_samples  test_samples  best_valid_balanced_accuracy  test_balanced_accuracy  valid_best_threshold  valid_best_news_filter  valid_best_pf  valid_best_pips  valid_best_trades
     hold_tp_before_sl_w96_m2.1     tp_before_sl            96         2.1          16056           8220          6105                      0.532544                0.534594                  0.57                    True       5.397727             38.7                  9
      hold_tp_sl_only_w192_m1.8       tp_sl_only           192         1.8          17035           8861          6284                      0.510553                0.492682                  0.57                   False       1.717143            175.7                112
      hold_tp_sl_only_w288_m2.1       tp_sl_only           288         2.1          14183           7315          5187                      0.519839                0.512783                  0.57                    True       1.530249             29.8                 26
 hold_positive_horizon_w96_m2.1 positive_horizon            96         2.1          16056           8220          6105                      0.505033                0.506587                  0.57                    True       1.353812             47.8                 54
hold_positive_horizon_w192_m1.8 positive_horizon           192         1.8          18355           9494          6938                      0.504336                0.499115                  0.57                   False       1.352143             64.9                 74
       hold_tp_sl_only_w96_m2.1       tp_sl_only            96         2.1          14884           7653          5501                      0.516055                0.500954                  0.55                    True       1.284141            309.6                430
    hold_tp_before_sl_w288_m2.1     tp_before_sl           288         2.1          15218           7827          5732                      0.536232                0.538618                  0.55                    True       1.095962             22.1                 85
hold_positive_horizon_w288_m2.1 positive_horizon           288         2.1          15218           7827          5732                      0.507467                0.515769                  0.51                    True       1.055017            194.1               1304
    hold_tp_before_sl_w192_m1.8     tp_before_sl           192         1.8          18355           9494          6938                      0.530839                0.540015                  0.53                    True       0.967352           -213.0               2210
```

## Best validation row per trained strategy
```
                       strategy segment  threshold  news_filter  trades  total_pnl_pips  winrate  profit_factor  max_drawdown  avg_trade_pips
     hold_tp_before_sl_w96_m2.1   valid       0.57         True       9            38.7 0.777778       5.397727     -0.000879        4.300000
      hold_tp_sl_only_w192_m1.8   valid       0.57        False     112           175.7 0.517857       1.717143     -0.002727        1.568750
      hold_tp_sl_only_w288_m2.1   valid       0.57         True      26            29.8 0.461538       1.530249     -0.001940        1.146154
 hold_positive_horizon_w96_m2.1   valid       0.57         True      54            47.8 0.425926       1.353812     -0.003171        0.885185
hold_positive_horizon_w192_m1.8   valid       0.57        False      74            64.9 0.445946       1.352143     -0.003440        0.877027
       hold_tp_sl_only_w96_m2.1   valid       0.55         True     430           309.6 0.437209       1.284141     -0.006090        0.720000
    hold_tp_before_sl_w288_m2.1   valid       0.55         True      85            22.1 0.388235       1.095962     -0.004348        0.260000
hold_positive_horizon_w288_m2.1   valid       0.51         True    1304           194.1 0.388804       1.055017     -0.016907        0.148850
    hold_tp_before_sl_w192_m1.8   valid       0.53         True    2210          -213.0 0.358371       0.967352     -0.034628       -0.096380
```

## Test row for selected strategies
```
                       strategy segment  threshold  news_filter  trades  total_pnl_pips  winrate  profit_factor  max_drawdown  avg_trade_pips
     hold_tp_before_sl_w96_m2.1    test       0.57        False       4            22.2 0.750000       5.826087      0.000000        5.550000
      hold_tp_sl_only_w288_m2.1    test       0.57         True      12            33.7 0.583333       2.660099     -0.000997        2.808333
    hold_tp_before_sl_w288_m2.1    test       0.57         True      15            35.6 0.533333       2.075529     -0.000489        2.373333
 hold_positive_horizon_w96_m2.1    test       0.55         True      51            33.1 0.490196       1.269544     -0.002786        0.649020
hold_positive_horizon_w288_m2.1    test       0.57         True     132            91.1 0.431818       1.264672     -0.005562        0.690152
      hold_tp_sl_only_w192_m1.8    test       0.55         True     136            78.0 0.426471       1.212072     -0.004830        0.573529
    hold_tp_before_sl_w192_m1.8    test       0.55         True     237            67.1 0.379747       1.097289     -0.007938        0.283122
hold_positive_horizon_w192_m1.8    test       0.50         True    2479           153.0 0.386043       1.022491     -0.014104        0.061718
       hold_tp_sl_only_w96_m2.1    test       0.52         True     932            22.2 0.378755       1.008543     -0.020146        0.023820
```

## Selected model
- Model: `C:\Projects\robot\data\models\trade_hold_reversal_gru_best.pth`
- Scaler: `C:\Projects\robot\data\models\trade_hold_reversal_gru_scaler.pkl`
- Config: `C:\Projects\robot\data\models\trade_hold_reversal_gru_config.pkl`
- Label mode: tp_sl_only
- CUSUM window: 192
- CUSUM multiplier: 1.8

## Notes
- Trading metrics use TP 8 pips, SL 4 pips, horizon 8 candles, cost 0.3 pip.
- News filter is evaluated as a separate entry filter, not as a training feature.
