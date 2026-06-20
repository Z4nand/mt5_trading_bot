# Trade/Hold CUSUM Reversal Experiment

## Idea
CUSUM-reversal defines the trade direction. The neural model solves only TRADE vs HOLD.
Feature count: 34. Quality features: 14.

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
     hold_tp_before_sl_w96_m2.1     tp_before_sl            96         2.1          16056           8220          6105                      0.541227                0.533956                  0.55                    True       1.238245             30.4                 48
    hold_tp_before_sl_w192_m1.8     tp_before_sl           192         1.8          18355           9494          6938                      0.537337                0.533117                  0.55                   False       1.213018            205.2                354
      hold_tp_sl_only_w192_m1.8       tp_sl_only           192         1.8          17035           8861          6284                      0.510325                0.503631                  0.55                   False       1.188637            125.5                263
       hold_tp_sl_only_w96_m2.1       tp_sl_only            96         2.1          14884           7653          5501                      0.519555                0.501605                  0.52                    True       1.140745            376.9               1015
hold_positive_horizon_w288_m2.1 positive_horizon           288         2.1          15218           7827          5732                      0.507391                0.508953                  0.53                    True       1.130978            271.4                794
hold_positive_horizon_w192_m1.8 positive_horizon           192         1.8          18355           9494          6938                      0.510804                0.510835                  0.52                    True       1.109117            218.3                749
 hold_positive_horizon_w96_m2.1 positive_horizon            96         2.1          16056           8220          6105                      0.512512                0.507161                  0.53                    True       1.101185            205.7                804
    hold_tp_before_sl_w288_m2.1     tp_before_sl           288         2.1          15218           7827          5732                      0.544359                0.540656                  0.50                    True       0.998340            -15.4               3130
      hold_tp_sl_only_w288_m2.1       tp_sl_only           288         2.1          14183           7315          5187                      0.513118                0.504734                  0.50                   False       0.979138           -244.0               3944
```

## Best validation row per trained strategy
```
                       strategy segment  threshold  news_filter  trades  total_pnl_pips  winrate  profit_factor  max_drawdown  avg_trade_pips
     hold_tp_before_sl_w96_m2.1   valid       0.55         True      48            30.4 0.416667       1.238245     -0.003034        0.633333
    hold_tp_before_sl_w192_m1.8   valid       0.55        False     354           205.2 0.420904       1.213018     -0.009746        0.579661
      hold_tp_sl_only_w192_m1.8   valid       0.55        False     263           125.5 0.433460       1.188637     -0.007297        0.477186
       hold_tp_sl_only_w96_m2.1   valid       0.52         True    1015           376.9 0.413793       1.140745     -0.012641        0.371330
hold_positive_horizon_w288_m2.1   valid       0.53         True     794           271.4 0.406801       1.130978     -0.013298        0.341814
hold_positive_horizon_w192_m1.8   valid       0.52         True     749           218.3 0.404539       1.109117     -0.012272        0.291455
 hold_positive_horizon_w96_m2.1   valid       0.53         True     804           205.7 0.415423       1.101185     -0.013703        0.255846
    hold_tp_before_sl_w288_m2.1   valid       0.50         True    3130           -15.4 0.365176       0.998340     -0.029982       -0.004920
      hold_tp_sl_only_w288_m2.1   valid       0.50        False    3944          -244.0 0.364858       0.979138     -0.034241       -0.061866
```

## Test row for selected strategies
```
                       strategy segment  threshold  news_filter  trades  total_pnl_pips  winrate  profit_factor  max_drawdown  avg_trade_pips
    hold_tp_before_sl_w192_m1.8    test       0.57         True      14            38.6 0.571429       2.331034     -0.001473        2.757143
      hold_tp_sl_only_w288_m2.1    test       0.57         True      19            23.9 0.473684       1.507431     -0.001871        1.257895
hold_positive_horizon_w192_m1.8    test       0.52         True     434           264.3 0.421659       1.227982     -0.004930        0.608986
hold_positive_horizon_w288_m2.1    test       0.57         True     183            88.7 0.420765       1.187170     -0.004028        0.484699
      hold_tp_sl_only_w192_m1.8    test       0.55         True     110            49.6 0.400000       1.170564     -0.004048        0.450909
    hold_tp_before_sl_w288_m2.1    test       0.53         True     804           161.9 0.376866       1.069637     -0.023542        0.201368
     hold_tp_before_sl_w96_m2.1    test       0.53         True     232            39.3 0.370690       1.057871     -0.010116        0.169397
 hold_positive_horizon_w96_m2.1    test       0.55         True     281            39.2 0.434164       1.057076     -0.008609        0.139502
       hold_tp_sl_only_w96_m2.1    test       0.50         True    1830            52.0 0.385792       1.010276     -0.021845        0.028415
```

## Selected model
- Model: `C:\Projects\robot\data\models\trade_hold_reversal_quality_gru_best.pth`
- Scaler: `C:\Projects\robot\data\models\trade_hold_reversal_quality_gru_scaler.pkl`
- Config: `C:\Projects\robot\data\models\trade_hold_reversal_quality_gru_config.pkl`
- Label mode: tp_before_sl
- CUSUM window: 192
- CUSUM multiplier: 1.8

## Notes
- Trading metrics use TP 8 pips, SL 4 pips, horizon 8 candles, cost 0.3 pip.
- News filter is evaluated as a separate entry filter, not as a training feature.
