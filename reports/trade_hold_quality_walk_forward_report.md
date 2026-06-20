# Trade/Hold Quality Walk-Forward

## Setup
- Architecture: CUSUM reversal gives direction, GRU predicts TRADE/HOLD.
- Features: base TradeSkip features + reversal quality features.
- Label: positive_horizon.
- Validation selects threshold/news filter; test year is unseen for that fold.
- Cost per trade: 0.30 pips, TP: 8.0 pips, SL: 4.0 pips.

## Summary
```
                 candidate  folds  positive_folds  total_test_pips  median_test_pips  worst_test_pips  total_test_trades  avg_test_pf  median_test_pf  worst_test_pf  max_abs_drawdown
positive_horizon_w288_m2.1      6               4            118.1             25.15            -75.5               1905     1.019120        1.024939       0.722834         -0.019006
positive_horizon_w192_m1.8      6               3             86.0             -3.40            -62.5               2041     1.076284        1.022833       0.952210         -0.021171
```

## Fold Results
```
                 candidate         fold  train_end           valid_window            test_window  selected_threshold  selected_news_filter  valid_trades  valid_total_pnl_pips  valid_profit_factor  test_trades  test_total_pnl_pips  test_winrate  test_profit_factor  test_max_drawdown  test_avg_trade_pips  train_samples  valid_samples  test_samples  test_balanced_accuracy
positive_horizon_w192_m1.8         2021 2020-01-01 2020-01-01..2021-01-01 2021-01-01..2022-01-01                0.53                 False           594                 126.6             1.075537          535                -62.5      0.386916            0.958936          -0.021171            -0.116822          15129           3225          3241                0.510167
positive_horizon_w192_m1.8         2022 2021-01-01 2021-01-01..2022-01-01 2022-01-01..2023-01-01                0.57                  True            91                  97.1             1.466603          193                 37.9      0.388601            1.073180          -0.008657             0.196373          18355           3241          3084                0.527242
positive_horizon_w192_m1.8         2023 2022-01-01 2022-01-01..2023-01-01 2023-01-01..2024-01-01                0.50                  True           697                 155.6             1.082987          615                -44.7      0.393496            0.972486          -0.020173            -0.072683          21598           3084          3166                0.502776
positive_horizon_w192_m1.8         2024 2023-01-01 2023-01-01..2024-01-01 2024-01-01..2025-01-01                0.53                 False           170                  26.8             1.066222          156                 66.2      0.474359            1.216199          -0.005748             0.424359          24683           3166          3040                0.510841
positive_horizon_w192_m1.8         2025 2024-01-01 2024-01-01..2025-01-01 2025-01-01..2026-01-01                0.53                  True           141                 105.8             1.389974          190                139.7      0.442105            1.284695          -0.006153             0.735263          27851           3040          3122                0.510479
positive_horizon_w192_m1.8 2026_partial 2025-01-01 2025-01-01..2026-01-01 2026-01-01..2027-01-01                0.51                 False          1275                 456.6             1.127332          352                -50.6      0.363636            0.952210          -0.012114            -0.143750          30892           3122           775                0.495288
positive_horizon_w288_m2.1         2021 2020-01-01 2020-01-01..2021-01-01 2021-01-01..2022-01-01                0.57                  True           116                  98.1             1.337113           85                 72.9      0.447059            1.358407          -0.005588             0.857647          12486           2731          2685                0.495932
positive_horizon_w288_m2.1         2022 2021-01-01 2021-01-01..2022-01-01 2022-01-01..2023-01-01                0.55                  True           199                  28.8             1.057901          217                -47.8      0.345622            0.923727          -0.012245            -0.220276          15218           2685          2516                0.517302
positive_horizon_w288_m2.1         2023 2022-01-01 2022-01-01..2023-01-01 2023-01-01..2024-01-01                0.50                  True           777                 206.2             1.097836          747                118.2      0.397590            1.059875          -0.019006             0.158233          17904           2516          2623                0.490874
positive_horizon_w288_m2.1         2024 2023-01-01 2023-01-01..2024-01-01 2024-01-01..2025-01-01                0.52                  True           485                 172.0             1.140523          414                  3.2      0.420290            1.003355          -0.009989             0.007729          20421           2623          2522                0.510260
positive_horizon_w288_m2.1         2025 2024-01-01 2024-01-01..2025-01-01 2025-01-01..2026-01-01                0.51                  True           258                 110.2             1.183851          359                 47.1      0.392758            1.046523          -0.008688             0.131198          23046           2522          2561                0.498754
positive_horizon_w288_m2.1 2026_partial 2025-01-01 2025-01-01..2026-01-01 2026-01-01..2027-01-01                0.51                  True           299                 205.4             1.262425           83                -75.5      0.301205            0.722834          -0.009827            -0.909639          25569           2561           647                0.494481
```
