# Исследовательский отчет по диагностике TradeSkip GRU

Дата подготовки: 2026-05-14  
Инструмент: EURUSD  
Таймфрейм: 15 минут  
Период данных: 2014-12-31 18:00:00 UTC - 2026-03-26 16:30:00 UTC

## 1. Цель проверки

Цель исследования - определить, почему текущая модель TradeSkip GRU не дает устойчивого улучшения результатов торгового робота.

Проверялись следующие гипотезы:

1. Результаты ухудшаются из-за смены рыночных фаз между train/validation/test.
2. Threshold модели TradeSkip выбран неоптимально.
3. Модель устарела, и необходимо walk-forward переобучение.
4. Бинарная разметка `TRADE/SKIP` слишком грубая.
5. Первичный источник направления сделки `event_cusum_direction` может быть слабым или неверным.

## 2. Исходная схема стратегии

Текущая рабочая концепция:

```text
OHLCV -> feature engineering -> event detector -> event_cusum_direction -> TradeSkip GRU -> BUY / SELL / SKIP
```

Event detector выделяет значимые свечи. Direction задается правилом:

```text
event_cusum_direction =  1 -> BUY
event_cusum_direction = -1 -> SELL
```

TradeSkip GRU не выбирает направление. Модель решает только, стоит ли торговать событие с уже заданным направлением.

## 3. Данные и разбиение

Загружено:

```text
CSV files:        45
raw rows:         277105
prepared rows:    276922
events:           21684
event rate:       7.83%
```

Текущее фиксированное разбиение:

```text
train: 2015-2020
valid: 2021-2023
test:  2024-2026Q1
```

Количество обучающих событий TradeSkip:

```text
train samples: 11517
valid samples: 5856
test samples:  4306
```

## 4. Проверка фазового сдвига рынка

Режимы рынка были оценены через:

- `volatility_96`;
- `ma_50_diff`;
- train-квантили для low/mid/high volatility;
- train-порог абсолютного отклонения от MA50 для trend/range.

Доля режима `range + low_vol` среди событий:

```text
train: 25.9%
valid: 29.2%
test:  42.9%
```

Вывод: фазовый сдвиг подтвержден. Тестовый период содержит существенно больше низковолатильного боковика, чем train.

## 5. Диагностика сохраненной TradeSkip GRU

При текущем threshold `0.51`:

| Split | Base winrate | Selected winrate | Selected PnL |
|---|---:|---:|---:|
| train 2015-2020 | 46.42% | 48.79% | -1261 pips |
| valid 2021-2023 | 45.87% | 47.04% | -1678 pips |
| test 2024-2026Q1 | 47.98% | 49.49% | -42 pips |

Особенность модели: вероятности `prob_trade` почти всегда находятся около `0.50-0.52`. При threshold выше `0.55` торговля почти полностью выключается.

Вывод: сохраненная модель TradeSkip GRU ведет себя как слабый фильтр, а не как уверенный классификатор качества входа.

## 6. Threshold scan

На сохраненной модели лучший test-результат был около threshold `0.52-0.53`, но valid оставался отрицательным.

Пример:

| Threshold | Train PnL | Valid PnL | Test PnL |
|---:|---:|---:|---:|
| 0.51 | -1261 pips | -1678 pips | -42 pips |
| 0.52 | -122 pips | -931 pips | +230 pips |
| 0.53 | -188 pips | -498 pips | +256 pips |

Вывод: подбор threshold по test давал бы переоптимизацию. На validation устойчивого положительного результата нет.

## 7. Walk-forward threshold без переобучения

Была выполнена схема:

```text
прошлые 3 года -> выбор threshold -> следующий год test
```

Результат:

```text
total test PnL: -603 pips
positive folds: 5/9
median test PnL: +24 pips
```

Expanding-window:

```text
total test PnL: -545 pips
positive folds: 4/9
median test PnL: -16 pips
```

Вывод: проблема не сводится к выбору threshold.

## 8. Walk-forward retraining TradeSkip

Была выполнена схема:

```text
train years -> validation year для threshold -> test year
```

Короткий исследовательский режим:

```text
epochs: 3
patience: 2
train window: 3 года
test years: 2018-2026
```

Результат бинарного TradeSkip retraining:

```text
total test PnL: -278 pips
positive folds: 5/9
```

Переобучение улучшило результат относительно сохраненной модели:

```text
saved-model walk-forward: -603 pips
retraining walk-forward:  -278 pips
```

Но общий результат остался отрицательным.

Основные слабые годы:

```text
2019: -308 pips
2021: -200 pips
2023: -177 pips
```

Последние годы выглядели лучше:

```text
2025: +179 pips
2026: +200 pips
```

## 9. Проверка min-edge target

Проверялась гипотеза: `TRADE` следует считать не как `pnl > 0`, а как `pnl > min_edge`.

Результаты:

| Variant | Total test PnL | Positive folds |
|---|---:|---:|
| binary retrain, edge 0.0 | -278 pips | 5/9 |
| binary retrain, edge 0.5 pip | -801 pips | 2/9 |
| binary retrain, edge 1.0 pip | -557 pips | 4/9 |

Вывод: простое ужесточение бинарного target ухудшило результат.

## 10. Проверка 3-class target

Проверялась постановка:

```text
BAD      = pnl <= -band
NEUTRAL  = -band < pnl < +band
GOOD     = pnl >= +band
```

Торговать предполагалось только при высокой вероятности класса `GOOD`.

Результаты:

| Variant | Total test PnL | Positive folds |
|---|---:|---:|
| binary retrain, edge 0.0 | -278 pips | 5/9 |
| 3-class, band 1.0 pip | -381 pips | 5/9 |
| 3-class, band 2.0 pips | -451 pips | 4/9 |
| 3-class, band 0.5 pip | -652 pips | 4/9 |
| 3-class, band 0.0 pip | -745 pips | 4/9 |

Вывод: 3-class target не улучшил результат относительно бинарной постановки.

## 11. Текущее сравнение вариантов

Итоговый рейтинг исследованных вариантов:

| Rank | Variant | Total test PnL | Positive folds |
|---:|---|---:|---:|
| 1 | binary retrain, edge 0.0 | -278 pips | 5/9 |
| 2 | 3-class, band 1.0 pip | -381 pips | 5/9 |
| 3 | 3-class, band 2.0 pips | -451 pips | 4/9 |
| 4 | binary retrain, edge 1.0 pip | -557 pips | 4/9 |
| 5 | 3-class, band 0.5 pip | -652 pips | 4/9 |
| 6 | 3-class, band 0.0 pip | -745 pips | 4/9 |
| 7 | binary retrain, edge 0.5 pip | -801 pips | 2/9 |

Лучший из проверенных вариантов все еще отрицательный.

## 12. Главный вывод на текущем этапе

TradeSkip GRU не показывает устойчивого преимущества при текущем источнике направления сделки.

Наиболее вероятная причина: слабость или неверная универсальность правила `event_cusum_direction -> BUY/SELL`.

Так как TradeSkip не выбирает направление, а только фильтрует уже заданные BUY/SELL, слабое направление ограничивает максимальное качество модели. Если первичный rule baseline имеет отрицательное математическое ожидание, модель вынуждена фильтровать шум, а не усиливать устойчивый edge.

## 13. Следующая гипотеза

Следующая область исследования - замена или параметризация event direction.

Кандидаты:

1. `cusum_momentum`: текущее правило.
2. `cusum_reversal`: инверсия CUSUM-направления.
3. `breakout_momentum`: BUY при breakout_up, SELL при breakout_down.
4. `level_reversal`: BUY у поддержки, SELL у сопротивления.
5. `regime_mixed`: momentum в тренде, reversal в боковике.

Также следует проверить параметры CUSUM:

```text
CUSUM_VOLATILITY_WINDOW: 48, 96, 192
CUSUM_THRESHOLD_MULT:    1.8, 2.2, 2.8, 3.4, 4.0
```

Оценивать нужно не только общий PnL, но и устойчивость по годам:

```text
year / trades / winrate / avg pnl / total pnl / positive years
```

## 14. Файлы с результатами

Основные CSV-артефакты:

```text
data/processed/regime_diagnostics/
data/processed/walk_forward_diagnostics/
data/processed/walk_forward_retrain_full_3epoch/
data/processed/walk_forward_retrain_full_3epoch_edge05/
data/processed/walk_forward_retrain_full_3epoch_edge1/
data/processed/multiclass_retrain_band0/
data/processed/multiclass_retrain_band05/
data/processed/multiclass_retrain_band1/
data/processed/multiclass_retrain_band2/
data/processed/model_variant_comparison.csv
```

## 15. Диагностика альтернативного event direction

После проверки TradeSkip была исследована гипотеза, что слабым звеном является не фильтр `TRADE/SKIP`, а первичное направление сделки.

Был выполнен grid по CUSUM-параметрам:

```text
CUSUM_VOLATILITY_WINDOW: 48, 96, 192
CUSUM_THRESHOLD_MULT:    1.8, 2.2, 2.8, 3.4, 4.0
```

И по правилам направления:

```text
cusum_momentum      текущая логика: CUSUM up -> BUY, CUSUM down -> SELL
cusum_reversal      инверсия: CUSUM up -> SELL, CUSUM down -> BUY
breakout_momentum   breakout_up -> BUY, breakout_down -> SELL
breakout_reversal   breakout_up -> SELL, breakout_down -> BUY
level_reversal      near_support -> BUY, near_resistance -> SELL
regime_mixed        level reversal в боковике, momentum/breakout в остальных режимах
```

### 15.1. Текущая конфигурация против инверсии

Текущая конфигурация CUSUM:

```text
volatility_window = 96
threshold_mult    = 2.8
```

Сравнение:

| Rule | Train PnL | Valid PnL | Test PnL | Test trades | Test winrate | Test avg pnl |
|---|---:|---:|---:|---:|---:|---:|
| cusum_momentum | -6283 pips | -3559 pips | -1067 pips | 4306 | 47.98% | -0.248 pips |
| cusum_reversal | +843 pips | +1493 pips | -60 pips | 4306 | 50.33% | -0.014 pips |

Вывод: текущая логика направления `CUSUM up -> BUY` системно отрицательная. Простая инверсия CUSUM почти полностью убирает отрицательное математическое ожидание на test и дает сильный плюс на train/valid.

### 15.2. Лучшие варианты по всей сетке

Лучший вариант по общей истории:

```text
volatility_window = 192
threshold_mult    = 1.8
direction_rule    = cusum_reversal
trades            = 34805
winrate           = 50.96%
avg pnl           = +0.126 pips
total pnl         = +4374 pips
positive years    = 9/12
worst year        = -645 pips
```

Топовые варианты по train+valid без выбора по test:

| Vol window | Mult | Rule | Train+Valid PnL | Train PnL | Valid PnL | Test PnL | Test trades | Test winrate | Test avg pnl |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 192 | 1.8 | cusum_reversal | +3718 pips | +2435 pips | +1283 pips | +656 pips | 6938 | 50.52% | +0.095 pips |
| 48 | 1.8 | cusum_reversal | +3288 pips | +2147 pips | +1141 pips | +367 pips | 8397 | 50.30% | +0.044 pips |
| 48 | 2.2 | cusum_reversal | +3044 pips | +1675 pips | +1369 pips | +635 pips | 6731 | 50.71% | +0.094 pips |
| 48 | 2.8 | cusum_reversal | +2891 pips | +1250 pips | +1641 pips | +364 pips | 5095 | 50.74% | +0.071 pips |
| 96 | 1.8 | cusum_reversal | +2876 pips | +1943 pips | +933 pips | +324 pips | 7272 | 50.41% | +0.045 pips |

Лучший test-only вариант:

```text
volatility_window = 192
threshold_mult    = 1.8
direction_rule    = level_reversal
test pnl          = +1019 pips
test trades       = 2930
test winrate      = 51.84%
test avg pnl      = +0.348 pips
```

Однако его не следует выбирать как основной только по test. По train+valid он уступает `cusum_reversal 192/1.8`.

### 15.3. Главный вывод по direction

Гипотеза подтвердилась: направление сделки является более важной точкой улучшения, чем архитектура TradeSkip.

Текущий `cusum_momentum` торгует в системно неблагоприятную сторону. На EURUSD 15m выделенные CUSUM-события чаще ведут себя как mean-reversion, а не как momentum continuation.

Наиболее обоснованный следующий кандидат:

```text
CUSUM_VOLATILITY_WINDOW = 192
CUSUM_THRESHOLD_MULT    = 1.8
direction_rule          = cusum_reversal
```

Это правило выбрано по train+valid и остается положительным на test:

```text
train: +2435 pips
valid: +1283 pips
test:  +656 pips
```

Практический следующий шаг: встроить поддержку альтернативного `direction_rule` в исследовательский backtest и затем повторить TradeSkip retraining уже поверх `cusum_reversal 192/1.8`.

Дополнительные CSV-артефакты:

```text
data/processed/event_direction_diagnostics/by_config_rule.csv
data/processed/event_direction_diagnostics/by_year.csv
data/processed/event_direction_diagnostics/period_comparison.csv
data/processed/event_direction_diagnostics/direction_trades.csv
```

## 16. TradeSkip retraining поверх нового направления

После выбора кандидата `cusum_reversal 192/1.8` был повторен walk-forward retraining TradeSkip.

Схема:

```text
train window: 3 года
validation:   1 год для выбора threshold
test:         следующий год
epochs:       3
patience:     2
test years:   2018-2026
```

Результат TradeSkip поверх старого направления:

```text
direction_rule = cusum_momentum
window = 96
mult = 2.8

total test PnL: -278 pips
positive folds: 5/9
```

Результат TradeSkip поверх нового направления:

```text
direction_rule = cusum_reversal
window = 192
mult = 1.8

total test PnL: +3444 pips
positive folds: 7/9
```

Годовая таблица:

| Test year | Threshold | Trades | Winrate | PnL | Avg PnL |
|---:|---:|---:|---:|---:|---:|
| 2018 | 0.50 | 2029 | 52.19% | +616 pips | +0.304 |
| 2019 | 0.45 | 2637 | 53.55% | +1389 pips | +0.527 |
| 2020 | 0.45 | 2895 | 50.78% | +148 pips | +0.051 |
| 2021 | 0.50 | 1449 | 52.80% | +519 pips | +0.358 |
| 2022 | 0.45 | 2616 | 49.96% | -96 pips | -0.037 |
| 2023 | 0.45 | 2909 | 50.81% | +273 pips | +0.094 |
| 2024 | 0.45 | 3034 | 50.73% | +282 pips | +0.093 |
| 2025 | 0.48 | 838 | 52.86% | +393 pips | +0.468 |
| 2026 | 0.45 | 727 | 49.38% | -79 pips | -0.109 |

### 16.1. Сравнение с rule baseline нового направления

Важно: сам rule baseline `cusum_reversal 192/1.8` за 2018-2026 дал:

```text
total PnL: +4352 pips
positive years: 8/9
```

Годовая таблица rule baseline:

| Year | Trades | Winrate | PnL | Avg PnL |
|---:|---:|---:|---:|---:|
| 2018 | 3186 | 51.57% | +698 pips | +0.219 |
| 2019 | 3065 | 53.67% | +1670 pips | +0.545 |
| 2020 | 3227 | 50.57% | +45 pips | +0.014 |
| 2021 | 3242 | 51.48% | +662 pips | +0.204 |
| 2022 | 3086 | 50.39% | +139 pips | +0.045 |
| 2023 | 3167 | 51.09% | +482 pips | +0.152 |
| 2024 | 3040 | 50.69% | +267 pips | +0.088 |
| 2025 | 3122 | 50.70% | +498 pips | +0.159 |
| 2026 | 776 | 49.10% | -108 pips | -0.140 |

Вывод: замена направления является главным улучшением. TradeSkip поверх нового направления уже дает положительный результат, но пока уступает простому rule baseline. Это означает, что фильтр TradeSkip в текущем виде удаляет часть полезных сделок и не компенсирует это достаточным ростом качества.

Текущая лучшая найденная стратегия без учета комиссий и спреда:

```text
event detector:
  CUSUM_VOLATILITY_WINDOW = 192
  CUSUM_THRESHOLD_MULT    = 1.8

direction:
  cusum_reversal

filter:
  без TradeSkip или с осторожным дополнительным фильтром после новой диагностики
```

Дополнительные CSV-артефакты:

```text
data/processed/walk_forward_retrain_cusum_reversal_192_18/folds.csv
data/processed/walk_forward_retrain_cusum_reversal_192_18/fold_diagnostics.csv
data/processed/walk_forward_retrain_cusum_reversal_192_18/training_history.csv
```
