# EURUSD Event-Driven AI Trading Robot

Проект торгового робота для EURUSD с двухэтапной event-driven архитектурой. Робот не пытается прогнозировать каждую свечу подряд: сначала выделяются значимые рыночные события, затем модель оценивает, стоит ли входить в сделку по найденному направлению.

```text
OHLC data -> Feature Engineering -> Event Detector -> Direction Rule -> TradeSkip GRU -> BUY / SELL / SKIP
```

Текущая рабочая логика:

- `Event Detector` выделяет потенциально информативные участки рынка;
- `event_cusum_direction` и rule-логика задают направление сделки;
- `TradeSkip GRU Reversal` фильтрует события и решает, входить или пропускать;
- бэктест проверяет сделки по `TP / SL / horizon`, с учетом комиссий и риск-менеджмента.

## Запуск приложения

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

После запуска приложение доступно по адресу:

```text
http://localhost:8501
```

## Основные страницы

| Страница | Назначение |
|---|---|
| `Обзор` | Краткое состояние робота, капитал, доходность, последние события |
| `Статистика` | Метрики стратегии, equity curve, зоны TP/SL, периодическая статистика |
| `Сделки` | Таблица сделок с ценой входа/выхода, PnL, причиной выхода |
| `Позиции` | Заготовка под текущие открытые позиции |
| `Настройки` | Сохранение параметров робота, SL/TP, комиссий и риска |
| `Журналы` | Системные события приложения |
| `Уведомления` | Сигнальные уведомления |
| `Backtest` | Сравнение стратегий, графики, активность сделок, confidence analysis |

Отдельная страница `Модель` удалена: диагностика модели и торговая статистика теперь сосредоточены в `Статистика` и `Backtest`.

## Пользовательский конфиг

Пользовательские параметры сохраняются в:

```text
config/user_settings/settings.json
```

В файл сохраняются:

- выбранная стратегия;
- тип модели;
- threshold confidence;
- horizon;
- параметры фиксированного `TP / SL`;
- фильтр новостей;
- комиссии и издержки;
- начальный баланс;
- риск на сделку;
- стоимость пункта за 1 лот.

При следующем запуске Streamlit подтягивает эти значения автоматически.

## Текущие параметры по умолчанию

```text
Strategy: TradeSkip GRU Reversal
Model: gru
Threshold confidence: 0.51
Horizon candles: 8
TP from entry: 8 pips
SL from entry: 4 pips
Initial balance: $1000
Risk per trade: 1%
Costs: enabled
Spread: 0.2 pips
Slippage: 0.1 pips
Commission: 0.0 pips
News filter: enabled
Skip before news: 60 minutes
```

## Расчет сделок

Сделка открывается по цене закрытия сигнальной свечи. Выход моделируется одним из трех способов:

- `TP` - цена достигла take profit;
- `SL` - цена достигла stop loss;
- `HORIZON` - TP/SL не достигнуты, выход по закрытию свечи через заданный горизонт.

SL/TP считаются как фиксированное расстояние от точки входа. Для EURUSD:

```text
8 pips = 0.0008
4 pips = 0.0004
```

High/low свечи учитываются, поэтому срабатывание TP/SL проверяется по теням свечи. Если в одной свече одновременно задеты TP и SL, используется консервативный вариант: сначала считается срабатывание SL.

## Комиссии и PnL

В бэктесте используется разделение:

```text
Gross PnL = результат движения цены без издержек
Cost = spread + slippage + commission
Net PnL = Gross PnL - Cost
```

Именно `Net PnL` используется в торговых метриках, equity curve и риск-менеджменте.

## Риск-менеджмент

Размер позиции рассчитывается от текущего капитала:

```text
risk_amount = equity * risk_per_trade_pct
lot_size = risk_amount / (SL_pips * pip_value_per_lot)
PnL_money = PnL_pips * pip_value_per_lot * lot_size
```

Это позволяет моделировать динамический размер позиции, а не фиксированный лот.

## Основные стратегии

| Стратегия | Идея |
|---|---|
| `TradeSkip GRU Reversal` | Основная стратегия: CUSUM/reversal направление + GRU-фильтр TRADE/SKIP |
| `TradeSkip GRU` | Event direction + GRU-фильтр без reversal-правила |
| `Rule CUSUM Reversal` | Правило без нейросети, вход на откат после CUSUM-события |
| `Event + GRU` | Event detector + GRU прогноз направления UP/DOWN |
| `Event + LSTM` | Event detector + LSTM прогноз направления UP/DOWN |
| `Rule baseline` | Базовая rule-стратегия без ML-фильтра |

Основной вариант для дальнейшей разработки: **TradeSkip GRU Reversal**.

## Метрики

Торговые метрики:

- Total Return;
- Total PnL;
- Final Balance;
- Win Rate;
- Profit Factor;
- Max Drawdown;
- Average Trade;
- Average Lot Size;
- количество сделок.

ML-метрики:

- Accuracy;
- Precision;
- Recall;
- F1 TRADE;
- F1 HOLD;
- F1 macro;
- Balanced accuracy;
- Actual profitable rate;
- Predicted trade rate.

Для TradeSkip метрики считаются по задаче `TRADE / HOLD`, а не по направлению `UP / DOWN`.

## Backtest-графики

На странице `Backtest` доступны:

- сравнение стратегий;
- кривая капитала;
- зоны входа с TP/SL-прямоугольниками;
- статистика по неделям, месяцам и годам;
- активность сделок по часам UTC;
- PnL по дням недели;
- распределение причин выхода `TP / SL / HORIZON`;
- связь confidence модели с результатом сделки;
- confidence analysis по разным порогам.

## Данные и модели

Исходные данные:

```text
data/
```

Экономический календарь:

```text
data/economic_calendar/
```

Сохраненные модели и scaler:

```text
data/models/
```

Ключевые файлы основной модели:

```text
trade_skip_reversal_gru_best.pth
trade_skip_reversal_gru_scaler.pkl
trade_skip_reversal_gru_config.pkl
```

## Разделение выборок

В проекте используется временное разделение:

```text
train: до 2021-01-01
validation: 2021-01-01 ... 2024-01-01
test: с 2024-01-01
```

Scaler обучается только на train-части. Для train и validation исключаются примеры, у которых будущий `horizon` пересекает границу следующего периода. Это снижает риск утечки будущей информации.

## Проверка проекта

Запуск тестов:

```powershell
python -m pytest tests -q
```

Быстрая проверка синтаксиса:

```powershell
python -m py_compile src\ui\streamlit_app.py src\strategy\backtest.py src\strategy\signal_generator.py
```

## Структура проекта

```text
robot/
├── app.py
├── README.md
├── config/
│   ├── settings.yaml
│   └── user_settings/
│       └── settings.json
├── data/                  # local only, ignored by Git
├── notebooks/
├── reports/
├── scripts/
├── src/
│   ├── config.py
│   ├── connector/
│   ├── features/
│   ├── models/
│   ├── strategy/
│   ├── ui/
│   └── user_settings.py
└── tests/
```

## Цель исследования

Проверить, повышает ли двухэтапный подход качество торговых решений по сравнению с прямым прогнозированием на всех свечах. Ключевая идея: сначала убрать большую часть рыночного шума через event detection, затем применять нейросеть только к наиболее информативным ситуациям.
