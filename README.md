# mt5_trading_bot
___
01.01.2015 | 31.03.2015 | янв- март
01.04.2015 | 30.06.2015 | апр -июнь
01.07.2015 | 30.09.2015 | июль -сент
01.10.2015 | 31.12.2015 | окт - дек
___
Планируемая архитектура:
mt5_trading_bot/
├── .env                      # API-ключи, логин/пароль MT5 (не в git!)
├── .gitignore
├── requirements.txt
├── config/
│   └── settings.yaml         # параметры: инструменты, таймфреймы, риски
│
├── src/
│   ├── __init__.py
│   │
│   ├── connector/            # Модуль связи с MT5
│   │   ├── __init__.py
│   │   ├── mt5_client.py     # init/shutdown, login, проверка соединения
│   │   ├── data_fetcher.py   # получение свечей, тиков, стакана
│   │   └── order_manager.py  # отправка ордеров, закрытие позиций
│   │
│   ├── features/             # Модуль генерации фичей
│   │   ├── __init__.py
│   │   ├── indicators.py     # RSI, SMA, ATR и т.п.
│   │   └── feature_pipeline.py  # сборка фичей в DataFrame
│   │
│   ├── models/               # ML-модели
│   │   ├── __init__.py
│   │   ├── base_model.py     # абстрактный класс модели
│   │   ├── lstm_model.py     # пример: LSTM/GRU на PyTorch
│   │   └── sklearn_model.py  # пример: градиентный бустинг
│   │
│   ├── strategy/             # Торговая логика
│   │   ├── __init__.py
│   │   ├── signal_generator.py  # модель → сигнал (BUY/SELL/HOLD)
│   │   └── risk_manager.py      # размер позиции, стоп-лосс, лимиты
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py         # логирование в файл и консоль
│       └── helpers.py        # вспомогательные функции
│
├── notebooks/                # Jupyter для исследований
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── data/
│   ├── raw/                  # сырые данные из MT5
│   ├── processed/            # фичи для обучения
│   └── models/               # сохранённые веса моделей (.pt, .pkl)
│
├── logs/                     # логи работы бота
│   └── bot.log
│
├── tests/
│   ├── __init__.py
│   ├── test_connector.py
│   ├── test_features.py
│   └── test_strategy.py
│
├── scripts/
│   ├── download_history.py   # скрипт выгрузки истории для обучения
│   └── backtest.py           # оффлайн-бэктест на исторических данных
│
└── main.py                   # точка входа: запуск торгового цикла





connector/	Всё взаимодействие с MT5: подключение, данные, ордера
features/	Расчёт технических индикаторов и сборка признаков
models/	Загрузка/инференс ML-моделей (обучение — в notebooks)
strategy/	Преобразование предсказаний в торговые сигналы + риск-менеджмент
utils/	Логирование, конфиги, хелперы

python -m tests.test_connect