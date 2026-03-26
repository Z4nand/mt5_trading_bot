# скрипт выгрузки истории для обучения

import MetaTrader5 as mt5
from src.connector.mt5_client import MT5Client
import pandas as pd


client = MT5Client()

client.connect()

symbols = mt5.symbols_get()  # без аргументов
print("type(symbols):", type(symbols))
print("len(symbols):", len(symbols))

first = symbols[0]
print("type(first):", type(first))
print("dir(first):", dir(first))
print("repr(first):", first)
print("as dict:", first._asdict())


client.disconnect()