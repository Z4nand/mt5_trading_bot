from datetime import datetime
from typing import Optional
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD
import pandas as pd
import os


#квартеты для загрузки данных батчами по несколько месяцев, тк есть ограничение по загрузке данных за раз(10 000)
QUARTERS = {
    1: (1, 1, 3, 31),    # Q1: январь-март
    2: (4, 1, 6, 30),    # Q2: апрель-июнь
    3: (7, 1, 9, 30),    # Q3: июль-сентябрь
    4: (10, 1, 12, 31),  # Q4: октябрь-декабрь
}

def fetch_m15(start:datetime, end: datetime) -> Optional[pd.DataFrame]:
    """
    Загружает 15-минутные данные EURUSD за указанный период
    """
    # start = datetime(2015, 1, 1)
    # end = datetime(2015, 1, 31)

    # Fetch 15-minute OHLCV data
    try:
        df = dukascopy_python.fetch(
            instrument=INSTRUMENT_FX_MAJORS_EUR_USD,
            interval=dukascopy_python.INTERVAL_MIN_15,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=start,
            end=end,
        )
        return df
    except Exception as e:
        print(f'Error! {e}')
        return None


def download_quarter(year: int , quarter: int, output_dir: str = "../data/raw/") -> Optional[pd.DataFrame]:
    """
    Загружает данные за один квартал
    
    Args:
        year: Год
        quarter: Номер квартала (1-4)
        output_dir: Директория для сохранения
    
    Returns:
        DataFrame с данными или None при ошибке
    """
    if quarter not in QUARTERS:
        print(f'Incorrect number of qurter:{quarter}')
        return None
    
    start_month, start_day, end_month, end_day = QUARTERS[quarter]
    start = datetime(year, start_month, start_day)
    end = datetime(year, end_month, end_day)

    print(f"Download: {year} Q{quarter}: {start.date()} - {end.date()}")

    df = fetch_m15(start, end)

    # Создаем директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка данных в файл
    if df is not None and len(df)>0:
        filename = f"EURUSD_15min_{year}_Q{quarter}.csv"
        filepath = f"{output_dir}{filename}"
        df.to_csv(filepath)
        print(f"Save {len(df)} candles . path: {filepath}")
        return df
    else:
        print('Nothing to save!')
        return None


def download_year_range(start_year: int, end_year: int) -> None:
    for year in range(start_year, end_year+1):
        for quarter in range(1, 5):
            download_quarter(year,quarter)
    print('Download success.')  


download_year_range(2015, 2026)