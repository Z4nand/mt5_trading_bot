from pathlib import Path

import pandas as pd

from src.config import DATA_DIR


DATETIME_CANDIDATES = ("timestamp", "datetime", "date", "time", "Date", "Time", "Gmt time")
COLUMN_ALIASES = {
    "open": ["open", "Open", "OPEN"],
    "high": ["high", "High", "HIGH"],
    "low": ["low", "Low", "LOW"],
    "close": ["close", "Close", "CLOSE", "bidclose"],
    "volume": ["volume", "Volume", "tick_volume", "Volume BTC", "vol"],
}


def list_csv_files(data_dir: Path = DATA_DIR) -> list[Path]:
    search_dir = data_dir
    if data_dir.resolve() == DATA_DIR.resolve() and (DATA_DIR / "raw").exists():
        search_dir = DATA_DIR / "raw"

    if not search_dir.exists():
        return []
    return sorted(search_dir.rglob("*.csv"))


def _find_column(columns: list[str], candidates: tuple[str, ...] | list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def load_price_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV-файл пустой: {csv_path.name}")

    datetime_col = _find_column(list(df.columns), DATETIME_CANDIDATES)
    if datetime_col is None:
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().mean() < 0.8:
            raise ValueError("Не найдена колонка даты/времени. Ожидается timestamp/datetime/date/time.")
        datetime_col = first_col

    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        source = _find_column(list(df.columns), aliases)
        if source:
            rename_map[source] = target

    df = df.rename(columns=rename_map)
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"В CSV нет обязательных колонок: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["timestamp"] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def load_all_price_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, list[str]]:
    files = list_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"В папке {data_dir} нет CSV-файлов.")

    frames = []
    loaded = []
    errors = []
    for file in files:
        try:
            frames.append(load_price_data(file))
            loaded.append(file.name)
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

    if not frames:
        raise ValueError("Не удалось загрузить ни один CSV. " + "; ".join(errors))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, loaded
