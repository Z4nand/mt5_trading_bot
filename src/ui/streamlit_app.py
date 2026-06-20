import traceback

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ModuleNotFoundError:
    go = None
    HAS_PLOTLY = False

from src import config
from src.connector.data_fetcher import load_all_price_data
from src.strategy.backtest import build_trades, calculate_equity_curve, calculate_risk_equity_curve, calculate_risk_trade_metrics, calculate_trade_metrics, apply_risk_sizing
from src.models.sequence_models import load_model
from src.models.sequence_models import load_model_with_config
from src.models.trade_skip_training import generate_trade_skip_signal_history
from src.strategy.signal_generator import add_labels_for_metrics, generate_rule_based_signal_history, generate_signal_history, load_scaler, prepare_rule_frame
from src.user_settings import DEFAULT_USER_SETTINGS, USER_SETTINGS_PATH, load_user_settings, save_user_settings


st.set_page_config(page_title="EURUSD AI Trading Robot", page_icon="📈", layout="wide")

TRADE_SKIP_DEFAULT_THRESHOLD = 0.51
FALLBACK_Q_CANDLES = 2000
NEWS_CALENDAR_PATH = config.DATA_DIR / "economic_calendar" / "economic_calendar.csv"
NEWS_FILTER_DEFAULT_MINUTES = 60
NEWS_FILTER_IMPACTS = {"medium", "high"}
DEFAULT_SPREAD_PIPS = 0.2
DEFAULT_SLIPPAGE_PIPS = 0.1
DEFAULT_COMMISSION_PIPS = 0.0
DEFAULT_INITIAL_BALANCE = 1_000.0
DEFAULT_RISK_PER_TRADE_PCT = 1.0
DEFAULT_PIP_VALUE_PER_LOT = 10.0
DEFAULT_TP_PIPS = 8.0
DEFAULT_SL_PIPS = 4.0
CONTROL_DEFAULTS_VERSION = 2

st.markdown(
    """
    <style>
    :root {
        --border: #d0d7de;
        --muted: #57606a;
        --accent: #0969da;
        --success: #1a7f37;
        --danger: #cf222e;
        --bg: #f6f8fa;
    }
    .stApp { background: var(--bg); color: #24292f; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--border); }
    h1, h2, h3 { color: #24292f; letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 16px;
    }
    .card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-ok { color: var(--success); font-weight: 700; }
    .status-bad { color: var(--danger); font-weight: 700; }
    .small-muted { color: var(--muted); font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_load_data():
    return load_all_price_data(config.DATA_DIR)


@st.cache_data(show_spinner=False)
def cached_prepare_data():
    price_df, _ = cached_load_data()
    return prepare_rule_frame(price_df)


@st.cache_data(show_spinner=False)
def cached_news_calendar():
    if not NEWS_CALENDAR_PATH.exists():
        return pd.DataFrame()
    news = pd.read_csv(NEWS_CALENDAR_PATH)
    if news.empty or "time_utc" not in news:
        return pd.DataFrame()
    news = news.copy()
    news["time_utc"] = pd.to_datetime(news["time_utc"], utc=True, errors="coerce")
    news["impact"] = news.get("impact", "").astype(str).str.lower()
    news["currency"] = news.get("currency", "").astype(str).str.upper()
    news = news.dropna(subset=["time_utc"])
    news = news[news["currency"].isin(["EUR", "USD"])]
    news = news[news["impact"].isin(NEWS_FILTER_IMPACTS)]
    news = news.drop_duplicates(subset=["time_utc", "country", "impact", "actual", "forecast", "previous", "unit"])
    return news.sort_values("time_utc").reset_index(drop=True)


def total_cost_threshold() -> float:
    if not st.session_state.get("include_costs", False):
        return 0.0
    total_pips = (
        float(st.session_state.get("spread_pips", DEFAULT_SPREAD_PIPS))
        + float(st.session_state.get("slippage_pips", DEFAULT_SLIPPAGE_PIPS))
        + float(st.session_state.get("commission_pips", DEFAULT_COMMISSION_PIPS))
    )
    return max(total_pips, 0.0) / 10000.0


@st.cache_resource(show_spinner=False)
def cached_model_resource(model_type: str, dataset_mode: str = "event"):
    if dataset_mode == "event":
        model_path = config.EVENT_MODEL_PATHS[model_type]
        config_path = config.EVENT_CONFIG_PATHS[model_type]
    else:
        model_path = config.FULL_MODEL_PATHS[model_type]
        config_path = config.FULL_CONFIG_PATHS[model_type]

    return load_model_with_config(model_path, config_path)


@st.cache_resource(show_spinner=False)
def cached_scaler_resource(model_type: str, dataset_mode: str = "event"):
    scaler_path = config.EVENT_SCALER_PATHS[model_type] if dataset_mode == "event" else config.FULL_SCALER_PATHS[model_type]
    return load_scaler(scaler_path)


def trade_skip_paths(model_type: str = "gru"):
    model_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_best.pth"
    scaler_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_scaler.pkl"
    config_path = config.MODELS_DIR / f"trade_skip_event_{model_type}_config.pkl"
    return model_path, scaler_path, config_path


def trade_skip_reversal_paths():
    return (
        config.TRADE_SKIP_REVERSAL_GRU_MODEL_PATH,
        config.TRADE_SKIP_REVERSAL_GRU_SCALER_PATH,
        config.TRADE_SKIP_REVERSAL_GRU_CONFIG_PATH,
    )


@st.cache_resource(show_spinner=False)
def cached_trade_skip_model_resource(model_type: str = "gru"):
    model_path, _, config_path = trade_skip_paths(model_type)
    return load_model_with_config(model_path, config_path)


@st.cache_resource(show_spinner=False)
def cached_trade_skip_scaler_resource(model_type: str = "gru"):
    _, scaler_path, _ = trade_skip_paths(model_type)
    return load_scaler(scaler_path)


@st.cache_data(show_spinner=False)
def cached_trade_skip_config(model_type: str = "gru"):
    _, _, config_path = trade_skip_paths(model_type)
    return joblib.load(config_path)


@st.cache_resource(show_spinner=False)
def cached_trade_skip_reversal_model_resource():
    model_path, _, config_path = trade_skip_reversal_paths()
    return load_model_with_config(model_path, config_path)


@st.cache_resource(show_spinner=False)
def cached_trade_skip_reversal_scaler_resource():
    _, scaler_path, _ = trade_skip_reversal_paths()
    return load_scaler(scaler_path)


@st.cache_data(show_spinner=False)
def cached_trade_skip_reversal_config():
    _, _, config_path = trade_skip_reversal_paths()
    return joblib.load(config_path)


@st.cache_data(show_spinner=False)
def cached_signal_history(max_rows: int, model_type: str, require_event: bool, _model, _scaler):
    price_df, _ = cached_load_data()
    return generate_signal_history(
        price_df,
        threshold=0.50,
        max_rows=max_rows,
        model_type=model_type,
        require_event=require_event,
        model=_model,
        scaler=_scaler,
    )


@st.cache_data(show_spinner=False)
def cached_trade_skip_signal_history(max_rows: int, model_type: str, feature_columns: tuple[str, ...], _model, _scaler):
    price_df, _ = cached_load_data()
    return generate_trade_skip_signal_history(
        price_df,
        model=_model,
        scaler=_scaler,
        feature_columns=list(feature_columns),
        threshold=0.0,
        max_rows=max_rows,
    )


@st.cache_data(show_spinner=False)
def cached_trade_skip_reversal_signal_history(max_rows: int, feature_columns: tuple[str, ...], _model, _scaler):
    price_df, _ = cached_load_data()
    model_config = cached_trade_skip_reversal_config()
    return generate_trade_skip_signal_history(
        price_df,
        model=_model,
        scaler=_scaler,
        feature_columns=list(feature_columns),
        threshold=0.0,
        max_rows=max_rows,
        direction_rule=model_config["direction_rule"],
        cusum_volatility_window=model_config["cusum_volatility_window"],
        cusum_threshold_mult=model_config["cusum_threshold_mult"],
    )

def log(level: str, message: str):
    st.session_state.setdefault("logs", [])
    st.session_state["logs"].append({"time": pd.Timestamp.now(), "level": level, "message": message})
    st.session_state["logs"] = st.session_state["logs"][-200:]


def notify(message: str):
    st.session_state.setdefault("notifications", [])
    st.session_state["notifications"].append({"time": pd.Timestamp.now(), "message": message})
    st.session_state["notifications"] = st.session_state["notifications"][-100:]


def apply_confidence_threshold(signals: pd.DataFrame, threshold: float, require_event: bool = True) -> pd.DataFrame:
    if signals.empty:
        return signals

    updated = signals.copy()
    event_allowed = updated["event"].eq(1) if require_event else pd.Series(True, index=updated.index)
    confident = updated["confidence"].ge(threshold)
    buy = event_allowed & confident & updated["prediction"].eq("UP")
    sell = event_allowed & confident & updated["prediction"].eq("DOWN")

    updated["decision"] = "NO TRADE"
    updated.loc[buy, "decision"] = "BUY"
    updated.loc[sell, "decision"] = "SELL"
    return updated


def apply_trade_skip_threshold(signals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if signals.empty:
        return signals

    updated = signals.copy()
    can_trade = updated["event"].eq(1) & updated["probability_trade"].ge(threshold)
    updated["decision"] = "NO TRADE"
    updated.loc[can_trade & updated["event_cusum_direction"].eq(1), "decision"] = "BUY"
    updated.loc[can_trade & updated["event_cusum_direction"].eq(-1), "decision"] = "SELL"
    updated["confidence"] = updated["probability_trade"]
    return updated


def apply_news_entry_filter(signals: pd.DataFrame, news: pd.DataFrame, minutes_before: int = NEWS_FILTER_DEFAULT_MINUTES) -> pd.DataFrame:
    if signals.empty or news.empty or "time" not in signals:
        return signals

    updated = signals.copy()
    signal_times = pd.to_datetime(updated["time"], utc=True, errors="coerce")
    valid_times = signal_times.notna()
    entry_times = signal_times.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    news_times = news["time_utc"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(news_times, entry_times, side="left")

    minutes_to_news = np.full(len(updated), np.nan)
    next_news_impact = np.array(["none"] * len(updated), dtype=object)
    next_news_currency = np.array(["none"] * len(updated), dtype=object)
    next_news_name = np.array([""] * len(updated), dtype=object)
    valid_next = valid_times.to_numpy() & (positions < len(news))
    if valid_next.any():
        selected_news = news.iloc[positions[valid_next]].reset_index(drop=True)
        selected_entry_times = entry_times[valid_next]
        selected_news_times = selected_news["time_utc"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        minutes_to_news[valid_next] = (selected_news_times - selected_entry_times) / np.timedelta64(1, "m")
        next_news_impact[valid_next] = selected_news["impact"].to_numpy()
        next_news_currency[valid_next] = selected_news["currency"].to_numpy()
        next_news_name[valid_next] = selected_news.get("event_name", pd.Series([""] * len(selected_news))).astype(str).to_numpy()

    blocked = pd.Series(minutes_to_news, index=updated.index).between(0, minutes_before, inclusive="both")
    blocked &= updated["decision"].isin(["BUY", "SELL"])
    updated["news_filter_blocked"] = blocked
    updated["minutes_to_news"] = minutes_to_news
    updated["next_news_impact"] = next_news_impact
    updated["next_news_currency"] = next_news_currency
    updated["next_news_name"] = next_news_name
    updated.loc[blocked, "decision"] = "NO TRADE"
    return updated


def prediction_from_signal_history(signals: pd.DataFrame) -> dict | None:
    if signals.empty:
        return None

    last = signals.iloc[-1]
    return {
        "timestamp": last["time"],
        "event": int(last["event"]),
        "prediction": last["prediction"],
        "probability_up": float(last["probability_up"]) if "probability_up" in last else None,
        "probability_trade": float(last["probability_trade"]) if "probability_trade" in last else None,
        "confidence": float(last["confidence"]),
        "decision": last["decision"],
    }


def format_pct(value):
    if value == np.inf:
        return "∞"
    return f"{value * 100:.2f}%"


def find_trade_time_column(trade_frame: pd.DataFrame) -> str | None:
    if "Дата" in trade_frame.columns:
        return "Дата"
    best_column = None
    best_score = -1.0
    for column in trade_frame.columns:
        if column == "Exit time":
            continue
        parsed = pd.to_datetime(trade_frame[column], errors="coerce", utc=True)
        score = float(parsed.notna().mean())
        if score > best_score:
            best_column = column
            best_score = score
    return best_column if best_score >= 0.8 else None


def summarize_trades_by_period(trade_frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    if trade_frame.empty:
        return pd.DataFrame()
    time_col = find_trade_time_column(trade_frame)
    if time_col is None:
        return pd.DataFrame()
    frame = trade_frame.copy()
    frame["entry_time"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    if "PnL pips" in frame:
        frame["pnl_pips"] = frame["PnL pips"].astype(float)
    else:
        frame["pnl_pips"] = frame["PnL"].astype(float) * 10000.0
    if "PnL Money" in frame:
        frame["pnl_money"] = frame["PnL Money"].astype(float)
    frame = frame.dropna(subset=["entry_time"])
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for period, group in frame.set_index("entry_time").groupby(pd.Grouper(freq=freq)):
        if group.empty:
            continue
        pnl = group["pnl_pips"].astype(float)
        pnl_money = group["pnl_money"].astype(float) if "pnl_money" in group else None
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_loss = abs(losses.sum())
        profit_factor = wins.sum() / gross_loss if gross_loss > 0 else np.inf if wins.sum() > 0 else 0.0
        rows.append(
            {
                "Period": period.date().isoformat(),
                "Trades": int(len(group)),
                "PnL pips": float(pnl.sum()),
                "Avg pips": float(pnl.mean()),
                "Win Rate": float((pnl > 0).mean()),
                "Profit Factor": float(profit_factor),
                "PnL $": float(pnl_money.sum()) if pnl_money is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def display_period_summary(frame: pd.DataFrame):
    if frame.empty:
        st.info("Нет сделок для этой периодической сводки.")
        return
    display = frame.copy()
    display["PnL pips"] = display["PnL pips"].round(1)
    display["Avg pips"] = display["Avg pips"].round(2)
    display["Win Rate"] = (display["Win Rate"] * 100).round(2)
    display["Profit Factor"] = display["Profit Factor"].replace(np.inf, np.nan).round(3)
    if "PnL $" in display:
        display["PnL $"] = display["PnL $"].round(2)
    st.dataframe(
        display.sort_values("Period", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={"Win Rate": st.column_config.NumberColumn("Win Rate", format="%.2f%%")},
    )


def render_price_chart(df: pd.DataFrame, signals: pd.DataFrame):
    plot_df = df.tail(600).copy()
    if not HAS_PLOTLY:
        st.warning("Plotly не установлен. Показан упрощенный line chart. Для candlestick установите plotly.")
        st.line_chart(plot_df[["close", "ma_10", "ma_20"]])
        if not signals.empty:
            st.dataframe(
                signals[signals["decision"].isin(["BUY", "SELL"])].tail(30),
                use_container_width=True,
                hide_index=True,
            )
        return

    fig = go.Figure()
    has_ohlc = {"open", "high", "low", "close"}.issubset(plot_df.columns)
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["open"],
                high=plot_df["high"],
                low=plot_df["low"],
                close=plot_df["close"],
                name="EURUSD",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ma_10"], mode="lines", name="MA10", line=dict(color="#0969da")))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ma_20"], mode="lines", name="MA20", line=dict(color="#8250df")))

    events = plot_df[plot_df["event"] == 1]
    fig.add_trace(
        go.Scatter(
            x=events.index,
            y=events["close"],
            mode="markers",
            name="event=1",
            marker=dict(size=8, color="#57606a", symbol="circle-open"),
        )
    )

    if not signals.empty:
        recent_signals = signals[signals["time"].isin(plot_df.index)]
        buy = recent_signals[recent_signals["decision"] == "BUY"]
        sell = recent_signals[recent_signals["decision"] == "SELL"]
        fig.add_trace(
            go.Scatter(
                x=buy["time"],
                y=buy["close"],
                mode="markers",
                name="BUY / UP",
                marker=dict(size=13, color="#1a7f37", symbol="triangle-up"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sell["time"],
                y=sell["close"],
                mode="markers",
                name="SELL / DOWN",
                marker=dict(size=13, color="#cf222e", symbol="triangle-down"),
            )
        )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trade_setup_chart(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int,
    tp_threshold: float | None,
    sl_threshold: float | None,
    max_trades: int = 30,
):
    st.subheader("Зоны входа TP/SL")
    if df.empty or signals.empty:
        st.info("Нет сделок для отображения.")
        return

    executable = signals[signals["decision"].isin(["BUY", "SELL"])].copy()
    if executable.empty:
        st.info("Нет исполненных сигналов для выбранных параметров.")
        return

    executable["time"] = pd.to_datetime(executable["time"], utc=True, errors="coerce")
    executable = executable.dropna(subset=["time"]).tail(max_trades)
    if executable.empty:
        st.info("Нет корректных дат сделок для отображения.")
        return

    price = df.sort_index().copy()
    if price.index.tz is None:
        signal_times = executable["time"].dt.tz_localize(None)
    else:
        signal_times = executable["time"].dt.tz_convert(price.index.tz)

    entry_positions = price.index.get_indexer(signal_times, method="nearest")
    entry_positions = entry_positions[entry_positions >= 0]
    if len(entry_positions) == 0:
        st.info("Не найдены свечи, соответствующие сигналам.")
        return

    start_pos = max(int(entry_positions.min()) - 30, 0)
    end_pos = min(int(entry_positions.max()) + int(horizon) + 30, len(price) - 1)
    plot_df = price.iloc[start_pos : end_pos + 1].copy()

    if not HAS_PLOTLY:
        st.line_chart(plot_df[["close"]])
        return

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="EURUSD",
        )
    )

    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    tp_threshold = float(tp_threshold or 0.0)
    sl_threshold = float(sl_threshold or 0.0)
    horizon = max(int(horizon), 1)

    for _, signal in executable.iterrows():
        entry_time = signal["time"]
        if price.index.tz is None:
            entry_time = entry_time.tz_localize(None)
        else:
            entry_time = entry_time.tz_convert(price.index.tz)
        entry_pos = price.index.get_indexer([entry_time], method="nearest")[0]
        if entry_pos < start_pos or entry_pos > end_pos:
            continue

        exit_pos = min(entry_pos + horizon, len(price) - 1)
        entry_x = price.index[entry_pos]
        exit_x = price.index[exit_pos]
        entry_price = float(price.iloc[entry_pos]["close"])
        direction = signal["decision"]

        if direction == "BUY":
            tp_y0, tp_y1 = entry_price, entry_price + tp_threshold
            sl_y0, sl_y1 = entry_price - sl_threshold, entry_price
            buy_x.append(entry_x)
            buy_y.append(entry_price)
        else:
            tp_y0, tp_y1 = entry_price - tp_threshold, entry_price
            sl_y0, sl_y1 = entry_price, entry_price + sl_threshold
            sell_x.append(entry_x)
            sell_y.append(entry_price)

        if tp_threshold > 0:
            fig.add_shape(
                type="rect",
                x0=entry_x,
                x1=exit_x,
                y0=min(tp_y0, tp_y1),
                y1=max(tp_y0, tp_y1),
                fillcolor="rgba(26, 127, 55, 0.16)",
                line=dict(color="rgba(26, 127, 55, 0.25)", width=1),
                layer="below",
            )
        if sl_threshold > 0:
            fig.add_shape(
                type="rect",
                x0=entry_x,
                x1=exit_x,
                y0=min(sl_y0, sl_y1),
                y1=max(sl_y0, sl_y1),
                fillcolor="rgba(207, 34, 46, 0.14)",
                line=dict(color="rgba(207, 34, 46, 0.25)", width=1),
                layer="below",
            )

    fig.add_trace(
        go.Scatter(
            x=buy_x,
            y=buy_y,
            mode="markers",
            name="BUY entry",
            marker=dict(size=15, color="#1a7f37", symbol="triangle-up", line=dict(color="#ffffff", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_x,
            y=sell_y,
            mode="markers",
            name="SELL entry",
            marker=dict(size=15, color="#cf222e", symbol="triangle-down", line=dict(color="#ffffff", width=1)),
        )
    )

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trade_quality_summary(trade_frame: pd.DataFrame):
    st.subheader("Качество сделок")
    if trade_frame.empty:
        st.info("Нет сделок для сводки качества.")
        return
    frame = trade_frame.copy()
    if "PnL pips" not in frame:
        frame["PnL pips"] = frame["PnL"].astype(float) * 10000.0
    if "Gross PnL" in frame:
        frame["Gross pips"] = frame["Gross PnL"].astype(float) * 10000.0
    else:
        frame["Gross pips"] = frame["PnL pips"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Лучшая сделка gross, пункты", f"{frame['Gross pips'].max():.1f}")
    c2.metric("Худшая сделка gross, пункты", f"{frame['Gross pips'].min():.1f}")
    c3.metric("Медиана net, пункты", f"{frame['PnL pips'].median():.1f}")
    c4.metric("Средняя уверенность", f"{frame.get('confidence', pd.Series([0.0])).astype(float).mean():.3f}")

    left, right = st.columns(2)
    with left:
        if "Exit reason" in frame:
            exit_summary = (
                frame.groupby("Exit reason")
                .agg(Trades=("PnL pips", "size"), Net_pips=("PnL pips", "sum"), Gross_pips=("Gross pips", "sum"), Avg_net_pips=("PnL pips", "mean"))
                .reset_index()
            )
            exit_summary["Net_pips"] = exit_summary["Net_pips"].round(1)
            exit_summary["Gross_pips"] = exit_summary["Gross_pips"].round(1)
            exit_summary["Avg_net_pips"] = exit_summary["Avg_net_pips"].round(2)
            exit_summary = exit_summary.rename(
                columns={
                    "Exit reason": "Причина выхода",
                    "Trades": "Сделки",
                    "Net_pips": "Net, пункты",
                    "Gross_pips": "Gross, пункты",
                    "Avg_net_pips": "Средний net",
                }
            )
            st.dataframe(exit_summary, use_container_width=True, hide_index=True)
    with right:
        render_histogram(frame["PnL pips"], "Распределение net PnL, пункты")


def prepare_trade_analysis_frame(trade_frame: pd.DataFrame) -> pd.DataFrame:
    if trade_frame.empty:
        return pd.DataFrame()
    time_col = find_trade_time_column(trade_frame)
    if time_col is None:
        return pd.DataFrame()

    frame = trade_frame.copy()
    frame["entry_time"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=["entry_time"])
    if frame.empty:
        return pd.DataFrame()

    frame["pnl_pips"] = frame["PnL pips"].astype(float) if "PnL pips" in frame else frame["PnL"].astype(float) * 10000.0
    frame["pnl_money"] = frame["PnL Money"].astype(float) if "PnL Money" in frame else np.nan
    frame["hour"] = frame["entry_time"].dt.hour
    frame["weekday_num"] = frame["entry_time"].dt.dayofweek
    weekday_names = {
        0: "Пн",
        1: "Вт",
        2: "Ср",
        3: "Чт",
        4: "Пт",
        5: "Сб",
        6: "Вс",
    }
    frame["weekday"] = frame["weekday_num"].map(weekday_names)
    return frame


def render_trade_activity_charts(trade_frame: pd.DataFrame):
    st.subheader("Активность сделок")
    frame = prepare_trade_analysis_frame(trade_frame)
    if frame.empty:
        st.info("Нет сделок для анализа активности.")
        return

    by_hour = (
        frame.groupby("hour")
        .agg(Сделки=("pnl_pips", "size"), Net_пункты=("pnl_pips", "sum"), Средний_результат=("pnl_pips", "mean"))
        .reindex(range(24), fill_value=0)
        .reset_index()
        .rename(columns={"hour": "Час UTC"})
    )
    by_weekday = (
        frame.groupby(["weekday_num", "weekday"])
        .agg(Сделки=("pnl_pips", "size"), Net_пункты=("pnl_pips", "sum"), Win_rate=("pnl_pips", lambda x: float((x > 0).mean()) if len(x) else 0.0))
        .reset_index()
        .sort_values("weekday_num")
    )

    left, right = st.columns(2)
    with left:
        if HAS_PLOTLY:
            fig = go.Figure(go.Bar(x=by_hour["Час UTC"], y=by_hour["Сделки"], marker_color="#0969da"))
            fig.update_layout(
                title="Когда робот чаще входит в сделки",
                xaxis_title="Час UTC",
                yaxis_title="Количество сделок",
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(by_hour.set_index("Час UTC")["Сделки"])
    with right:
        if HAS_PLOTLY:
            fig = go.Figure(go.Bar(x=by_weekday["weekday"], y=by_weekday["Net_пункты"], marker_color="#1a7f37"))
            fig.update_layout(
                title="Net PnL по дням недели",
                xaxis_title="День недели",
                yaxis_title="Net, пункты",
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(by_weekday.set_index("weekday")["Net_пункты"])


def render_backtest_behavior_charts(trade_frame: pd.DataFrame):
    st.subheader("Поведение сделок")
    frame = prepare_trade_analysis_frame(trade_frame)
    if frame.empty:
        st.info("Нет сделок для поведенческих графиков.")
        return

    left, right = st.columns(2)
    with left:
        if "Exit reason" in frame and HAS_PLOTLY:
            exit_summary = frame.groupby("Exit reason").agg(Сделки=("pnl_pips", "size"), Net_пункты=("pnl_pips", "sum")).reset_index()
            fig = go.Figure(go.Bar(x=exit_summary["Exit reason"], y=exit_summary["Сделки"], marker_color="#8250df"))
            fig.update_layout(
                title="Причины выхода",
                xaxis_title="Причина",
                yaxis_title="Количество сделок",
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif "Exit reason" in frame:
            st.bar_chart(frame["Exit reason"].value_counts())
    with right:
        if "confidence" in frame and HAS_PLOTLY:
            fig = go.Figure(
                go.Scatter(
                    x=frame["confidence"].astype(float),
                    y=frame["pnl_pips"],
                    mode="markers",
                    marker=dict(color=np.where(frame["pnl_pips"] > 0, "#1a7f37", "#cf222e"), size=7, opacity=0.65),
                    name="Сделки",
                )
            )
            fig.update_layout(
                title="Уверенность модели и результат",
                xaxis_title="Уверенность",
                yaxis_title="Net, пункты",
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif "confidence" in frame:
            st.scatter_chart(frame[["confidence", "pnl_pips"]])


def render_line_chart(x, y, title: str = "", color: str = "#0969da"):
    if HAS_PLOTLY:
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color)))
        fig.update_layout(height=360, title=title, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame({title or "value": y}, index=x))


def render_bar_chart(df: pd.DataFrame, x_col: str, y_col: str):
    if HAS_PLOTLY:
        fig = go.Figure(go.Bar(x=df[x_col], y=df[y_col], marker_color=["#0969da", "#1a7f37", "#57606a"]))
        fig.update_layout(height=360, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x_col)[y_col])


def render_histogram(series: pd.Series, title: str):
    if HAS_PLOTLY:
        fig = go.Figure(go.Histogram(x=series, marker_color="#0969da", nbinsx=25))
        fig.update_layout(height=320, title=title, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
    else:
        bins = pd.cut(series, bins=20).value_counts().sort_index()
        st.bar_chart(bins)


def model_metrics(signals: pd.DataFrame, prepared_df: pd.DataFrame, horizon: int):
    labeled = add_labels_for_metrics(signals, prepared_df, horizon, threshold=config.DEFAULT_LABEL_THRESHOLD)
    labeled = labeled[labeled["event"] == 1].copy()
    if labeled.empty or "actual" not in labeled:
        return {}, np.array([[0, 0], [0, 0]])
    y_true = labeled["actual"]
    y_pred = labeled["prediction"]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Accuracy ÃÂ¼ÃÂ¾ÃÂ´ÃÂµÃÂ»ÃÂ¸": accuracy_score(y_true, y_pred),
        "Accuracy модели": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="UP", zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label="UP", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, pos_label="UP", zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
    }, confusion_matrix(y_true, y_pred, labels=["DOWN", "UP"])


def trade_skip_model_metrics(
    signals: pd.DataFrame,
    prepared_df: pd.DataFrame,
    horizon: int,
    tp_threshold: float,
    sl_threshold: float,
    cost_per_trade: float,
):
    labeled = signals[signals["event"] == 1].copy()
    if labeled.empty or prepared_df.empty:
        return {}, np.array([[0, 0], [0, 0]])

    candidate_signals = labeled.copy()
    candidate_signals["decision"] = "NO TRADE"
    candidate_signals.loc[candidate_signals["event_cusum_direction"].eq(1), "decision"] = "BUY"
    candidate_signals.loc[candidate_signals["event_cusum_direction"].eq(-1), "decision"] = "SELL"
    candidate_trades = build_trades(
        candidate_signals,
        prepared_df,
        horizon=horizon,
        tp_threshold=tp_threshold,
        sl_threshold=sl_threshold,
        cost_per_trade=cost_per_trade,
    )
    if candidate_trades.empty:
        return {}, np.array([[0, 0], [0, 0]])

    time_col = candidate_trades.columns[0]
    actual_trade = candidate_trades.set_index(time_col)["PnL"].astype(float).gt(0).astype(int)
    labeled = labeled[labeled["time"].isin(actual_trade.index)].copy()
    if labeled.empty:
        return {}, np.array([[0, 0], [0, 0]])

    y_true = labeled["time"].map(actual_trade).astype(int)
    y_pred = labeled["decision"].isin(["BUY", "SELL"]).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Accuracy Ð¼Ð¾Ð´ÐµÐ»Ð¸": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 TRADE": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 HOLD": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Actual profitable rate": float(y_true.mean()) if len(y_true) else 0.0,
        "Predicted trade rate": float(y_pred.mean()) if len(y_pred) else 0.0,
    }, confusion_matrix(y_true, y_pred, labels=[0, 1])


def confidence_table(signals: pd.DataFrame, prepared_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    labeled = add_labels_for_metrics(signals, prepared_df, horizon, threshold=config.DEFAULT_LABEL_THRESHOLD)
    labeled = labeled[labeled["event"] == 1].copy()
    for conf_threshold in [0.50, 0.52, 0.55, 0.57, 0.60]:
        subset = labeled[labeled["confidence"] >= conf_threshold]
        if subset.empty:
            accuracy = 0.0
        else:
            accuracy = accuracy_score(subset["actual"], subset["prediction"])
        rows.append({"threshold": conf_threshold, "accuracy": accuracy, "trades": len(subset)})
    return pd.DataFrame(rows)


def trade_skip_threshold_table(
    raw_signals: pd.DataFrame,
    prepared_df: pd.DataFrame,
    horizon: int,
    tp_threshold: float,
    sl_threshold: float,
    news: pd.DataFrame | None = None,
    news_filter: bool = False,
    news_minutes: int = NEWS_FILTER_DEFAULT_MINUTES,
    cost_per_trade: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for threshold in [0.45, 0.50, 0.51, 0.52, 0.53, 0.55, 0.57, 0.60]:
        signals = apply_trade_skip_threshold(raw_signals, threshold)
        if news_filter and news is not None:
            signals = apply_news_entry_filter(signals, news, int(news_minutes))
        trades = build_trades(
            signals,
            prepared_df,
            horizon=horizon,
            tp_threshold=tp_threshold,
            sl_threshold=sl_threshold,
            cost_per_trade=cost_per_trade,
        )
        metrics = calculate_trade_metrics(trades)
        rows.append(
            {
                "threshold": threshold,
                "trades": metrics["Trades"],
                "total_return": metrics["Total Return"],
                "winrate, %": metrics["Win Rate"] * 100,
                "profit_factor": metrics["Profit Factor"],
            }
        )
    return pd.DataFrame(rows)


def summarize_strategy(name: str, strategy_signals: pd.DataFrame, prepared_df: pd.DataFrame, horizon: int, tp_threshold: float, sl_threshold: float, cost_per_trade: float, initial_balance: float, risk_per_trade_pct: float, pip_value_per_lot: float) -> dict:
    strategy_trades = build_trades(
        strategy_signals,
        prepared_df,
        horizon=horizon,
        tp_threshold=tp_threshold,
        sl_threshold=sl_threshold,
        cost_per_trade=cost_per_trade,
    )
    strategy_trades = apply_risk_sizing(strategy_trades, initial_capital=initial_balance, risk_per_trade_pct=risk_per_trade_pct, sl_threshold=sl_threshold, pip_value_per_lot=pip_value_per_lot)
    trade_metrics = calculate_risk_trade_metrics(strategy_trades, initial_capital=initial_balance)
    clf_metrics, _ = model_metrics(strategy_signals, prepared_df, horizon)
    return {
        "Strategy": name,
        "Accuracy": clf_metrics.get("Accuracy Ð¼Ð¾Ð´ÐµÐ»Ð¸", 0.0),
        "F1": clf_metrics.get("F1-score", 0.0),
        "Balanced accuracy": clf_metrics.get("Balanced accuracy", 0.0),
        "Trades": trade_metrics["Trades"],
        "Total Return": trade_metrics["Total Return"],
        "Win Rate": trade_metrics["Win Rate"],
        "Profit Factor": trade_metrics["Profit Factor"],
        "Accuracy": clf_metrics.get("Accuracy", 0.0),
    }


def summarize_trade_skip_strategy(name: str, strategy_signals: pd.DataFrame, prepared_df: pd.DataFrame, horizon: int, tp_threshold: float, sl_threshold: float, cost_per_trade: float, initial_balance: float, risk_per_trade_pct: float, pip_value_per_lot: float) -> dict:
    strategy_trades = build_trades(
        strategy_signals,
        prepared_df,
        horizon=horizon,
        tp_threshold=tp_threshold,
        sl_threshold=sl_threshold,
        cost_per_trade=cost_per_trade,
    )
    strategy_trades = apply_risk_sizing(strategy_trades, initial_capital=initial_balance, risk_per_trade_pct=risk_per_trade_pct, sl_threshold=sl_threshold, pip_value_per_lot=pip_value_per_lot)
    trade_metrics = calculate_risk_trade_metrics(strategy_trades, initial_capital=initial_balance)
    clf_metrics, _ = trade_skip_model_metrics(strategy_signals, prepared_df, horizon, tp_threshold, sl_threshold, cost_per_trade)
    return {
        "Strategy": name,
        "Accuracy": clf_metrics.get("Accuracy", 0.0),
        "F1": clf_metrics.get("F1 macro", clf_metrics.get("F1-score", 0.0)),
        "Balanced accuracy": clf_metrics.get("Balanced accuracy", 0.0),
        "Trades": trade_metrics["Trades"],
        "Total Return": trade_metrics["Total Return"],
        "Win Rate": trade_metrics["Win Rate"],
        "Profit Factor": trade_metrics["Profit Factor"],
    }


def build_strategy_comparison_rows(prepared_df: pd.DataFrame, price_df: pd.DataFrame, q_candles: int, threshold: float, horizon: int, tp_threshold: float, sl_threshold: float, cost_per_trade: float, initial_balance: float, risk_per_trade_pct: float, pip_value_per_lot: float, news_calendar: pd.DataFrame, news_filter_enabled: bool, news_filter_minutes: int) -> list[dict]:
    strategy_rows = []
    try:
        reversal_config = cached_trade_skip_reversal_config()
        reversal_model = cached_trade_skip_reversal_model_resource()
        reversal_scaler = cached_trade_skip_reversal_scaler_resource()
        raw_reversal_signals = cached_trade_skip_reversal_signal_history(
            q_candles,
            tuple(reversal_config["feature_columns"]),
            reversal_model,
            reversal_scaler,
        )
        reversal_signals = apply_trade_skip_threshold(raw_reversal_signals, threshold)
        if news_filter_enabled:
            reversal_signals = apply_news_entry_filter(reversal_signals, news_calendar, int(news_filter_minutes))
        strategy_rows.append(summarize_trade_skip_strategy("TradeSkip GRU Reversal", reversal_signals, prepared_df, horizon, tp_threshold, sl_threshold, cost_per_trade, initial_balance, risk_per_trade_pct, pip_value_per_lot))
    except Exception:
        strategy_rows.append({"Strategy": "TradeSkip GRU Reversal", "Accuracy": 0.0, "F1": 0.0, "Balanced accuracy": 0.0, "Trades": 0, "Total Return": 0.0, "Win Rate": 0.0, "Profit Factor": 0.0})

    try:
        trade_skip_config = cached_trade_skip_config("gru")
        trade_skip_model = cached_trade_skip_model_resource("gru")
        trade_skip_scaler = cached_trade_skip_scaler_resource("gru")
        raw_trade_skip_signals = cached_trade_skip_signal_history(
            q_candles,
            "gru",
            tuple(trade_skip_config["feature_columns"]),
            trade_skip_model,
            trade_skip_scaler,
        )
        trade_skip_signals = apply_trade_skip_threshold(raw_trade_skip_signals, threshold)
        strategy_rows.append(summarize_trade_skip_strategy("TradeSkip GRU", trade_skip_signals, prepared_df, horizon, tp_threshold, sl_threshold, cost_per_trade, initial_balance, risk_per_trade_pct, pip_value_per_lot))
    except Exception:
        strategy_rows.append({"Strategy": "TradeSkip GRU", "Accuracy": 0.0, "F1": 0.0, "Balanced accuracy": 0.0, "Trades": 0, "Total Return": 0.0, "Win Rate": 0.0, "Profit Factor": 0.0})

    for strategy_model_type in ["gru", "lstm"]:
        try:
            strategy_model = cached_model_resource(strategy_model_type, "event")
            strategy_scaler = cached_scaler_resource(strategy_model_type, "event")
            raw_strategy_signals = cached_signal_history(q_candles, strategy_model_type, True, strategy_model, strategy_scaler)
            strategy_signals = apply_confidence_threshold(raw_strategy_signals, threshold, require_event=True)
            strategy_rows.append(summarize_strategy(f"Event + {strategy_model_type.upper()}", strategy_signals, prepared_df, horizon, tp_threshold, sl_threshold, cost_per_trade, initial_balance, risk_per_trade_pct, pip_value_per_lot))
        except Exception:
            strategy_rows.append({"Strategy": f"Event + {strategy_model_type.upper()}", "Accuracy": 0.0, "F1": 0.0, "Balanced accuracy": 0.0, "Trades": 0, "Total Return": 0.0, "Win Rate": 0.0, "Profit Factor": 0.0})

    if not prepared_df.empty:
        rule_signals = generate_rule_based_signal_history(price_df, max_rows=q_candles)
        strategy_rows.append(summarize_strategy("Rule CUSUM Reversal", rule_signals, prepared_df, horizon, tp_threshold, sl_threshold, cost_per_trade, initial_balance, risk_per_trade_pct, pip_value_per_lot))
    return strategy_rows


def render_strategy_comparison(strategy_rows: list[dict]):
    strategy_df = pd.DataFrame(strategy_rows)
    st.subheader("Сравнение стратегий")
    strategy_display_df = strategy_df.copy()
    if "Win Rate" in strategy_display_df:
        strategy_display_df["Win Rate"] = (strategy_display_df["Win Rate"] * 100).round(2)
    strategy_display_df = strategy_display_df.rename(
        columns={
            "Strategy": "Стратегия",
            "Accuracy": "Accuracy",
            "Balanced accuracy": "Balanced accuracy",
            "Trades": "Сделки",
            "Total Return": "Доходность",
            "Win Rate": "Win Rate",
            "Profit Factor": "Profit Factor",
        }
    )
    st.dataframe(
        strategy_display_df,
        use_container_width=True,
        hide_index=True,
        column_config={"Win Rate": st.column_config.NumberColumn("Win Rate", format="%.2f%%")},
    )
    if not strategy_df.empty:
        render_bar_chart(strategy_df, "Strategy", "Total Return")


def render_period_performance(trade_frame: pd.DataFrame):
    st.subheader("Результаты по периодам")
    week_tab, month_tab, year_tab = st.tabs(["Недели", "Месяцы", "Годы"])
    with week_tab:
        display_period_summary(summarize_trades_by_period(trade_frame, "W-MON"))
    with month_tab:
        display_period_summary(summarize_trades_by_period(trade_frame, "ME"))
    with year_tab:
        display_period_summary(summarize_trades_by_period(trade_frame, "YE"))


def render_confidence_analysis(raw_signals: pd.DataFrame, signals: pd.DataFrame, prepared_df: pd.DataFrame, horizon: int, tp_threshold: float, sl_threshold: float, cost_per_trade: float, selected_strategy: str, news_calendar: pd.DataFrame, news_filter_enabled: bool, news_filter_minutes: int):
    if signals.empty or prepared_df.empty:
        return
    st.subheader("Анализ порога уверенности")
    if selected_strategy in ("TradeSkip GRU", "TradeSkip GRU Reversal"):
        st.dataframe(
            trade_skip_threshold_table(
                raw_signals,
                prepared_df,
                horizon,
                tp_threshold,
                sl_threshold,
                news=news_calendar,
                news_filter=news_filter_enabled and selected_strategy == "TradeSkip GRU Reversal",
                news_minutes=int(news_filter_minutes),
                cost_per_trade=cost_per_trade,
            ),
            use_container_width=True,
            hide_index=True,
            column_config={"winrate, %": st.column_config.NumberColumn("winrate, %", format="%.2f%%")},
        )
    else:
        st.dataframe(confidence_table(signals, prepared_df, horizon), use_container_width=True, hide_index=True)


def current_user_settings() -> dict:
    return {key: st.session_state.get(key, default) for key, default in DEFAULT_USER_SETTINGS.items()}


def persist_user_settings() -> None:
    save_user_settings(current_user_settings())


def initialize_control_state():
    defaults = {
        "robot_enabled": True,
        "selected_strategy": "TradeSkip GRU Reversal",
        "selected_model_type": config.MODEL_TYPE,
        "threshold": TRADE_SKIP_DEFAULT_THRESHOLD,
        "horizon": config.DEFAULT_HORIZON_CANDLES,
        "fixed_exit_enabled": True,
        "tp_pips": DEFAULT_TP_PIPS,
        "sl_pips": DEFAULT_SL_PIPS,
        "tp_threshold": DEFAULT_TP_PIPS / 10000.0,
        "sl_threshold": DEFAULT_SL_PIPS / 10000.0,
        "settings_tp_pips": DEFAULT_TP_PIPS,
        "settings_sl_pips": DEFAULT_SL_PIPS,
        "news_filter_enabled": True,
        "news_filter_minutes": NEWS_FILTER_DEFAULT_MINUTES,
        "include_costs": True,
        "spread_pips": DEFAULT_SPREAD_PIPS,
        "slippage_pips": DEFAULT_SLIPPAGE_PIPS,
        "commission_pips": DEFAULT_COMMISSION_PIPS,
        "initial_balance": DEFAULT_INITIAL_BALANCE,
        "risk_per_trade_pct": DEFAULT_RISK_PER_TRADE_PCT,
        "pip_value_per_lot": DEFAULT_PIP_VALUE_PER_LOT,
        "control_defaults_version": CONTROL_DEFAULTS_VERSION,
    }
    defaults.update(load_user_settings())
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.get("control_defaults_version") != CONTROL_DEFAULTS_VERSION:
        st.session_state["initial_balance"] = DEFAULT_INITIAL_BALANCE
        st.session_state["sl_pips"] = DEFAULT_SL_PIPS
        st.session_state["sl_threshold"] = DEFAULT_SL_PIPS / 10000.0
        st.session_state["include_costs"] = True
        st.session_state["control_defaults_version"] = CONTROL_DEFAULTS_VERSION
    if st.session_state.get("fixed_exit_enabled", True):
        st.session_state["tp_threshold"] = float(st.session_state["tp_pips"]) / 10000.0
        st.session_state["sl_threshold"] = float(st.session_state["sl_pips"]) / 10000.0


def render_model_controls(key_prefix: str):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Параметры модели")
    strategies = ["TradeSkip GRU Reversal", "Rule CUSUM Reversal", "TradeSkip GRU", "Event + GRU", "Event + LSTM", "Rule baseline"]
    with st.form(f"{key_prefix}_model_controls_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            robot_enabled_value = st.checkbox("Робот включен", value=bool(st.session_state["robot_enabled"]))
            selected_strategy_value = st.selectbox(
                "Стратегия",
                strategies,
                index=strategies.index(st.session_state["selected_strategy"]),
            )
        with c2:
            selected_model_type_value = st.selectbox(
                "Модель",
                ["gru", "lstm"],
                index=["gru", "lstm"].index(st.session_state["selected_model_type"]),
            )
            threshold_value = st.slider("Порог уверенности", min_value=0.40, max_value=0.95, value=float(st.session_state["threshold"]), step=0.01)
        with c3:
            horizon_value = st.number_input("Горизонт, свечей", min_value=1, max_value=96, value=int(st.session_state["horizon"]), step=1)
            fixed_exit_enabled_value = st.checkbox("Фиксированные SL/TP от входа", value=bool(st.session_state["fixed_exit_enabled"]))
            if fixed_exit_enabled_value:
                tp_pips_value = st.number_input("TP от входа, пункты", min_value=0.0, max_value=100.0, value=float(st.session_state["tp_pips"]), step=0.5)
                tp_threshold_value = tp_pips_value / 10000.0
            else:
                tp_threshold_value = st.number_input("TP threshold", min_value=0.0, max_value=0.01, value=float(st.session_state["tp_threshold"]), step=0.0001, format="%.4f")
                tp_pips_value = tp_threshold_value * 10000.0
        with c4:
            if fixed_exit_enabled_value:
                sl_pips_value = st.number_input("SL от входа, пункты", min_value=0.0, max_value=100.0, value=float(st.session_state["sl_pips"]), step=0.5)
                sl_threshold_value = sl_pips_value / 10000.0
            else:
                sl_threshold_value = st.number_input("SL threshold", min_value=0.0, max_value=0.01, value=float(st.session_state["sl_threshold"]), step=0.0001, format="%.4f")
                sl_pips_value = sl_threshold_value * 10000.0
            news_filter_enabled_value = st.checkbox("Фильтр новостей", value=bool(st.session_state["news_filter_enabled"]))
            news_filter_minutes_value = st.number_input("Не входить до новости, мин", min_value=0, max_value=240, value=int(st.session_state["news_filter_minutes"]), step=15)
        submitted = st.form_submit_button("Применить параметры", type="primary")
    if submitted:
        st.session_state["robot_enabled"] = robot_enabled_value
        st.session_state["selected_strategy"] = selected_strategy_value
        st.session_state["selected_model_type"] = selected_model_type_value
        st.session_state["threshold"] = threshold_value
        st.session_state["horizon"] = horizon_value
        st.session_state["fixed_exit_enabled"] = fixed_exit_enabled_value
        st.session_state["tp_pips"] = tp_pips_value
        st.session_state["sl_pips"] = sl_pips_value
        st.session_state["settings_tp_pips"] = tp_pips_value
        st.session_state["settings_sl_pips"] = sl_pips_value
        st.session_state["tp_threshold"] = tp_threshold_value
        st.session_state["sl_threshold"] = sl_threshold_value
        st.session_state["news_filter_enabled"] = news_filter_enabled_value
        st.session_state["news_filter_minutes"] = news_filter_minutes_value
        persist_user_settings()
    st.markdown("</div>", unsafe_allow_html=True)


def render_cost_settings():
    st.subheader("Комиссии и издержки")
    st.toggle("Учитывать комиссии", key="include_costs")
    cost_rows = [
        {"Комиссия": "Spread", "Значение, pips": float(st.session_state["spread_pips"])},
        {"Комиссия": "Slippage", "Значение, pips": float(st.session_state["slippage_pips"])},
        {"Комиссия": "Commission", "Значение, pips": float(st.session_state["commission_pips"])},
    ]
    edited = st.data_editor(
        pd.DataFrame(cost_rows),
        use_container_width=True,
        hide_index=True,
        disabled=["Комиссия"],
        column_config={"Значение, pips": st.column_config.NumberColumn("Значение, pips", min_value=0.0, max_value=20.0, step=0.1, format="%.2f")},
        key="costs_editor",
    )
    if not edited.empty:
        values = dict(zip(edited["Комиссия"], edited["Значение, pips"]))
        st.session_state["spread_pips"] = float(values.get("Spread", DEFAULT_SPREAD_PIPS))
        st.session_state["slippage_pips"] = float(values.get("Slippage", DEFAULT_SLIPPAGE_PIPS))
        st.session_state["commission_pips"] = float(values.get("Commission", DEFAULT_COMMISSION_PIPS))
        persist_user_settings()
    total = total_cost_threshold() * 10000
    st.caption(f"Итого учитывается: {total:.2f} пункта на сделку")


def render_risk_controls():
    st.subheader("Управление риском")
    c1, c2, c3 = st.columns(3)
    with c1:
        initial_balance_value = st.number_input("Начальный баланс, $", min_value=100.0, max_value=10_000_000.0, value=float(st.session_state["initial_balance"]), step=100.0)
    with c2:
        risk_value = st.number_input("Риск на сделку, %", min_value=0.0, max_value=20.0, value=float(st.session_state["risk_per_trade_pct"]), step=0.1)
    with c3:
        pip_value = st.number_input("Стоимость пункта за 1 лот, $", min_value=0.01, max_value=1000.0, value=float(st.session_state["pip_value_per_lot"]), step=0.5)
    st.session_state["initial_balance"] = initial_balance_value
    st.session_state["risk_per_trade_pct"] = risk_value
    st.session_state["pip_value_per_lot"] = pip_value
    persist_user_settings()


if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "notifications" not in st.session_state:
    st.session_state["notifications"] = []
initialize_control_state()

st.sidebar.title("AI Trading Robot")
page = st.sidebar.radio(
    "Навигация",
    ["Обзор", "Статистика", "Сделки", "Позиции", "Настройки", "Журналы", "Уведомления", "Backtest"],
)

if page in ("Статистика", "Backtest"):
    render_model_controls("stats" if page == "Статистика" else "backtest")
    render_risk_controls()

robot_enabled = st.session_state["robot_enabled"]
selected_strategy = st.session_state["selected_strategy"]
selected_model_type = st.session_state["selected_model_type"]
threshold = st.session_state["threshold"]
horizon = st.session_state["horizon"]
tp_threshold = st.session_state["tp_threshold"]
sl_threshold = st.session_state["sl_threshold"]
news_filter_enabled = st.session_state["news_filter_enabled"]
news_filter_minutes = st.session_state["news_filter_minutes"]
cost_per_trade = total_cost_threshold()
initial_balance = float(st.session_state["initial_balance"])
risk_per_trade_pct = float(st.session_state["risk_per_trade_pct"])
pip_value_per_lot = float(st.session_state["pip_value_per_lot"])

data_error = None
model_error = None
price_df = pd.DataFrame()
prepared_df = pd.DataFrame()
signals = pd.DataFrame()
raw_signals = pd.DataFrame()
trades = pd.DataFrame()
news_calendar = pd.DataFrame()
last_prediction = None
loaded_files = []
q_candles = FALLBACK_Q_CANDLES

try:
    price_df, loaded_files = cached_load_data()
    news_calendar = cached_news_calendar()
    log("INFO", f"Загружены CSV-файлы: {', '.join(loaded_files)}")
    prepared_df = cached_prepare_data()
    valid_end = pd.Timestamp(config.VALID_END_DATE)
    if prepared_df.index.tz is not None:
        valid_end = valid_end.tz_localize(prepared_df.index.tz)
    q_candles = max(int((prepared_df.index >= valid_end).sum()), FALLBACK_Q_CANDLES)
    log("INFO", f"Сгенерированы признаки и events: строк {len(prepared_df)}, event=1 {int(prepared_df['event'].sum())}")
except Exception as exc:
    data_error = exc
    log("ERROR", f"Ошибка загрузки данных: {exc}")

# Тест по последним свечам
if data_error is None:
    try:

        if selected_strategy == "TradeSkip GRU Reversal":
            trade_skip_config = cached_trade_skip_reversal_config()
            model_resource = cached_trade_skip_reversal_model_resource()
            scaler_resource = cached_trade_skip_reversal_scaler_resource()
            raw_signals = cached_trade_skip_reversal_signal_history(
                q_candles,
                tuple(trade_skip_config["feature_columns"]),
                model_resource,
                scaler_resource,
            )
            signals = apply_trade_skip_threshold(raw_signals, threshold)
            if news_filter_enabled:
                signals = apply_news_entry_filter(signals, news_calendar, int(news_filter_minutes))
        elif selected_strategy == "TradeSkip GRU":
            trade_skip_model_type = "gru"
            trade_skip_config = cached_trade_skip_config(trade_skip_model_type)
            model_resource = cached_trade_skip_model_resource(trade_skip_model_type)
            scaler_resource = cached_trade_skip_scaler_resource(trade_skip_model_type)
            raw_signals = cached_trade_skip_signal_history(
                q_candles,
                trade_skip_model_type,
                tuple(trade_skip_config["feature_columns"]),
                model_resource,
                scaler_resource,
            )
            signals = apply_trade_skip_threshold(raw_signals, threshold)
        elif selected_strategy in ("Rule CUSUM Reversal", "Rule baseline"):
            raw_signals = generate_rule_based_signal_history(price_df, max_rows=q_candles)
            signals = raw_signals.copy()
        else:
            active_model_type = "lstm" if selected_strategy == "Event + LSTM" else "gru"
            model_resource = cached_model_resource(active_model_type, "event")
            scaler_resource = cached_scaler_resource(active_model_type, "event")
            raw_signals = cached_signal_history(q_candles, active_model_type, True, model_resource, scaler_resource)
            signals = apply_confidence_threshold(raw_signals, threshold, require_event=True)
        trades = build_trades(signals, prepared_df, horizon=horizon, tp_threshold=tp_threshold, sl_threshold=sl_threshold, cost_per_trade=cost_per_trade)
        trades = apply_risk_sizing(
            trades,
            initial_capital=initial_balance,
            risk_per_trade_pct=risk_per_trade_pct,
            sl_threshold=sl_threshold,
            pip_value_per_lot=pip_value_per_lot,
        )
        last_prediction = prediction_from_signal_history(signals)
        log("INFO", f"Сгенерированы сигналы модели: {len(signals)}")
        if last_prediction and last_prediction["event"] == 1:
            notify("Найден новый event")
        if last_prediction and last_prediction["decision"] in ("BUY", "SELL"):
            notify(f"Создан сигнал {last_prediction['decision']}")
        notify("Модель загружена")
    except Exception as exc:
        model_error = exc
        log("ERROR", f"Ошибка inference: {exc}")

if data_error is not None:
    st.error(f"Не удалось загрузить данные: {data_error}")
    st.info("Положите CSV-файлы EURUSD в папку data/ и обновите страницу.")
if model_error is not None:
    st.error(f"Модель или scaler недоступны: {model_error}")
    with st.expander("Технические детали"):
        st.code(traceback.format_exc())

metrics = calculate_risk_trade_metrics(trades, initial_capital=initial_balance)
equity = calculate_risk_equity_curve(trades, initial_capital=initial_balance)
ml_metrics = {}
if selected_strategy in ("TradeSkip GRU", "TradeSkip GRU Reversal") and not signals.empty and not prepared_df.empty:
    ml_metrics, _ = trade_skip_model_metrics(signals, prepared_df, int(horizon), float(tp_threshold), float(sl_threshold), float(cost_per_trade))
elif not signals.empty and not prepared_df.empty:
    ml_metrics, _ = model_metrics(signals, prepared_df, int(horizon))

if page == "Обзор":
    st.title("Обзор торгового робота")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Статус робота", "Запущен" if robot_enabled else "Остановлен")
    c2.metric("Текущий капитал", f"${metrics['Current Capital']:,.2f}")
    c3.metric("Прибыль за период", format_pct(metrics["Total Return"]))
    c4.metric("Количество сделок", int(metrics["Trades"]))
    c5.metric("Win rate", format_pct(metrics["Win Rate"]))
    signal_text = last_prediction["decision"] if last_prediction else "Нет данных"
    c6.metric("Последний сигнал", signal_text)

    st.subheader("Кривая капитала")
    render_line_chart(equity["Дата"], equity["Equity"], "Кривая капитала")

    st.subheader("Последние события")
    events_view = prepared_df[prepared_df.get("event", 0) == 1].tail(10)[["close", "event_cusum_direction", "near_level", "strong_candle", "breakout"]] if not prepared_df.empty else pd.DataFrame()
    st.dataframe(events_view, use_container_width=True)

elif page == "Статистика":
    st.title("Статистика")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Финальный баланс", f"${metrics['Current Capital']:,.2f}")
    k2.metric("Доходность", format_pct(metrics["Total Return"]))
    k3.metric("Сделки", int(metrics["Trades"]))
    k4.metric("Win Rate", format_pct(metrics["Win Rate"]))
    k5.metric("Profit Factor", "∞" if metrics["Profit Factor"] == np.inf else f"{metrics['Profit Factor']:.2f}")
    k6.metric("Макс. просадка", format_pct(metrics["Max Drawdown"]))

    rows = [
        ("Доходность", format_pct(metrics["Total Return"])),
        ("Итоговый PnL, $", f"${metrics.get('Total PnL Money', 0.0):,.2f}"),
        ("Финальный баланс, $", f"${metrics['Current Capital']:,.2f}"),
        ("Win Rate", format_pct(metrics["Win Rate"])),
        ("Profit Factor", "∞" if metrics["Profit Factor"] == np.inf else f"{metrics['Profit Factor']:.2f}"),
        ("Макс. просадка", format_pct(metrics["Max Drawdown"])),
        ("Макс. просадка, $", f"${metrics.get('Max Drawdown Money', 0.0):,.2f}"),
        ("Средняя сделка, $", f"${metrics.get('Average Trade Money', 0.0):,.2f}"),
        ("Средний лот", f"{metrics.get('Average Lot Size', 0.0):.3f}"),
        ("Accuracy модели", format_pct(ml_metrics.get("Accuracy", 0))),
        ("Precision", format_pct(ml_metrics.get("Precision", 0))),
        ("Recall", format_pct(ml_metrics.get("Recall", 0))),
        ("F1 TRADE", format_pct(ml_metrics.get("F1 TRADE", ml_metrics.get("F1-score", 0)))),
        ("F1 HOLD", format_pct(ml_metrics.get("F1 HOLD", 0))),
        ("F1 macro", format_pct(ml_metrics.get("F1 macro", ml_metrics.get("F1-score", 0)))),
        ("Balanced accuracy", format_pct(ml_metrics.get("Balanced accuracy", 0))),
        ("Actual profitable rate", format_pct(ml_metrics.get("Actual profitable rate", 0))),
        ("Predicted trade rate", format_pct(ml_metrics.get("Predicted trade rate", 0))),
    ]
    stat_tab, chart_tab, periods_tab, quality_tab = st.tabs(["Сводка", "Зоны сделки", "Периоды", "Качество"])
    with stat_tab:
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(pd.DataFrame(rows, columns=["Метрика", "Значение"]), use_container_width=True, hide_index=True)
        with right:
            render_line_chart(equity["Дата"], equity["Equity"], "Кривая капитала", color="#1a7f37")
    with chart_tab:
        render_trade_setup_chart(prepared_df, signals, int(horizon), float(tp_threshold), float(sl_threshold))
        st.subheader("Направление event")
        if prepared_df.empty:
            st.warning("Нет подготовленных данных для графика.")
        else:
            render_price_chart(prepared_df, signals)
    with periods_tab:
        render_period_performance(trades)
    with quality_tab:
        render_trade_quality_summary(trades)

elif page == "Графики":
    st.title("Графики EURUSD")
    if prepared_df.empty:
        st.warning("Нет подготовленных данных для графика.")
    else:
        render_price_chart(prepared_df, signals)

elif page == "Сделки":
    st.title("Сделки")
    if trades.empty:
        st.info("Сделки пока не сгенерированы. Проверьте CSV, модель и threshold confidence.")
    else:
        st.dataframe(trades.sort_values("Дата", ascending=False), use_container_width=True, hide_index=True)

elif page == "Позиции":
    st.title("Позиции")
    open_positions = pd.DataFrame()
    if open_positions.empty:
        st.info("Открытых позиций сейчас нет")
    else:
        st.dataframe(open_positions, use_container_width=True)

elif page == "Настройки":
    st.title("Настройки")
    strategies = ["TradeSkip GRU Reversal", "Rule CUSUM Reversal", "TradeSkip GRU", "Event + GRU", "Event + LSTM", "Rule baseline"]
    with st.form("settings_form"):
        enabled = st.checkbox("Робот включен", value=bool(st.session_state["robot_enabled"]))
        instrument = st.selectbox("Инструмент", ["EURUSD"])
        strategy = st.selectbox("Стратегия", strategies, index=strategies.index(st.session_state["selected_strategy"]))
        threshold_form = st.slider("Порог уверенности", min_value=0.40, max_value=0.95, value=float(st.session_state["threshold"]), step=0.01)
        horizon_form = st.number_input("Горизонт, свечей", min_value=1, max_value=96, value=int(st.session_state["horizon"]), step=1)
        risk = st.number_input("Риск на сделку, %", min_value=0.1, max_value=10.0, value=float(st.session_state["risk_per_trade_pct"]), step=0.1)

        st.subheader("SL/TP")
        fixed_exit = st.checkbox("Фиксированные SL/TP от входа", value=bool(st.session_state["fixed_exit_enabled"]))
        c1, c2 = st.columns(2)
        with c1:
            if fixed_exit:
                tp_pips_form = st.number_input("TP от входа, пункты", min_value=0.0, max_value=100.0, value=float(st.session_state["tp_pips"]), step=0.5)
                tp_threshold_form = tp_pips_form / 10000.0
            else:
                tp_threshold_form = st.number_input("TP threshold", min_value=0.0, max_value=0.01, value=float(st.session_state["tp_threshold"]), step=0.0001, format="%.4f")
                tp_pips_form = tp_threshold_form * 10000.0
        with c2:
            if fixed_exit:
                sl_pips_form = st.number_input("SL от входа, пункты", min_value=0.0, max_value=100.0, value=float(st.session_state["sl_pips"]), step=0.5)
                sl_threshold_form = sl_pips_form / 10000.0
            else:
                sl_threshold_form = st.number_input("SL threshold", min_value=0.0, max_value=0.01, value=float(st.session_state["sl_threshold"]), step=0.0001, format="%.4f")
                sl_pips_form = sl_threshold_form * 10000.0

        st.subheader("Комиссии и издержки")
        include_costs_form = st.checkbox("Учитывать комиссии", value=bool(st.session_state["include_costs"]))
        cost_rows = [
            {"Комиссия": "Spread", "Значение, pips": float(st.session_state["spread_pips"])},
            {"Комиссия": "Slippage", "Значение, pips": float(st.session_state["slippage_pips"])},
            {"Комиссия": "Commission", "Значение, pips": float(st.session_state["commission_pips"])},
        ]
        edited_costs = st.data_editor(
            pd.DataFrame(cost_rows),
            use_container_width=True,
            hide_index=True,
            disabled=["Комиссия"],
            column_config={"Значение, pips": st.column_config.NumberColumn("Значение, pips", min_value=0.0, max_value=20.0, step=0.1, format="%.2f")},
            key="settings_costs_editor",
        )
        submitted = st.form_submit_button("Сохранить настройки", type="primary")

    if submitted:
        st.session_state["robot_enabled"] = enabled
        st.session_state["selected_strategy"] = strategy
        st.session_state["threshold"] = threshold_form
        st.session_state["horizon"] = horizon_form
        st.session_state["fixed_exit_enabled"] = fixed_exit
        st.session_state["tp_pips"] = tp_pips_form
        st.session_state["sl_pips"] = sl_pips_form
        st.session_state["settings_tp_pips"] = tp_pips_form
        st.session_state["settings_sl_pips"] = sl_pips_form
        st.session_state["tp_threshold"] = tp_threshold_form
        st.session_state["sl_threshold"] = sl_threshold_form
        st.session_state["include_costs"] = include_costs_form
        st.session_state["risk_per_trade_pct"] = risk
        if not edited_costs.empty:
            values = dict(zip(edited_costs["Комиссия"], edited_costs["Значение, pips"]))
            st.session_state["spread_pips"] = float(values.get("Spread", DEFAULT_SPREAD_PIPS))
            st.session_state["slippage_pips"] = float(values.get("Slippage", DEFAULT_SLIPPAGE_PIPS))
            st.session_state["commission_pips"] = float(values.get("Commission", DEFAULT_COMMISSION_PIPS))
        persist_user_settings()
        st.success(f"Настройки сохранены: {USER_SETTINGS_PATH}")
        log("INFO", f"Настройки обновлены: {enabled}, {instrument}, {strategy}, threshold={threshold_form}, horizon={horizon_form}, tp={st.session_state['tp_threshold']}, sl={st.session_state['sl_threshold']}, risk={risk}")

elif page == "Журналы":
    st.title("Журналы")
    logs = pd.DataFrame(st.session_state.get("logs", []))
    if logs.empty:
        st.info("Журнал пока пуст.")
    else:
        st.dataframe(logs.sort_values("time", ascending=False), use_container_width=True, hide_index=True)

elif page == "Backtest":
    st.title("Бэктест")
    strategy_rows = build_strategy_comparison_rows(
        prepared_df=prepared_df,
        price_df=price_df,
        q_candles=q_candles,
        threshold=float(threshold),
        horizon=int(horizon),
        tp_threshold=float(tp_threshold),
        sl_threshold=float(sl_threshold),
        cost_per_trade=float(cost_per_trade),
        initial_balance=initial_balance,
        risk_per_trade_pct=risk_per_trade_pct,
        pip_value_per_lot=pip_value_per_lot,
        news_calendar=news_calendar,
        news_filter_enabled=bool(news_filter_enabled),
        news_filter_minutes=int(news_filter_minutes),
    )
    summary_tab, setup_tab, periods_tab, activity_tab, confidence_tab = st.tabs(
        ["Сводка", "Зоны сделки", "Периоды", "Активность", "Порог уверенности"]
    )
    with summary_tab:
        render_strategy_comparison(strategy_rows)
        left, right = st.columns([1, 1])
        with left:
            render_line_chart(equity["Дата"], equity["Equity"], "Кривая капитала", color="#1a7f37")
        with right:
            render_trade_quality_summary(trades)
    with setup_tab:
        render_trade_setup_chart(prepared_df, signals, int(horizon), float(tp_threshold), float(sl_threshold))
    with periods_tab:
        render_period_performance(trades)
    with activity_tab:
        render_trade_activity_charts(trades)
        render_backtest_behavior_charts(trades)
    with confidence_tab:
        render_confidence_analysis(
            raw_signals=raw_signals,
            signals=signals,
            prepared_df=prepared_df,
            horizon=int(horizon),
            tp_threshold=float(tp_threshold),
            sl_threshold=float(sl_threshold),
            cost_per_trade=float(cost_per_trade),
            selected_strategy=selected_strategy,
            news_calendar=news_calendar,
            news_filter_enabled=bool(news_filter_enabled),
            news_filter_minutes=int(news_filter_minutes),
        )

elif page == "Уведомления":
    st.title("Уведомления")
    notifications = pd.DataFrame(st.session_state.get("notifications", []))
    if notifications.empty:
        st.info("Уведомлений пока нет.")
    else:
        st.dataframe(notifications.sort_values("time", ascending=False), use_container_width=True, hide_index=True)
