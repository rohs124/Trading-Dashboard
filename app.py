import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import plotly.express as px
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import yfinance as yf

# --- Page Config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")

# --- API Keys ---
ALPHA_API_KEY = st.secrets["alpha_vantage"]["api_key"]
EXCHANGE_API_KEY = st.secrets["exchange_rate_api"]["api_key"]
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

# --- Caching & Data Functions ---
@st.cache_data(ttl=3600)
def get_stock_data_alpha(ticker):
    try:
        df, _ = ts.get_daily(symbol=ticker, outputsize='compact')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df = df.sort_index()

        # Existing indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['UpperBand'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
        df['LowerBand'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # --- New Indicators ---

        # EMA 12 and 26
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD and Signal line
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close_prev = (df['High'] - df['Close'].shift()).abs()
        low_close_prev = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # ADX (Average Directional Index)
        # Calculate Directional Movement
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr14 = df['ATR']  # already 14-day ATR computed above
        plus_di = 100 * (plus_dm.rolling(window=14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(window=14).sum() / tr14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=14).mean()

        # Stochastic Oscillator (14,3)
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
        df['%D'] = df['%K'].rolling(window=3).mean()

        return df.dropna()
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None

@st.cache_data(ttl=300)
def get_exchange_rate(from_currency, to_currency):
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/pair/{from_currency}/{to_currency}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["result"] == "success":
            return data["conversion_rate"]
        else:
            st.warning(f"Failed to get live rate for {from_currency}/{to_currency}: {data.get('error-type', 'Unknown error')}")
            return None
    except Exception as e:
        st.warning(f"Error fetching live exchange rate: {e}")
        return None

@st.cache_data(ttl=3600)
def get_forex_history_yf(from_currency, to_currency):
    ticker = f"{from_currency}{to_currency}=X"
    try:
        data = yf.download(ticker, period="90d", interval="1d", progress=False)
        if data.empty:
            return None
        df = data[['Close']].rename(columns={'Close': f"{from_currency}/{to_currency}"})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        st.warning(f"Error fetching forex history from yfinance: {e}")
        return None

# --- Chart Renderer ---
def plot_chart(df, title, chart_type, show_ma, show_bollinger, show_volume, key_prefix):
    fig = go.Figure()

    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='green')))

    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='Upper Band', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='Lower Band', line=dict(color='gray', dash='dot')))

    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightgray', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Volume'))

    fig.update_layout(title=title, height=400, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")

chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=1)

# Portfolio Metrics multiselect
selected_metrics = st.sidebar.multiselect(
    "Select portfolio metrics to display",
    options=[
        "Close Price", "MA20", "MA50", "UpperBand", "LowerBand",
        "RSI", "EMA12", "EMA26", "MACD", "MACD_Signal",
        "ADX", "%K (Stochastic)", "%D (Stochastic)", "ATR", "Volume"
    ],
    default=["Close Price", "RSI", "Volume"]
)

show_ma = any(m in selected_metrics for m in ["MA20", "MA50", "EMA12", "EMA26"])
show_bollinger = any(m in selected_metrics for m in ["UpperBand", "LowerBand"])
show_volume = "Volume" in selected_metrics

# Forex Metrics multiselect (separate)
st.sidebar.markdown("---")
st.sidebar.header("📉 Forex Metrics")
forex_metrics = st.sidebar.multiselect(
    "Select Forex chart metrics to display",
    options=[
        "Exchange Rate", "MA20", "Volatility (Std Dev)"
    ],
    default=["Exchange Rate"]
)

# --- TSX ETF Chart ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_volume, "tsx")

# --- Forex Section ---
st.header("💱 Forex Tracker")

st.subheader("USD/CAD Live Rate")
usd_cad = get_exchange_rate("USD", "CAD")
if usd_cad:
    st.metric(label="USD to CAD", value=f"{usd_cad:.4f}")
else:
    st.warning("Unable to fetch USD to CAD live rate.")

col1, col2 = st.columns(2)
with col1:
    fx_from = st.text_input("From Currency (e.g., EUR)", value="EUR").strip().upper()
with col2:
    fx_to = st.text_input("To Currency (e.g., USD)", value="USD").strip().upper()

if len(fx_from) != 3 or len(fx_to) != 3:
    st.error("Please enter valid 3-letter currency codes for 'From' and 'To' fields.")
else:
    rate = get_exchange_rate(fx_from, fx_to)
    if rate:
        st.metric(label=f"{fx_from} to {fx_to}", value=f"{rate:.4f}")
    else:
        st.warning(f"Unable to fetch live rate for {fx_from} to {fx_to}.")

    st.subheader("📉 Forex History (Last 90 Days)")
    forex_df = get_forex_history_yf(fx_from, fx_to)
    if forex_df is not None and not forex_df.empty:
        fig_fx = go.Figure()

        if "Exchange Rate" in forex_metrics:
            fig_fx.add_trace(go.Scatter(
                x=forex_df.index,
                y=forex_df.iloc[:, 0],
                mode='lines',
                name=f"{fx_from}/{fx_to} Rate"
            ))

        if "MA20" in forex_metrics:
            ma20 = forex_df.iloc[:, 0].rolling(window=20).mean()
            fig_fx.add_trace(go.Scatter(
                x=forex_df.index,
                y=ma20,
                mode='lines',
                name="MA20",
                line=dict(dash='dot')
            ))

        if "Volatility (Std Dev)" in forex_metrics:
            volatility = forex_df.iloc[:, 0].rolling(window=20).std()
            fig_fx.add_trace(go.Scatter(
                x=forex_df.index,
                y=volatility,
                mode='lines',
                name="Volatility (Std Dev)",
                line=dict(dash='dash')
            ))

        fig_fx.update_layout(
            title=f"{fx_from} to {fx_to} Exchange Rate (Last 90 Days)",
            height=400,
            xaxis_title="Date",
