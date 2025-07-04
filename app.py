import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")

# --- Caching & Data Functions ---
@st.cache_data(ttl=3600)
def get_stock_data_yf(ticker):
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        df.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'}, inplace=True)
        df = df.dropna()
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

        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Stochastic %K and %D
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
        df['%D'] = df['%K'].rolling(3).mean()

        # ADX
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr14 = df['ATR'] * 14  # True Range * 14 for ADX calculation approximation
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        df['ADX'] = dx.rolling(14).mean()

        return df.dropna()
    except Exception as e:
        st.error(f"Failed to load {ticker} data: {e}")
        return None

@st.cache_data(ttl=300)
def get_exchange_rate(from_currency, to_currency):
    url = f"https://v6.exchangerate-api.com/v6/YOUR_EXCHANGE_API_KEY/pair/{from_currency}/{to_currency}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["result"] == "success":
            return data["conversion_rate"]
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_forex_history_yf(from_currency, to_currency):
    try:
        pair = f"{from_currency}{to_currency}=X"
        data = yf.download(pair, period="3mo", progress=False)
        data = data.rename(columns={"Close": f"{from_currency}/{to_currency}"})
        return data[[f"{from_currency}/{to_currency}"]].dropna()
    except Exception as e:
        st.warning(f"Failed to get forex history for {from_currency}/{to_currency}: {e}")
        return None

# --- Chart Renderer ---
def plot_chart(df, title, chart_type, metrics_selected, key_prefix):
    fig = go.Figure()

    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

    if 'MA20' in metrics_selected:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
    if 'MA50' in metrics_selected:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='green')))
    if 'Bollinger Bands' in metrics_selected:
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='Upper Band', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='Lower Band', line=dict(color='gray', dash='dot')))
    if 'Volume' in metrics_selected:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightgray', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Volume'))

    fig.update_layout(title=title, height=400, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

def plot_macd(df, key_prefix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(dash='dash')))
    fig.update_layout(title="MACD", height=300, xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_macd")

def plot_adx(df, key_prefix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
    fig.update_layout(title="ADX", height=300, xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_adx")

def plot_stochastic(df, key_prefix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D', line=dict(dash='dash')))
    fig.update_layout(title="Stochastic Oscillator", height=300, xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_stochastic")

def plot_atr(df, key_prefix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR'))
    fig.update_layout(title="ATR", height=300, xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_atr")

def plot_cumulative_return(df, key_prefix):
    df['Cumulative Return'] = (1 + df['Close'].pct_change()).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative Return'], mode='lines', name='Cumulative Return'))
    fig.update_layout(title="Portfolio Cumulative Return", height=400, xaxis_title="Date", yaxis_title="Return")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_cumreturn")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=1)

stock_metrics_options = ['MA20', 'MA50', 'Bollinger Bands', 'Volume']
stock_metrics = st.sidebar.multiselect("Select Stock Metrics to display:", stock_metrics_options, default=['MA20', 'MA50', 'Volume'])

forex_metrics_options = ['MA20', 'MA50', 'Volume']
forex_metrics = st.sidebar.multiselect("Select Forex Metrics to display:", forex_metrics_options, default=['MA20'])

# --- TSX ETF Chart ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_yf("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite ETF (XIC.TO)", chart_type, stock_metrics, "tsx")

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
        plot_chart(forex_df, f"{fx_from} to {fx_to} Exchange Rate (Last 90 Days)", chart_type, forex_metrics, "forex")
    else:
        st.warning(f"No historical forex data available for {fx_from}/{fx_to}.")

# --- Portfolio Section ---
st.header("💼 Portfolio Metrics")
portfolio_input = st.text_input("Enter tickers with weights (e.g., SHOP.TO,10 ENB.TO,5 BNS.TO,20)")
portfolio = {}
if portfolio_input:
    try:
        for item in portfolio_input.split():
            ticker, weight = item.split(',')
            portfolio[ticker.strip().upper()] = float(weight)
    except:
        st.error("Invalid format. Use: TICKER,WEIGHT")

if portfolio:
    # Aggregate portfolio weighted close prices
    combined_prices = pd.DataFrame()
    for idx, (ticker, weight) in enumerate(portfolio.items()):
        data = get_stock_data_yf(ticker)
        if data is None:
            continue
        data_weighted = data['Close'] * weight
        combined_prices[ticker] = data_weighted

    # Plot combined portfolio price
    if not combined_prices.empty:
        price_fig = go.Figure()
        for ticker in combined_prices.columns:
            price_fig.add_trace(go.Scatter(x=combined_prices.index, y=combined_prices[ticker],
                                           name=f"{ticker} Weighted Close"))
        price_fig.update_layout(title="Portfolio Weighted Close Prices", height=400, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(price_fig, use_container_width=True)

    # Plot individual ticker indicators with metrics selected
    for ticker, weight in portfolio.items():
        data = get_stock_data_yf(ticker)
        if data is None:
            continue
        st.subheader(f"{ticker} Indicators")
        plot_chart(data, f"{ticker} Price Chart", chart_type, stock_metrics, f"{ticker}_chart")
        plot_macd(data, f"{ticker}_macd")
        plot_adx(data, f"{ticker}_adx")
        plot_stochastic(data, f"{ticker}_stochastic")
        plot_atr(data, f"{ticker}_atr")

    # Portfolio cumulative return
    portfolio_returns = combined_prices.sum(axis=1).pct_change()
    portfolio_cum_return = (1 + portfolio_returns).cumprod() - 1
    cum_return_fig = go.Figure()
    cum_return_fig.add_trace(go.Scatter(x=portfolio_cum_return.index, y=portfolio_cum_return, mode='lines', name='Portfolio Cumulative Return'))
    cum_return_fig.update_layout(title="Portfolio Cumulative Return", height=400, xaxis_title="Date", yaxis_title="Return")
    st.plotly_chart(cum_return_fig, use_container_width=True)

    # RSI Suggestions
    st.subheader("🧠 AI RSI-Based Suggestions")
    for ticker in portfolio:
        data = get_stock_data_yf(ticker)
        if data is not None and "RSI" in data.columns:
            latest_rsi = data["RSI"].iloc[-1]
            if latest_rsi < 30:
                st.info(f"{ticker} is **oversold** (RSI={latest_rsi:.2f}) → Potential Buy")
            elif latest_rsi > 70:
                st.warning(f"{ticker} is **overbought** (RSI={latest_rsi:.2f}) → Consider Reducing")
            else:
                st.success(f"{ticker} RSI is neutral ({latest_rsi:.2f})")
