import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries

# --- Page config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("Canadian Market & Forex Dashboard")

# --- Load API keys securely ---
ALPHA_API_KEY = st.secrets["alpha_vantage"]["api_key"]
EXCHANGE_API_KEY = st.secrets["exchange_rate_api"]["api_key"]

ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

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
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading {ticker} from Alpha Vantage: {e}")
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
            st.error(f"Failed to get rate for {from_currency}/{to_currency}: {data.get('error-type')}")
            return None
    except Exception as e:
        st.error(f"Exception fetching rate for {from_currency}/{to_currency}: {e}")
        return None

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)

st.sidebar.markdown("📊 **Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_volume = st.sidebar.checkbox("Show Volume", False)
show_portfolio_metrics = st.sidebar.checkbox("Show Portfolio Metrics (RSI & Volume Combined)", True)

# --- Stock Ticker Selection ---
st.sidebar.header("🗂 Portfolio Settings")
portfolio_tickers_input = st.sidebar.text_area(
    "Enter your stock tickers separated by commas (e.g. SHOP.TO, ENB.TO, BNS.TO)",
    value="SHOP.TO, ENB.TO, BNS.TO"
)
portfolio_tickers = [t.strip().upper() for t in portfolio_tickers_input.split(",") if t.strip()]

# --- Plot function for stock charts ---
def plot_chart(df, title, chart_type, show_ma, show_bollinger, show_rsi, show_volume, key_prefix):
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candlestick'
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='green')))

    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='UpperBand', line=dict(dash='dot', color='gray')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='LowerBand', line=dict(dash='dot', color='gray')))

    if show_volume and 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightgray', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'))

    fig.update_layout(title=title, height=400, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_main")

    if show_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi.update_layout(title="RSI", height=200, yaxis_title='RSI')
        st.plotly_chart(fig_rsi, use_container_width=True, key=f"{key_prefix}_rsi")

# --- Portfolio Combined RSI & Volume Chart ---
def plot_portfolio_metrics(portfolio_data):
    fig = go.Figure()
    for ticker, df in portfolio_data.items():
        # Close price line (solid)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name=f"{ticker} Close"
        ))
        # RSI line (dashed)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], mode='lines', name=f"{ticker} RSI", line=dict(dash='dash')
        ))
        # Volume line (dotted)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Volume'], mode='lines', name=f"{ticker} Volume", line=dict(dash='dot')
        ))
    fig.update_layout(title="Portfolio Metrics: Close Price, RSI (dashed), Volume (dotted)",
                      height=450, xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True, key="portfolio_metrics")

# --- Forex Section ---

st.header("💱 Forex Dashboard")

# Fixed USD/CAD comparison
usd_cad_rate = get_exchange_rate("USD", "CAD")
if usd_cad_rate:
    st.metric("USD/CAD Exchange Rate", f"{usd_cad_rate:.4f}")
else:
    st.write("Failed to fetch USD/CAD rate")

# User selects forex pairs to compare
st.subheader("Compare Forex Pairs")
user_forex_from = st.text_input("From Currency (e.g. EUR)", "EUR").upper()
user_forex_to = st.text_input("To Currency (e.g. USD)", "USD").upper()

# Show rate for user selection
user_rate = get_exchange_rate(user_forex_from, user_forex_to)
if user_rate:
    st.metric(f"{user_forex_from}/{user_forex_to} Exchange Rate", f"{user_rate:.4f}")
else:
    st.write(f"Failed to fetch {user_forex_from}/{user_forex_to} rate")

# Historical data is not provided by exchangerate-api free tier, so skipping graphs for dynamic pairs

# --- TSX Composite Proxy Section ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite Proxy ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_rsi, show_volume, "tsx")
else:
    st.error("Failed to load TSX Composite data.")

# --- Portfolio Section ---
st.header("📊 Portfolio")

portfolio_data = {}
for ticker in portfolio_tickers:
    df = get_stock_data_alpha(ticker)
    if df is not None:
        portfolio_data[ticker] = df
    else:
        st.warning(f"Data for {ticker} not available.")

if portfolio_data:
    fig = go.Figure()
    for ticker, df in portfolio_data.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name=ticker
        ))
    fig.update_layout(title="Portfolio Close Prices", height=450, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True, key="portfolio_close_prices")

    if show_portfolio_metrics:
        plot_portfolio_metrics(portfolio_data)

# --- Simple Rule-Based LLM-Like Analysis ---
st.header("🤖 Portfolio Analysis")
analysis_msgs = []
for ticker, df in portfolio_data.items():
    latest_rsi = df['RSI'].iloc[-1]
    weekly_change = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
    msg = f"**{ticker}**: "
    if latest_rsi < 30:
        msg += "RSI below 30, stock might be oversold - consider buying. "
    elif latest_rsi > 70:
        msg += "RSI above 70, stock might be overbought - consider caution. "
    if weekly_change > 5:
        msg += "Price increased >5% in last week, trend is bullish. "
    elif weekly_change < -5:
        msg += "Price decreased >5% in last week, trend is bearish. "
    analysis_msgs.append(msg)
for msg in analysis_msgs:
    st.write(msg)
