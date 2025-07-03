import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries

# --- Page config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("\ud83d\udcc8 Canadian Market & Forex Dashboard")

# --- Load API keys securely ---
API_KEY = st.secrets["exchange_rate_api"]["api_key"]
ALPHA_API_KEY = st.secrets.get("alpha_vantage", {}).get("api_key", "")
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

# --- Get TSX Stock Data ---
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

# --- Get Exchange Rate ---
@st.cache_data(ttl=300)
def get_exchange_rate(from_currency, to_currency):
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{from_currency}/{to_currency}"
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
st.sidebar.header("\u2699\ufe0f Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=1)

st.sidebar.markdown("\ud83d\udcca **Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_volume = st.sidebar.checkbox("Show Volume", False)
show_portfolio_metrics = st.sidebar.checkbox("Show Portfolio Metrics", True)

# --- Chart Plotting Function ---
def plot_chart(df, title, chart_type, show_ma, show_bollinger, show_volume, key_prefix):
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Candlestick'))
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

# --- TSX Composite Chart ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite Proxy ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_volume, "tsx")

# --- Forex Section ---
st.header("\ud83d\udcb1 Forex Tracker")

st.subheader("USD/CAD Live Exchange Rate")
usd_cad = get_exchange_rate("USD", "CAD")
if usd_cad:
    st.metric(label="USD to CAD", value=f"{usd_cad:.4f}")

st.subheader("Dynamic Forex Comparison")
col1, col2 = st.columns(2)
with col1:
    fx_from = st.text_input("From Currency (e.g., EUR)", value="EUR")
with col2:
    fx_to = st.text_input("To Currency (e.g., USD)", value="USD")

rate = get_exchange_rate(fx_from.upper(), fx_to.upper())
if rate:
    st.metric(label=f"{fx_from.upper()} to {fx_to.upper()}", value=f"{rate:.4f}")

# --- Portfolio Section ---
st.header("\ud83d\udcbc Portfolio Metrics")
portfolio_input = st.text_input("Enter tickers with weights (e.g., SHOP.TO,10 ENB.TO,5 BNS.TO,20)")
portfolio = {}
if portfolio_input:
    try:
        for item in portfolio_input.split():
            ticker, weight = item.split(',')
            portfolio[ticker.strip().upper()] = float(weight)
    except:
        st.error("Invalid input format. Use: TICKER,WEIGHT")

if portfolio:
    portfolio_df = pd.DataFrame()
    rsi_df = pd.DataFrame()
    volume_df = pd.DataFrame()

    for ticker, weight in portfolio.items():
        data = get_stock_data_alpha(ticker)
        if data is not None:
            portfolio_df[ticker] = data['Close'] * weight
            rsi_df[ticker] = data['RSI']
            volume_df[ticker] = data['Volume']

    st.subheader("\ud83d\udd22 Combined Close Price (Weighted)")
    fig = go.Figure()
    for col in portfolio_df.columns:
        fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df[col], mode='lines', name=col))
    fig.update_layout(title="Portfolio Close Prices", height=400, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI & Volume (per stock)")
    fig_rsi = go.Figure()
    for col in rsi_df.columns:
        fig_rsi.add_trace(go.Scatter(x=rsi_df.index, y=rsi_df[col], name=f"{col} RSI", line=dict(dash='dot')))
    fig_rsi.update_layout(title="Portfolio RSI", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_vol = go.Figure()
    for col in volume_df.columns:
        fig_vol.add_trace(go.Scatter(x=volume_df.index, y=volume_df[col], name=f"{col} Volume", line=dict(dash='solid')))
    fig_vol.update_layout(title="Portfolio Volume", height=300)
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("\ud83e\uddd1\u200d\ud83e\udd1d AI-Driven Portfolio Suggestions")
    for ticker in rsi_df.columns:
        latest_rsi = rsi_df[ticker].iloc[-1]
        if latest_rsi < 30:
            st.info(f"{ticker} is oversold (RSI={latest_rsi:.2f}) - Potential Buy")
        elif latest_rsi > 70:
            st.warning(f"{ticker} is overbought (RSI={latest_rsi:.2f}) - Consider Reducing")
        else:
            st.success(f"{ticker} RSI is Neutral ({latest_rsi:.2f})")

