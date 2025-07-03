import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries

# --- API keys ---
ALPHA_API_KEY = st.secrets.get("alpha_vantage", {}).get("api_key", "")
EXCHANGE_API_KEY = st.secrets.get("exchange_rate_api", {}).get("api_key", "")

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
        # Calculate moving averages & Bollinger Bands for close price
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['UpperBand'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
        df['LowerBand'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()

        # RSI calculation
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

def plot_tsx_chart(df, chart_type, show_ma, show_bollinger, show_rsi, show_volume):
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

    fig.update_layout(title="TSX Composite Proxy ETF (XIC.TO)", height=400, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi.update_layout(title="RSI", height=200, yaxis_title='RSI')
        st.plotly_chart(fig_rsi, use_container_width=True)

def plot_portfolio_combined_chart(dfs, tickers, show_ma, show_bollinger):
    fig = go.Figure()
    for ticker in tickers:
        df = dfs.get(ticker)
        if df is None:
            continue
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f"{ticker} Close"))
        if show_ma:
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name=f"{ticker} MA20", line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name=f"{ticker} MA50", line=dict(dash='dot')))
        if show_bollinger:
            fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name=f"{ticker} UpperBand", line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name=f"{ticker} LowerBand", line=dict(dash='dash')))

    fig.update_layout(
        title="Portfolio Stocks Close Price Comparison",
        height=500,
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Ticker / Metric",
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_rsi_volume_chart(dfs, tickers):
    fig = go.Figure()
    # Add RSI traces on left y-axis
    for ticker in tickers:
        df = dfs.get(ticker)
        if df is None:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], mode='lines', name=f"{ticker} RSI",
            yaxis='y1'
        ))

    # Add Volume bars on right y-axis
    for ticker in tickers:
        df = dfs.get(ticker)
        if df is None:
            continue
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name=f"{ticker} Volume", opacity=0.3,
            yaxis='y2'
        ))

    fig.update_layout(
        title="Portfolio RSI and Volume",
        height=400,
        yaxis=dict(
            title="RSI",
            range=[0, 100],
            side='left'
        ),
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right',
            showgrid=False,
            position=0.85
        ),
        xaxis_title="Date",
        legend_title="Ticker / Metric",
        barmode='overlay'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)

st.sidebar.markdown("📊 **Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", False)  
show_volume = st.sidebar.checkbox("Show Volume", False)

# Portfolio input & multi-select
st.sidebar.header("📁 Portfolio Input")
portfolio_input = st.sidebar.text_area(
    "Enter your portfolio (Ticker:Weight comma-separated)",
    value="SHOP.TO:0.4, ENB.TO:0.3, BNS.TO:0.3"
)

portfolio = {}
try:
    for item in portfolio_input.split(","):
        ticker, weight = item.strip().split(":")
        portfolio[ticker.strip().upper()] = float(weight.strip())
except Exception:
    st.sidebar.error("Invalid portfolio input format. Use TICKER:WEIGHT, separated by commas.")

selected_tickers = st.sidebar.multiselect(
    "Select tickers to compare",
    options=list(portfolio.keys()),
    default=list(portfolio.keys())
)

st.title("📈 Canadian Market & Forex Dashboard")

tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_tsx_chart(tsx_data, chart_type, show_ma, show_bollinger, show_rsi, show_volume)
else:
    st.error("Failed to load TSX Composite data.")

portfolio_dfs = {}
for ticker in selected_tickers:
    df = get_stock_data_alpha(ticker)
    if df is not None:
        portfolio_dfs[ticker] = df
    else:
        st.warning(f"Failed to load data for {ticker}")

if portfolio_dfs:
    plot_portfolio_combined_chart(portfolio_dfs, selected_tickers, show_ma, show_bollinger)
    if show_rsi or show_volume:
        plot_portfolio_rsi_volume_chart(portfolio_dfs, selected_tickers)
else:
    st.info("No portfolio tickers selected or data unavailable.")

# Forex dashboard
st.header("💱 Forex Dashboard")
forex_pairs = [
    ("USD", "CAD"),
    ("EUR", "USD"),
    ("GBP", "USD"),
    ("USD", "JPY")
]

cols = st.columns(len(forex_pairs))
for i, (from_curr, to_curr) in enumerate(forex_pairs):
    with cols[i]:
        st.subheader(f"{from_curr}/{to_curr}")
        rate = get_exchange_rate(from_curr, to_curr)
        if rate is not None:
            st.metric(label="Exchange Rate", value=f"{rate:.4f}")
        else:
            st.write("Failed to fetch rate")
