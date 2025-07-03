import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import requests

st.set_page_config(page_title="Canadian Market Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")

# === Yahoo Finance Stock Data ===
@st.cache_data(ttl=3600)
def get_stock_data_yf(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
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
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading {ticker} from Yahoo Finance: {e}")
        return None

# === Chart Plotting ===
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

# === Sidebar Settings ===
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)

st.sidebar.markdown("📊 **Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_volume = st.sidebar.checkbox("Show Volume", False)

# === TSX Composite Proxy ===
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_yf("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite Proxy ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_rsi, show_volume, "tsx")

# === Forex Dashboard ===
st.header("💱 Forex Dashboard")

exchange_rate_api_key = "ce58044a6513892a421f6b4e"
base_url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/pair"

forex_pairs = [
    ("USD", "CAD"),
    ("EUR", "USD"),
    ("GBP", "USD"),
    ("USD", "JPY")
]

cols = st.columns(len(forex_pairs))

for i, (base, target) in enumerate(forex_pairs):
    with cols[i]:
        st.subheader(f"{base}/{target}")
        try:
            url = f"{base_url}/{base}/{target}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                rate = data['conversion_rate']
                st.metric(label="Exchange Rate", value=f"{rate:.4f}")
            else:
                st.error("Failed to fetch rate")
        except Exception as e:
            st.error(f"Error: {e}")
