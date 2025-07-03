import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries

# --- Page config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")

# --- Load API keys securely ---
API_KEY = st.secrets["exchange_rate_api"]["api_key"]
ALPHA_API_KEY = st.secrets.get("alpha_vantage", {}).get("api_key", "")
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
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)

st.sidebar.markdown("📊 **Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_volume = st.sidebar.checkbox("Show Volume", False)

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

# --- TSX Composite Proxy Section ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite Proxy ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_rsi, show_volume, "tsx")
else:
    st.error("Failed to load TSX Composite data.")

# --- Forex Dashboard Section ---
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
