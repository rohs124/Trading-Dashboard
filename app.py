
import streamlit as st

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")
st.markdown("---")

st.markdown("""
### 👋 Welcome!

This dashboard is a one-stop platform for **Canadian stock analysis**, **portfolio monitoring**, and **forex tracking**.

Designed for:
- 💼 Retail & institutional traders
- 📊 Financial analysts
- 📈 Data-driven investors

---

### 🔍 What This Dashboard Does

- Visualizes **TSX Composite** index proxy (XIC.TO)
- Tracks **portfolio performance** with RSI-based signals
- Provides live & historical **forex rates**
- Performs **news-based sentiment analysis** using NLP

---

### 🔗 Data Sources

- 📡 [Alpha Vantage](https://www.alphavantage.co/)
- 🧠 [Yahoo Finance](https://finance.yahoo.com/)
- 💱 [ExchangeRate.host](https://exchangerate.host/)
- 📰 News via `yfinance.Ticker().news`

---

> Use the sidebar to explore the different tools built into this dashboard.
""")
