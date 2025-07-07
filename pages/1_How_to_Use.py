import streamlit as st

st.set_page_config(page_title="📘 How to Use", layout="wide")
st.title("📘 How to Use This Dashboard")
st.markdown("---")

st.markdown("""
### 🧭 Navigation Overview

Use the sidebar to access:
- `Home` → Dashboard introduction
- `Portfolio` → Multi-ticker metrics & AI analysis
- `Sentiment` → Keyword-based sentiment & suggestions
- `Forex` → Real-time and historical FX analysis

---

### 📉 TSX Composite ETF (XIC.TO)

- Visualizes daily trends using **candlestick or line charts**
- Customize with **Moving Averages**, **Bollinger Bands**, or **Volume**

---

### 💱 Forex Tracker

- Default: USD/CAD live exchange rate
- Enter custom currency pairs (e.g., EUR to GBP)
- View **90-day historical FX chart**

---

### 💼 Portfolio Section

1. **Enter tickers & weights:**  
   Format → `SHOP.TO,10 ENB.TO,5 BNS.TO,20`
2. **Choose metrics to plot:**  
   - Close Price  
   - MA50  
   - RSI  
   - Volume  
   - Upper Bollinger Band
3. View cumulative return chart & RSI-based recommendations

---

### 🧠 News Sentiment

- Enter a **company name** or **sector keyword** (e.g., "Tesla", "Banking")
- The app fetches news headlines & analyzes their sentiment
- Returns a **summary recommendation** (positive/neutral/negative)

---

### 🧰 Tips

- Use tickers from **Yahoo Finance** (e.g., `AAPL`, `ENB.TO`)
- Use keywords like `"AI"`, `"renewables"`, `"oil"` for sentiment
- Hover over charts for tooltips and time-series values

---

### 📌 Future Enhancements (Planned)

- Real-time charting with WebSocket feeds
- Transformer-based NLP for deeper sentiment analysis
- Portfolio Sharpe ratio, volatility, and drawdown
- User login with saved portfolios

---

Still stuck or want to contribute?  
📬 Reach out: **your.email@example.com**
""")
