import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Title of the app
st.set_page_config(page_title="Risk-Parity Portfolio Dashboard", layout="wide")
st.title("📈 Risk-Parity Portfolio Dashboard")

# Load S&P 500 tickers and names with additional assets
@st.cache_data
def get_assets():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0][["Symbol", "Security"]]
    sp500_dict = {row["Symbol"]: f"{row['Symbol']} - {row['Security']}" for _, row in sp500_df.iterrows()}
    
    # Add additional assets (Gold, Oil, Bonds, Dollar Index)
    extra_assets = {
        "GC=F": "GC=F - Gold Futures",
        "ZN=F": "ZN=F - 10-Year Treasury Note Futures",
        "DX=F": "DX=F - US Dollar Index Futures"
    }
    
    sp500_dict.update(extra_assets)
    return sp500_dict

assets_dict = get_assets()

# Sidebar for user input
st.sidebar.header("Customize Portfolio")
selected_tickers = st.sidebar.multiselect("Select stocks and assets", list(assets_dict.values()), 
                                          default=[assets_dict["AAPL"], assets_dict["GC=F"], assets_dict["DX=F"], assets_dict["ZN=F"]])

# Convert selected values back to tickers
selected_tickers = [key for key, value in assets_dict.items() if value in selected_tickers]

# Backtesting start year selector
start_year = st.sidebar.slider("Select Backtesting Start Year", min_value=2001, max_value=2024, value=2010)

# Ensure at least one ticker is selected
if not selected_tickers:
    st.warning("Please select at least one asset.")
    st.stop()

# Download data
data = yf.download(selected_tickers, start=f"{start_year}-01-01")
data.ffill(inplace=True)
data.dropna(inplace=True)
prices = data["Close"].resample("ME").last()

# Risk-free rate
risk_free_rate = yf.download("^IRX", start=f"{start_year}-01-01")["Close"]
risk_free_rate = (risk_free_rate / 100) / 12
risk_free_rate = risk_free_rate.reindex(prices.index, method='ffill')
avg_risk_free_rate = risk_free_rate.mean().iloc[0]

# Log returns
log_returns = np.log(prices).pct_change().dropna()

# Rolling volatility & inverse volatility
window_size = st.sidebar.slider("Rolling Window Size (Months)", min_value=12, max_value=60, value=36, step=6)
rolling_vol = log_returns.rolling(window_size).std().replace(0, np.nan)
rolling_vol.ffill(inplace=True)
rolling_inverse_vol = 1 / rolling_vol
risk_parity_weights = rolling_inverse_vol.apply(lambda column: column / rolling_inverse_vol.sum(axis=1))
risk_parity_weights = risk_parity_weights.shift(1)
portfolio_returns = (log_returns * risk_parity_weights).sum(axis=1)
cum_returns_portfolio = (1 + portfolio_returns).cumprod()

# Drawdowns
cumulative_max = cum_returns_portfolio.cummax()
drawdown = (cum_returns_portfolio / cumulative_max) - 1
max_drawdown = drawdown.min()
rolling_drawdown = drawdown.rolling(window_size, min_periods=1).min()

# Layout for better visual presentation
col1, col2 = st.columns(2)

# Cumulative Returns Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=cum_returns_portfolio.index, y=cum_returns_portfolio - 1,
                         mode='lines', name='Cumulative Returns', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=rolling_drawdown.index, y=rolling_drawdown,
                         mode='lines', name='Rolling Drawdowns', line=dict(color='purple', dash='dot')))
fig.update_layout(title="Cumulative Returns & Drawdowns", xaxis_title="Date", yaxis_title="Return",
                  template="plotly_white")
col1.plotly_chart(fig, use_container_width=True)

# Display Drawdowns
fig_dd = px.area(drawdown, title="Drawdown Over Time", labels={"value": "Drawdown"}, template="plotly_white")
col2.plotly_chart(fig_dd, use_container_width=True)

# Portfolio Weights
fig_w = px.line(risk_parity_weights, title="Risk-Parity Weights Over Time", template="plotly_white")
st.plotly_chart(fig_w, use_container_width=True)

# Performance Metrics
st.subheader("📊 Performance Metrics")
col3, col4 = st.columns(2)

with col3:
    st.markdown(f"**📈 Annualized Return:** {portfolio_returns.mean() * window_size * 100:.2f}%")
    st.markdown(f"**📊 Annualized Volatility:** {portfolio_returns.std() * np.sqrt(window_size) * 100:.2f}%")
    st.markdown(f"**📌 Sharpe Ratio:** {(portfolio_returns.mean() * window_size - avg_risk_free_rate) / (portfolio_returns.std() * np.sqrt(window_size)):.2f}")
    st.markdown(f"**🔄 Sortino Ratio:** {(portfolio_returns.mean() * window_size - avg_risk_free_rate) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(window_size)):.2f}")

with col4:
    st.markdown(f"**📉 Calmar Ratio:** {(portfolio_returns.mean() * window_size) / abs(max_drawdown):.2f}")
    st.markdown(f"**📉 Max Drawdown:** {max_drawdown:.2%}")
    st.markdown("**🏦 Total Debt:** Data Unavailable")
    st.markdown("**📊 Dividend Yield:** Data Unavailable")

# 📌 Footer
st.markdown("""
    ---
    📌 **Risk Parity Portfolio Creation** | Created by [Matthieu Lombardo](https://www.linkedin.com/in/matthieu-lombardo)
""", unsafe_allow_html=True)
