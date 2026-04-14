# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math
import time

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker_text = st.sidebar.text_input(
    "Enter 2 to 5 tickers (comma-separated)",
    value="AAPL, MSFT"
)

tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))  # remove duplicates, keep order

if len(tickers) < 2 or len(tickers) > 5:
    st.sidebar.error("Please enter between 2 and 5 ticker symbols.")
    st.stop()

# Default date range: one year back from today
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))
run_analysis = st.sidebar.button("Run Analysis")

# Validate that the date range makes sense
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < 365:
    st.sidebar.error("Please select a date range of at least 1 year.")
    st.stop()

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(tickers: list[str], start: date, end: date):
    benchmark = "^GSPC"
    all_symbols = tickers + [benchmark]

    data = {}
    failed = []

    for symbol in all_symbols:
        success = False

        for attempt in range(3):
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )

                if not df.empty and "Adj Close" in df.columns:
                    data[symbol] = df["Adj Close"]
                    success = True
                    time.sleep(1)
                    break

            except Exception:
                pass

            time.sleep(2 * (attempt + 1))

        if not success:
            failed.append(symbol)

    return data, failed

@st.cache_data(show_spinner="Calculating return analytics...", ttl=3600)
def compute_price_return_analysis(prices: pd.DataFrame, user_tickers: list[str]):
    returns = prices.pct_change().dropna()

    stats = pd.DataFrame({
        "Annualized Mean Return": returns.mean() * 252,
        "Annualized Volatility": returns.std() * math.sqrt(252),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurtosis(),
        "Min Daily Return": returns.min(),
        "Max Daily Return": returns.max(),
    })

    selected_returns = returns[user_tickers]
    equal_weight_return = selected_returns.mean(axis=1)
    equal_weight_wealth = 10000 * (1 + equal_weight_return).cumprod()

    wealth = 10000 * (1 + returns).cumprod()
    wealth["Equal-Weight Portfolio"] = equal_weight_wealth

    return returns, stats, wealth

# -- Main logic -------------------------------------------
if run_analysis and tickers:
    try:
        data_dict, failed_tickers = load_data(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

    if failed_tickers:
        st.error(
            "These ticker(s) failed to download or had insufficient data: "
            + ", ".join(failed_tickers)
        )
        st.warning(
            "Yahoo Finance may be rate-limiting requests right now. "
            "This can happen on Streamlit Cloud when many apps share the same outbound IP."
        )

    if not data_dict:
        st.stop()

    prices = pd.concat(data_dict, axis=1)
    prices.columns = prices.columns.get_level_values(0)

    missing_pct = prices.isna().mean()
    drop_cols = [col for col in prices.columns if missing_pct[col] > 0.05 and col != "^GSPC"]

    if drop_cols:
        st.warning(
            "Dropped ticker(s) with more than 5% missing values: "
            + ", ".join(drop_cols)
        )
        prices = prices.drop(columns=drop_cols)

    rows_before = len(prices)
    prices = prices.dropna()
    rows_after = len(prices)

    if rows_after < rows_before:
        st.info("Data was truncated to the overlapping date range across tickers.")

    usable_tickers = [t for t in tickers if t in prices.columns]
    returns, stats_df, wealth_df = compute_price_return_analysis(prices, usable_tickers)
    if len(usable_tickers) < 2:
        st.error("At least 2 valid stock tickers with sufficient data are required.")
        st.stop()


    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price & Returns", "Risk & Distribution", "Correlation & Portfolio", "About"]
    )

    with tab1:
        st.subheader("Price and Return Analysis")

    selected_series = st.multiselect(
        "Select stocks to show on charts",
        options=list(prices.columns),
        default=list(prices.columns)
    )

    st.markdown("### Summary Statistics")
    st.dataframe(
        stats_df.style.format({
            "Annualized Mean Return": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Skewness": "{:.3f}",
            "Kurtosis": "{:.3f}",
            "Min Daily Return": "{:.2%}",
            "Max Daily Return": "{:.2%}",
        }),
        use_container_width=True
    )

    st.markdown("### Adjusted Closing Prices")
    price_fig = go.Figure()
    for col in selected_series:
        price_fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices[col],
                mode="lines",
                name=col
            )
        )

    price_fig.update_layout(
        title="Adjusted Closing Prices",
        xaxis_title="Date",
        yaxis_title="Adjusted Close Price (USD)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(price_fig, use_container_width=True)

    st.markdown("### Daily Returns")
    st.dataframe(returns[selected_series].tail(), use_container_width=True)

    st.markdown("### Cumulative Wealth Index ($10,000 Initial Investment)")
    wealth_fig = go.Figure()
    wealth_to_plot = [c for c in selected_series if c in wealth_df.columns]

    if "Equal-Weight Portfolio" not in wealth_to_plot:
        wealth_to_plot.append("Equal-Weight Portfolio")

    for col in wealth_to_plot:
        wealth_fig.add_trace(
            go.Scatter(
                x=wealth_df.index,
                y=wealth_df[col],
                mode="lines",
                name=col
            )
        )

    wealth_fig.update_layout(
        title="Growth of $10,000",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(wealth_fig, use_container_width=True)
    with tab2:
        st.subheader("Risk & Distribution")
        st.info("This section will include rolling volatility, histogram, Q-Q plot, Jarque-Bera test, and box plot.")

    with tab3:
        st.subheader("Correlation & Portfolio")
        st.info("This section will include the correlation heatmap, scatter plot, rolling correlation, and two-asset portfolio explorer.")

    with tab4:
        st.subheader("About / Methodology")
        st.markdown(
            """
            This app compares multiple stocks using adjusted close prices from Yahoo Finance.

            **Key assumptions**
            - Daily returns are simple arithmetic returns using `pct_change()`
            - Annualized return uses 252 trading days
            - Annualized volatility uses daily standard deviation times sqrt(252)
            - Benchmark: S&P 500 (`^GSPC`)
            """
        )

else:
    st.info("Enter 2 to 5 stock tickers, choose dates, and click Run Analysis.")