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
    if len(usable_tickers) < 2:
        st.error("At least 2 valid stock tickers with sufficient data are required.")
        st.stop()

    returns = prices.pct_change()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price & Returns", "Risk & Distribution", "Correlation & Portfolio", "About"]
    )

    with tab1:
        summary_rows = []
        for symbol in usable_tickers:
            latest_close = float(prices[symbol].iloc[-1])
            total_return = float((prices[symbol].iloc[-1] / prices[symbol].iloc[0]) - 1)
            volatility = float(returns[symbol].std())
            ann_volatility = volatility * math.sqrt(252)
            max_close = float(prices[symbol].max())
            min_close = float(prices[symbol].min())

            summary_rows.append({
                "Ticker": symbol,
                "Latest Close": f"${latest_close:,.2f}",
                "Total Return": f"{total_return:.2%}",
                "Annualized Volatility": f"{ann_volatility:.2%}",
                "Period High": f"${max_close:,.2f}",
                "Period Low": f"${min_close:,.2f}",
            })

        summary_df = pd.DataFrame(summary_rows)

        st.subheader("Key Metrics")
        st.dataframe(summary_df, use_container_width=True)

        selected_series = st.multiselect(
            "Select stocks to display on the price chart",
            options=list(prices.columns),
            default=list(prices.columns)
        )

        st.subheader("Adjusted Closing Prices")

        fig = go.Figure()
        for col in selected_series:
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1.5)
                )
            )

        fig.update_layout(
            title="Adjusted Closing Prices",
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
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