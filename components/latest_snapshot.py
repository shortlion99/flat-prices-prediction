import streamlit as st
import pandas as pd

def latest_snapshot(df):
    """Render five KPI cards summarizing the most recent month of the market."""
    
    df["month"] = pd.to_datetime(df["month"])
    latest_month = df["month"].max()
    prev_month = latest_month - pd.DateOffset(months=1)

    latest_df = df[df["month"] == latest_month]
    prev_df = df[df["month"] == prev_month]

    # Median resale price
    median_price_latest = latest_df["resale_price"].median()
    median_price_prev = prev_df["resale_price"].median() if not prev_df.empty else None
    median_price_change = (
        (median_price_latest - median_price_prev) / median_price_prev * 100
        if median_price_prev and median_price_prev > 0
        else 0
    )

    # Transactions
    txn_latest = len(latest_df)
    txn_prev = len(prev_df) if not prev_df.empty else None
    txn_change = (
        (txn_latest - txn_prev) / txn_prev * 100
        if txn_prev and txn_prev > 0
        else 0
    )

    # Price Index (2017=100 baseline)
    baseline_year = 2017
    base_price = df[df["month"].dt.year == baseline_year]["resale_price"].median()
    price_index_latest = (
        median_price_latest / base_price * 100 if base_price else None
    )
    price_index_change = (
        (price_index_latest - 100) if price_index_latest else None
    )

    # Most expensive region (by median resale price in latest month)
    region_stats = (
        latest_df.groupby("region")["resale_price"]
        .median()
        .sort_values(ascending=False)
    )
    most_exp_region = region_stats.index[0]
    most_exp_price = region_stats.iloc[0]

    # Most popular flat type (by count in latest month)
    flat_type_stats = (
        latest_df["flat_type"].value_counts(normalize=True)
        .mul(100)
        .round(1)
    )
    most_pop_type = flat_type_stats.index[0]
    most_pop_share = flat_type_stats.iloc[0]


    # --- Market Snapshot Section ---
    with st.container(border=True):
        st.subheader(f"📅 Market Snapshot – {latest_month.strftime('%B %Y')}")
        st.caption("All figures below reflect the most recent month compared with the previous month.")

        cols = st.columns(5)

        with cols[0]:
            st.metric(
                label="**Median Resale Price**",
                value=f"${median_price_latest:,.0f}",
                delta=f"{median_price_change:+.1f}% MoM",
            )

        with cols[1]:
            st.metric(
                label="**Transactions**",
                value=f"{txn_latest:,}",
                delta=f"{txn_change:+.1f}% MoM"
            )

        with cols[2]:
            st.metric(
                label="**Price Index (2017=100)**",
                value=f"{price_index_latest:.0f}",
                delta=f"{price_index_change:+.1f}%"
            )

        with cols[3]:
            st.metric(
                label="**Most Expensive Region**",
                value=most_exp_region,
                delta=f"${most_exp_price:,.0f}"
            )

        with cols[4]:
            st.metric(
                label="**Most Popular Flat Type**",
                value=most_pop_type,
                delta=f"{most_pop_share:.1f}% of sales"
            )
