import streamlit as st
import duckdb
from components.price_trends import show_price_trends
from components.flat_distribution import show_flat_distribution

def show():
    con = duckdb.connect("data/hdb_df_geocoded_condensed.duckdb")
    df = con.execute("""
        SELECT 
            month,
            flat_type,
            region,
            flat_model,
            resale_price,
            price_per_sqm,
        FROM resale
    """).df()

    # Components
    st.subheader("📈 Price Trends")
    show_price_trends(df)

    st.subheader("🏠 Flat Type Distribution")
    show_flat_distribution(df)
