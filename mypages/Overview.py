import streamlit as st
import duckdb
import pandas as pd
from components.price_trends import show_price_trends
from components.flat_distribution import show_flat_distribution
from components.latest_snapshot import latest_snapshot
from components.map_overview import show_map

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


    with st.container():
        latest_snapshot(df)
        col1, col2 = st.columns([2, 1]) 
        with col1:
            show_price_trends(df)
        with col2:
            show_flat_distribution(df)

        show_map()
