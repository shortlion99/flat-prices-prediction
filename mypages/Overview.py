import streamlit as st
import duckdb
import pandas as pd
from components.price_trends import show_price_trends
from components.flat_distribution import show_flat_distribution
from components.latest_snapshot import latest_snapshot

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
    latest_snapshot(df)

    show_price_trends(df)

    show_flat_distribution(df)
