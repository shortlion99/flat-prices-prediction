import streamlit as st
import duckdb
import pandas as pd
from components.price_trends import show_price_trends
from components.flat_distribution import show_flat_distribution
from components.latest_snapshot import latest_snapshot
from components.map_overview import show_map
from components.resale_price_distribution import show_resale_price_distribution
from components.resale_price_relationships import show_relationship

def show():
    con = duckdb.connect("data/hdb_df_geocoded_condensed.duckdb")
    df = con.execute("""
        SELECT 
            month,
            town,
            flat_type,
            region,
            flat_model,
            resale_price,
            price_per_sqm,
            district_number,
            nearest_mrt_distance_km,
            nearest_supermarkets_distance_km,
            nearest_schools_distance_km,
            floor_area_sqm,
            remaining_lease,
            storey_mid
        FROM resale
    """).df()


    with st.container():
        st.title("📊 Market Overview")
        st.caption(
            "Explore historical market trends, price distributions, and geographic insights across Singapore's HDB resale market."
        )
        latest_snapshot(df)
        col1, col2 = st.columns([2, 1]) 
        with col1:
            show_price_trends(df)
        with col2:
            show_flat_distribution(df)

        show_map()
        show_resale_price_distribution(df)
        show_relationship(df)
