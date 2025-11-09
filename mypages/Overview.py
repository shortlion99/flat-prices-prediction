import streamlit as st
from data.data_access import get_duckdb_conn
from components.price_trends import show_price_trends
from components.flat_distribution import show_flat_distribution
from components.latest_snapshot import latest_snapshot
from components.map_overview import show_map
from components.resale_price_distribution import show_resale_price_distribution
from components.resale_price_relationships import show_relationship

@st.cache_data(show_spinner=False)
def load_data():
    con = get_duckdb_conn()
    return con.execute("""
        SELECT 
            month, town, flat_type, region, flat_model,
            resale_price, price_per_sqm, district_number,
            nearest_mrt_distance_km, nearest_supermarkets_distance_km,
            nearest_schools_distance_km, floor_area_sqm,
            remaining_lease, storey_mid
        FROM resale
    """).df()

def show():
    df = load_data()

    with st.container():
        st.title("📊 Market Overview")
        st.caption(
            "Explore historical market trends, price distributions, and geographic insights across Singapore's HDB resale market."
        )
        latest_snapshot(df) # Market Snapshot
        col1, col2 = st.columns([2, 1]) 
        with col1:
            show_price_trends(df) # Historical Price Trends
        with col2:
            show_flat_distribution(df) # Flat Type Distribution

        show_map() # Map of Resale Transactions
        show_resale_price_distribution(df) # Mean Resale Price Analysis
        show_relationship(df) # Variable Impact on Resale Price
