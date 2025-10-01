import streamlit as st
import duckdb
import pydeck as pdk
import pandas as pd

@st.cache_data
def load_map_data(limit=20000):
    con = duckdb.connect("data/hdb_df_geocoded_condensed.duckdb")
    query = f"""
        SELECT 
            latitude, 
            longitude, 
            resale_price/1000 as resale_price,
            ROUND(price_per_sqm,2) AS price_per_sqm,
            flat_type,
            flat_model,
            floor_area_sqm,
            storey_range,
            region,
            month,
            district_number,
            nearest_mrt_name,
            ROUND(nearest_mrt_distance_km,2) AS nearest_mrt_distance_km,
            nearest_schools_name,
            ROUND(nearest_schools_distance_km,2) AS nearest_schools_distance_km
        FROM resale
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        LIMIT {limit}
    """
    df = con.execute(query).df()
    con.close()
    return df

def show_map():
    with st.container(border=True):
        st.markdown("## 🗺️ Map of Resale Transactions")
        st.caption("Scatterplot shows individual transactions as dots. Heatmap view highlights price intensity across regions.")

        col1, col2 = st.columns([1, 2])
        with col1:
            layer_type = st.radio("", ["Scatterplot", "Heatmap"], horizontal=True)

        with col2:
            limit = st.slider("Transactions displayed", 5000, 200000, 20000, step=10000)

        map_df = load_map_data(limit=limit)
        # Normalize resale_price
        price_min, price_max = map_df["resale_price"].min(), map_df["resale_price"].max()
        map_df["price_norm"] = (map_df["resale_price"] - price_min) / (price_max - price_min)
        map_df["month"] = pd.to_datetime(map_df["month"], errors="coerce").dt.strftime("%b %Y")
        
        # --- Scatter layer ---
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_radius=60,
            get_fill_color=[149, 6, 6, 160],  # dark red theme (#950606)
            pickable=True,
        )

        # --- Heatmap layer ---
        heatmap = pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_weight="resale_price",
            radius_pixels=40,
        )

        layers = [scatter] if layer_type == "Scatterplot" else [heatmap]

        # --- View ---
        view_state = pdk.ViewState(
            latitude=1.3521,
            longitude=103.8198,
            zoom=11.2,
            pitch=0,
        )

        # --- Render ---
        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=view_state,
                layers=layers,
                tooltip = {
                    "html": """
                        <div style="font-size: 13px; padding: 6px; color: #222;">
                            <b>📅 Transaction Month:</b> {month} <br/>
                            <b>💰 Price:</b> ${resale_price}k <br/>
                            <b>📐 Price per sqm:</b> ${price_per_sqm} <br/>
                            <b>🏠 Flat:</b> {flat_type}, {flat_model}, {floor_area_sqm} sqm <br/>
                            <b>🏢 Storey:</b> {storey_range} <br/>
                            <b>📍 Location:</b> {region}, District {district_number} <br/>
                            <b>🚇 Nearest MRT:</b> {nearest_mrt_name} ({nearest_mrt_distance_km} km) <br/>
                            <b>🏫 Nearest School:</b> {nearest_schools_name} ({nearest_schools_distance_km} km)
                        </div>
                    """,
                    "style": {
                        "backgroundColor": "rgba(255, 255, 255, 0.9)",
                        "color": "black",
                        "border-radius": "6px"
                    },
                }
            )
        )
