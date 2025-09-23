import streamlit as st
import pandas as pd
import pydeck as pdk
import data.download_data  # ensures the CSV is downloaded

def show_map():
    # Load dataset
    df = pd.read_csv("data/hdb_resale_full.csv")

    # Keep only necessary columns for mapping
    map_df = df[["latitude", "longitude", "resale_price", "nearest_mrt_name", "nearest_schools_name"]].dropna()

    # --- Configure pydeck map ---
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_radius=50,
        get_fill_color=[149, 6, 6, 160],  # same dark red theme (#950606)
        pickable=True,
    )

    # View settings
    view_state = pdk.ViewState(
        latitude=1.3521,
        longitude=103.8198,
        zoom=11,
        pitch=0,
    )

    # Render map
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={
                "html": "<b>Price:</b> {resale_price}<br/><b>MRT:</b> {nearest_mrt_name}<br/><b>School:</b> {nearest_schools_name}",
                "style": {"color": "white"},
            },
        )
    )
