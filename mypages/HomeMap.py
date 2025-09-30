import streamlit as st
import duckdb
import pydeck as pdk


@st.cache_data
def load_map_data(limit=20000):
    con = duckdb.connect("data/hdb_df_geocoded_condensed.duckdb")
    query = f"""
        SELECT latitude, longitude, resale_price, 
               nearest_mrt_name, nearest_schools_name
        FROM resale
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        LIMIT {limit}
    """
    df = con.execute(query).df()
    con.close()
    return df


def show_map():
    st.title("HDB Resale Map")
    map_df = load_map_data()

    # --- Configure pydeck map ---
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_radius=50,
        get_fill_color=[149, 6, 6, 160],  # dark red theme (#950606)
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
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={
                "html": "<b>Price:</b> {resale_price}<br/><b>MRT:</b> {nearest_mrt_name}<br/><b>School:</b> {nearest_schools_name}",
                "style": {"color": "white"},
            },
        )
    )
