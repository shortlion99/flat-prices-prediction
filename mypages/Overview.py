import streamlit as st
import pandas as pd
import altair as alt

def show():
    st.title("📈 Overview - SG Property Price Forecast")

    # Dummy data
    historical_data = pd.DataFrame([
        {"year": 2018, "price": 450000},
        {"year": 2019, "price": 470000},
        {"year": 2020, "price": 480000},
        {"year": 2021, "price": 500000},
        {"year": 2022, "price": 510000},
    ])

    forecast_data = pd.DataFrame([
        {"year": 2023, "price": 520000, "low": 505000, "high": 535000},
        {"year": 2024, "price": 530000, "low": 510000, "high": 550000},
        {"year": 2025, "price": 540000, "low": 515000, "high": 565000},
    ])

    chart_data = pd.concat([historical_data, forecast_data], ignore_index=True)
    user_inputs, output_panel = st.columns([0.8, 2.5], gap="medium")

    with user_inputs:
        st.subheader("User Inputs")
        district = st.selectbox("Select District", ["District 1", "District 2"])
        flat_type = st.selectbox("Flat Type", ["3-Room", "4-Room"])
        size = st.number_input("Size (sqm)", min_value=10, max_value=200, value=100)
        lease_years = st.number_input("Lease Years Remaining", min_value=0, max_value=99, value=60)
        forecast_horizon = st.slider("Forecast Horizon (Years)", 1, 10, 5)
        if st.button("Generate Insights"):
            st.success("Insights generated below.")

    with output_panel:
        predicted_price, feature_importance = st.columns([1, 1])
        with predicted_price:
            st.subheader("Current Predicted Price")
            st.markdown("### $520,000")
            st.caption("95% CI: $505k – $535k")
        with feature_importance:
            st.subheader("Feature Importance")
            st.markdown(
                """
                - **Lease Years Remaining** – 40%  
                - **District** – 25%  
                - **Size** – 20%  
                - **Proximity to MRT** – 15%
                """
            )

        st.subheader("Forecasted Price Trend")
        # Area chart with confidence interval
        base = alt.Chart(chart_data).encode(
            x=alt.X("year:O", title="Year"),
        )
        band = base.mark_area(opacity=0.3, color="#93c5fd").encode(
            y="low:Q",
            y2="high:Q",
            tooltip=["year", "low", "high"],
        )
        line = base.mark_line(color="#2563eb").encode(
            y=alt.Y("price:Q", title="Price"),
            tooltip=["year", "price"],
        )
        chart = (band + line).properties(width=700, height=400)
        st.altair_chart(chart, use_container_width=True)
