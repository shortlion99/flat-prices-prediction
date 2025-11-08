import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import statsmodels.api as sm

def show_relationship(df):
    """Interactive scatterplot with optional regression line for price relationships."""
    
    with st.container(border=True):
        st.markdown("## 📌 Variable Impact on Resale Price")

        # --- Selection Row ---
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # x-axis
        with col1:
            x_axis_labels = {
                "floor_area_sqm": "Floor Area (sqm)",
                "remaining_lease": "Remaining Lease (years)",
                "storey_mid": "Approximate Storey Level",
                "nearest_mrt_distance_km": "Distance to Nearest MRT (km)",
                "nearest_hawker_distance_km": "Distance to Nearest Hawker Centre (km)",
                "nearest_supermarkets_distance_km": "Distance to Nearest Supermarket (km)",
            }
            x_var = st.selectbox(
                "**Independent Variable (X-axis):**",
                list(x_axis_labels.keys()),
                format_func=lambda x: x_axis_labels[x],
            )

        # y-axis
        with col2:
            y_axis_labels = {
                "resale_price": "Resale Price",
                "price_per_sqm": "Price per sqm",
            }
            y_var = st.selectbox(
                "**Outcome Variable (Y-axis):**",
                list(y_axis_labels.keys()),
                index=0,
                format_func=lambda x: y_axis_labels[x],
            )
        
        # Group By
        with col3:
            color_labels = {
                None: "None",
                "region": "Region",
                "flat_type": "Flat Type",
                "town": "Town",
            }
            color_var = st.selectbox(
                "**Group By:**",
                list(color_labels.keys()),
                index=1,
                format_func=lambda x: color_labels[x],
            )


        show_regression = st.checkbox("Show regression line", value=True)

        # Filter out missing values
        df_plot = df[[x_var, y_var, color_var] if color_var else [x_var, y_var]].dropna()

        # Base scatterplot
        fig = px.scatter(
            df_plot,
            x=x_var,
            y=y_var,
            color=color_var if color_var else None,
            trendline="ols" if show_regression else None,
            labels={x_var: x_var.replace("_", " ").title(), y_var: y_var.replace("_", " ").title()},
            title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}",
            opacity=0.3,
        )

        st.plotly_chart(fig, use_container_width=True)
