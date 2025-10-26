
import streamlit as st
import plotly.express as px
import pandas as pd

def show_resale_price_distribution(df):
    """Visualize resale prices by flat type and town."""
    
    # Labels
    measure_labels = {
        "resale_price": "Resale Price",
        "price_per_sqm": "Price per sqm"
    }
    group_by_labels = {
        "flat_type": "Flat Type",
        "region": "Region",
        "town": "Town",
        "flat_model": "Flat Model",
        "district_number": "District"
    }
    agg_labels = {
        "mean": "Mean",
        "median": "Median"
    }
    
    with st.container(border=True, height="stretch"):
        st.markdown("## 💰 Mean Resale Price Analysis")
        # --- User selection ---
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            value_choice = st.selectbox(
                "**Measure:**",
                list(measure_labels.keys()),
                format_func=lambda x: measure_labels[x],
            )
            
        with col2:
            group_choice = st.selectbox(
                "**Group by:**",
                list(group_by_labels.keys()),
                format_func=lambda x: group_by_labels[x],
            )
            
        with col3:
            agg_choice = st.selectbox(
                "**Aggregate:**",
                list(agg_labels.keys()),
                index=0,
                format_func=lambda x: agg_labels[x],
            )
            
        # --- Aggregate ---
        agg_df = (
            df.groupby(group_choice)[value_choice]
            .agg(agg_choice)
            .reset_index()
            .sort_values(value_choice, ascending=False)
        )

        # --- Plot ---
        fig = px.bar(
            agg_df,
            x=group_choice,
            y=value_choice,
            color=value_choice,
            color_continuous_scale="Reds",
            title=f"HDB: {agg_choice.title()} {value_choice.replace('_', ' ').title()} by {group_choice.replace('_', ' ').title()}",
        )
        
        y_label = "Resale Price (SGD)" if value_choice == "resale_price" else "Price per sqm (SGD)"
        fig.update_layout(
            margin=dict(l=20, r=20, t=60, b=80),
            xaxis_title=group_choice.replace("_", " ").title(),
            yaxis_title=y_label,
            coloraxis_showscale=False
        )

        # Rotate x labels if many categories
        if agg_df[group_choice].nunique() > 8:
            fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)
