import streamlit as st
import plotly.express as px

def show_flat_distribution(df):
    """ Visualizes the composition of transactions by flat type, showing market mix"""

    with st.container(border=True, height="stretch"):
        flat_counts = df["flat_type"].value_counts().reset_index()
        flat_counts.columns = ["Flat Type", "Transactions"]

        fig = px.pie(
            flat_counts,
            names="Flat Type",
            values="Transactions",
            hole=0.3,
        )

        fig.update_traces(
            textinfo="percent",
            pull=[0.04] * len(flat_counts),
            textposition="outside"
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",          
                yanchor="top", y=-0.1, 
                xanchor="center", x=0.5, 
            )
        )

        st.markdown("## 🏠 Flat Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
