import streamlit as st
import plotly.express as px

def show_flat_distribution(df):
    flat_counts = df["flat_type"].value_counts().reset_index()
    flat_counts.columns = ["Flat Type", "Transactions"]

    fig = px.bar(flat_counts, x="Flat Type", y="Transactions",
                 title="Flat Type Distribution (2017–2025)")
    st.plotly_chart(fig, use_container_width=True)
