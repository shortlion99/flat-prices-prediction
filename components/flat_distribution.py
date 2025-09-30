import streamlit as st
import plotly.express as px

def show_flat_distribution(df):
    with st.container(border=True):
        flat_counts = df["flat_type"].value_counts().reset_index()
        flat_counts.columns = ["Flat Type", "Transactions"]

        fig = px.bar(flat_counts, x="Flat Type", y="Transactions")
        st.markdown("## 🏠 Flat Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
