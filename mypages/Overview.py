import streamlit as st
import duckdb
import altair as alt

def show():
    st.title("🔍 DuckDB Data")

    con = duckdb.connect("data/hdb_df_geocoded_condensed.duckdb")

    # See what columns exist
    st.write("Table schema:")
    schema = con.execute("PRAGMA table_info(resale)").fetchdf()
    st.dataframe(schema)

    # Preview first rows
    st.write("First few rows:")
    preview = con.execute("SELECT * FROM resale LIMIT 5").fetchdf()
    st.dataframe(preview)