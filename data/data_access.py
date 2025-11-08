# data_access.py
import os
import duckdb
import streamlit as st

DB_PATH = "data/hdb_df_geocoded_condensed.duckdb"

@st.cache_resource
def get_duckdb_conn(path: str = DB_PATH):
    return duckdb.connect(path, read_only=True)

