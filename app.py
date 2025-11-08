import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import subprocess
import sys

# Import page modules
import mypages.Overview as overview
import mypages.Analytics as analytics
import mypages.Chatbot as chatbot

CSV_PATH = Path("data/hdb_df_geocoded_condensed.csv")
DUCKDB_PATH = Path("data/hdb_df_geocoded_condensed.duckdb")

@st.cache_resource(show_spinner="Preparing data... This will take about a minute.")
def ensure_data_ready():
    if not (CSV_PATH.exists() and DUCKDB_PATH.exists()):
        subprocess.check_call([sys.executable, "data/download_data.py"])
    return True

ensure_data_ready()

# --- Page Config ---
st.set_page_config(
    page_title="HDB Resale Market Explorer",
    page_icon="🏠",
    layout="wide",
)

# --- Init state ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

# --- Navbar ---
page_list = ["Overview", "Analytics", "Chatbot"]
default_index = page_list.index(st.session_state.current_page)

# --- Global CSS for Styling ---
st.markdown(
    """
    <style>
    /* Global font and size for everything */
    * {
        font-family: "Helvetica Neue", Arial, sans-serif !important;
        font-size: 15px !important;
        
    }

    /* Headings */
    h1 {font-size: 28px !important; font-weight: 700 !important; }
    h2 {font-size: 22px !important; font-weight: 600 !important; }
    h3 {font-size: 20px !important; font-weight: 600 !important; }

    /* Navbar */
    .nav-container {
        display: flex;
        justify-content: center;
        margin-top: 5px;
        margin-bottom: 10px;
        font-family: "Helvetica Neue", Arial, sans-serif !important;
        font-size: 18px !important;
    }
    .block-container {
        padding-top: 3rem !important;}

    /* Main content */
    .main {
        padding-top: 0.5rem !important;
        font-family: "Helvetica Neue", Arial, sans-serif !important;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Navbar ---
with st.container():
    selected = option_menu(
        menu_title=None,
        options=page_list,
        icons=["bar-chart", "graph-up", "chat-dots"],
        menu_icon="cast",
        default_index=default_index,   # sync with session_state
        orientation="horizontal",
        key="nav",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"font-size": "20px"},
            "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px"},
            "nav-link-selected": {"background-color": "#950606"},
        }
    )

# store the last selected page in a safe custom key
st.session_state.current_page = selected


# --- Page Routing ---
if st.session_state.current_page == "Overview":
    overview.show()
elif st.session_state.current_page == "Analytics":
    analytics.show()
elif st.session_state.current_page == "Chatbot":
    chatbot.show()
