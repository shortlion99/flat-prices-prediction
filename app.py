import streamlit as st
from streamlit_option_menu import option_menu

# Import page modules
import mypages.Overview as overview
import mypages.Analytics as analytics
import mypages.Chatbot as chatbot
import mypages.HomeMap as homemap

# --- Page Config ---
st.set_page_config(page_title="HDB Resale Market Explorer", layout="wide")

# --- Init state ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# --- Navbar ---
page_list = ["Home", "Overview", "Analytics", "Chatbot"]
default_index = page_list.index(st.session_state.current_page)

# --- Global CSS for Styling ---
st.markdown(
    """
    <style>
    /* Global font and size for everything */
    * {
        font-family: "Helvetica Neue", Arial, sans-serif !important;
        font-size: 15px !important;
        color: #111111 !important;
        
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
        icons=["house", "bar-chart", "graph-up", "chat-dots"],
        menu_icon="cast",
        default_index=default_index,   # sync with session_state
        orientation="horizontal",
        key="nav",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"font-size": "20px"},
            "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px"},
            "nav-link-selected": {"background-color": "#950606", "color": "white"},
        }
    )

# store the last selected page in a safe custom key
st.session_state.current_page = selected


# --- Page Routing ---
if st.session_state.current_page == "Home":
    st.markdown("# HDB Resale Market Explorer")
    homemap.show_map()

    st.markdown("## Purpose")
    st.markdown(
        "This dashboard provides comprehensive predictive analytics and insights for "
        "Singapore's HDB resale market, designed to help home buyers, investors, "
        "policymakers, and researchers make informed decisions. "
        "Using transaction data from 2017 onwards, we combine machine learning models with interactive visualizations "
        "to deliver accurate property price predictions and market trend analysis."
    )

    st.markdown("## Features")
    st.markdown(
    """
**Overview** – Market trends, district price maps, and historical analysis across HDB resale properties.  
**Analytics** – Price prediction tools based on location, amenities, and property attributes. Includes time-series forecasting and built-in models.  
**Chatbot** – Conversational AI assistant providing instant property market insights and personalised guidance.  
    """
)


elif st.session_state.current_page == "Overview":
    overview.show()
elif st.session_state.current_page == "Analytics":
    analytics.show()
elif st.session_state.current_page == "Chatbot":
    chatbot.show()
