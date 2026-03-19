import streamlit as st

# EPL Dark Theme
EPL_THEME = {
    "config": {
        "theme": {
            "primary": "#1f77b4",  # EPL Green-ish
            "backgroundColor": "#0e1117",
            "secondaryBackgroundColor": "#262730",
            "textColor": "#FAFAFA",
            "font": "sans serif"
        }
    }
}

import streamlit as st

def init_app():
    st.set_page_config(
        page_title="EPL Match Predictor",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{{ {f.read()} }}</style>", unsafe_allow_html=True)

# Custom CSS for EPL styling
CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%);
}
.stMetric > div > div > div > div {
    color: #00D4AA;
    font-weight: bold;
}
.card {
    background-color: #262730;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #00D4AA;
}
</style>
"""

def init_app():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

