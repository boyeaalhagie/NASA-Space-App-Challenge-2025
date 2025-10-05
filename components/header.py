#!/usr/bin/env python3
"""
Header component for NASA Exoplanet Detection System
Handles NASA logo and main header display
"""

import streamlit as st
import base64

def get_nasa_logo_base64():
    """Get NASA logo as base64 encoded string"""
    try:
        with open("nasa.png", "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

def create_nasa_header():
    """Create and display the NASA-themed header with logo"""
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{}" style="width: 70px; height: 70px; margin-right: 20px;">
            <h1>NASA Exoplanet Detection System</h1>
        </div>
        <p>Machine Learning Classification of Kepler Objects of Interest</p>
    </div>
    """.format(get_nasa_logo_base64()), unsafe_allow_html=True)

def set_page_config():
    """Set Streamlit page configuration with NASA theme"""
    st.set_page_config(
        page_title="NASA Exoplanet Detection System",
        page_icon="nasa.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
