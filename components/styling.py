#!/usr/bin/env python3
"""
Styling component for NASA Exoplanet Detection System
Handles all CSS styling and theme management
"""

import streamlit as st
import os

def load_css():
    """Load external CSS file for styling"""
    try:
        with open('styles.css', 'r') as f:
            css = f.read()
        return css
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styling.")
        return ""

def apply_nasa_styling():
    """Apply NASA-themed styling to the Streamlit app"""
    css = load_css()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def get_section_header_html(text):
    """Get HTML for section headers with NASA styling"""
    return f'<div class="section-header">{text}</div>'

def get_metric_container_html():
    """Get HTML for metric containers with NASA styling"""
    return '<div class="metric-container">'

def get_plot_container_html():
    """Get HTML for plot containers with NASA styling"""
    return '<div class="plot-container">'

def get_tab_content_container_html():
    """Get HTML for tab content containers with NASA styling"""
    return '<div class="tab-content-container">'

def get_footer_html():
    """Get HTML for footer with NASA styling"""
    return '''
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #6B7290; font-size: 0.9rem;">
        <p style="margin-bottom: 0.2rem;">NASA Space App Challenge 2025</p>
        <p style="margin-top: 0;">Developed by <a href="https://www.linkedin.com/in/alhagie-a-boye-0568771aa/" target="_blank" style="color: #0B3D91; text-decoration: underline; font-weight: semibold;">Alhagie Boye</a></p>
    </div>
    '''

def get_nasa_colors():
    """Get NASA color palette"""
    return {
        'primary_blue': '#0B3D91',
        'secondary_blue': '#1E40AF', 
        'nasa_red': '#DC2626',
        'dark_gray': '#1F2937',
        'medium_gray': '#374151',
        'light_gray': '#6B7280',
        'white': '#FFFFFF',
        'black': '#000000'
    }

def get_prediction_colors():
    """Get colors for prediction categories"""
    return {
        'CONFIRMED': '#10B981',  
        'CANDIDATE': '#F59E0B',  
        'FALSE POSITIVE': '#EF4444'  
    }
