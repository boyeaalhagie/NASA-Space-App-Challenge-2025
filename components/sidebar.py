#!/usr/bin/env python3
"""
Sidebar component for NASA Exoplanet Detection System
Handles input forms, advanced features, and bulk processing
"""

import streamlit as st
import pandas as pd
import time
from .styling import get_section_header_html

def create_input_form():
    """Create input form for new predictions with collapsible sections"""
    st.sidebar.markdown(get_section_header_html("Input Parameters"), unsafe_allow_html=True)
    
    # Use st.expander for fast, native collapsible sections
    with st.sidebar.expander("Transit Parameters", expanded=False):
        period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)
        duration = st.number_input("Transit Duration (hours)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
        depth = st.number_input("Transit Depth (ppm)", min_value=1.0, max_value=100000.0, value=1000.0, step=10.0)
        impact = st.number_input("Impact Parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        ror = st.number_input("Planet-Star Radius Ratio", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
        prad = st.number_input("Planetary Radius (Earth radii)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
        teq = st.number_input("Equilibrium Temperature (K)", min_value=100.0, max_value=3000.0, value=300.0, step=10.0)
        insol = st.number_input("Insolation Flux (Earth flux)", min_value=0.1, max_value=10000.0, value=100.0, step=1.0)
    
    with st.sidebar.expander("Stellar Properties", expanded=False):
        steff = st.number_input("Stellar Temperature (K)", min_value=2000.0, max_value=10000.0, value=6000.0, step=100.0)
        srad = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        smass = st.number_input("Stellar Mass (Solar mass)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        slog = st.number_input("Stellar Surface Gravity (log10)", min_value=3.0, max_value=5.0, value=4.4, step=0.1)
        smet = st.number_input("Stellar Metallicity (dex)", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    
    with st.sidebar.expander("Signal Quality", expanded=False):
        model_snr = st.number_input("Signal-to-Noise Ratio", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)
    
    with st.sidebar.expander("False Positive Flags", expanded=False):
        fp_nt = st.selectbox("Not Transit-like Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        fp_ss = st.selectbox("Stellar Eclipse Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        fp_co = st.selectbox("Centroid Offset Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        fp_ec = st.selectbox("Ephemeris Match Flag", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    return {
        'koi_period': period,
        'koi_duration': duration,
        'koi_depth': depth,
        'koi_impact': impact,
        'koi_ror': ror,
        'koi_prad': prad,
        'koi_teq': teq,
        'koi_insol': insol,
        'koi_steff': steff,
        'koi_srad': srad,
        'koi_smass': smass,
        'koi_slogg': slog,
        'koi_smet': smet,
        'koi_model_snr': model_snr,
        'koi_fpflag_nt': fp_nt,
        'koi_fpflag_ss': fp_ss,
        'koi_fpflag_co': fp_co,
        'koi_fpflag_ec': fp_ec
    }

def add_advanced_features(model, make_prediction_func):
    """Add advanced features to the sidebar"""
    st.sidebar.markdown(get_section_header_html("Advanced Options"), unsafe_allow_html=True)
    st.sidebar.markdown('<p style="color: #991B1B;">Coming Soon...</p>', unsafe_allow_html=True)
    
    # Bulk data upload
    with st.sidebar.expander("Bulk Data Upload"):
        # Sample CSV download
        st.markdown("**Sample CSV Format:**")
        try:
            with open("sample_exoplanet_data.csv", "r") as f:
                sample_csv = f.read()
            st.download_button(
                label="Download Sample CSV",
                data=sample_csv,
                file_name="sample_exoplanet_data.csv",
                mime="text/csv"
            )
        except FileNotFoundError:
            st.warning("Sample CSV file not found")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                bulk_data = pd.read_csv(uploaded_file)
                # Store upload time for auto-removal
                st.session_state.upload_time = time.time()
                
                if st.button("Predict All"):
                    try:
                        # Process bulk predictions
                        predictions = []
                        probabilities = []
                        prediction_labels = []
                        
                        # Get class labels from the model's label encoder
                        classes = model.label_encoder.classes_
                        
                        for index, row in bulk_data.iterrows():
                            # Create input data for this row
                            input_data = {
                                'koi_period': row.get('koi_period', 10.0),
                                'koi_duration': row.get('koi_duration', 5.0),
                                'koi_depth': row.get('koi_depth', 1000.0),
                                'koi_impact': row.get('koi_impact', 0.5),
                                'koi_ror': row.get('koi_ror', 0.1),
                                'koi_model_snr': row.get('koi_model_snr', 20.0),
                                'koi_prad': row.get('koi_prad', 2.0),
                                'koi_teq': row.get('koi_teq', 300.0),
                                'koi_insol': row.get('koi_insol', 100.0),
                                'koi_steff': row.get('koi_steff', 6000.0),
                                'koi_srad': row.get('koi_srad', 1.0),
                                'koi_smass': row.get('koi_smass', 1.0),
                                'koi_slogg': row.get('koi_slogg', 4.4),
                                'koi_smet': row.get('koi_smet', 0.0),
                                'koi_fpflag_nt': row.get('koi_fpflag_nt', 0),
                                'koi_fpflag_ss': row.get('koi_fpflag_ss', 0),
                                'koi_fpflag_co': row.get('koi_fpflag_co', 0),
                                'koi_fpflag_ec': row.get('koi_fpflag_ec', 0)
                            }
                            
                            # Make prediction
                            prediction, prob = make_prediction_func(input_data)
                            predictions.append(prediction)
                            probabilities.append(prob)
                            
                            # Convert prediction index to label
                            if prediction is not None:
                                prediction_label = classes[prediction]
                                prediction_labels.append(prediction_label)
                            else:
                                prediction_labels.append("ERROR")
                        
                        # Create results DataFrame
                        results_df = bulk_data.copy()
                        results_df['Prediction'] = prediction_labels
                        results_df['Confidence'] = [max(prob) * 100 if prob is not None else 0 for prob in probabilities]
                        
                        # Store results in session state for display in main area
                        st.session_state.bulk_results = results_df
                        st.session_state.bulk_predictions = prediction_labels
                        st.session_state.bulk_probabilities = probabilities
                        
                        # Store completion time for auto-removal
                        st.session_state.completion_time = time.time()
                        
                    except Exception as e:
                        st.error(f"Error processing bulk data: {e}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Hyperparameter tuning
    with st.sidebar.expander("Model Settings"):
        st.write("**Ensemble Weights:**")
        xgb_weight = st.slider("XGBoost Weight", 0.0, 1.0, 0.4, 0.1)
        lgb_weight = st.slider("LightGBM Weight", 0.0, 1.0, 0.3, 0.1)
        cat_weight = st.slider("CatBoost Weight", 0.0, 1.0, 0.3, 0.1)
        
        if st.button("Retrain with New Weights"):
            st.info("Retraining with custom weights...")
            # TODO: Implement retraining functionality

def create_prediction_button(make_prediction_func, input_data):
    """Create the main prediction button"""
    if st.sidebar.button("Run Prediction", type="primary"):
        with st.spinner("Analyzing data..."):
            prediction, probabilities = make_prediction_func(input_data)
            
            if prediction is not None:
                st.success("Prediction complete!")
                return prediction, probabilities
    return None, None
