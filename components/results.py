#!/usr/bin/env python3
"""
Results component for NASA Exoplanet Detection System
Handles prediction results display and bulk results
"""

import streamlit as st
import plotly.graph_objects as go
from .styling import get_prediction_colors

def display_prediction_results(prediction, probabilities, model):
    """Display individual prediction results"""
    if prediction is None:
        return
    
    # Get class labels from the model's label encoder
    classes = model.label_encoder.classes_
    
    # Main prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Convert prediction index to label
        prediction_label = classes[prediction]
        st.metric("Predicted Class", prediction_label)
    
    with col2:
        confidence = max(probabilities) * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    with col3:
        # Color code based on prediction
        if prediction_label == "CONFIRMED":
            st.markdown("""
            <div style="background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); 
                       color: #065F46; padding: 1rem; text-align: center; 
                       font-weight: bold; border: 2px solid #10B981;">
                <span style="color: #10B981; font-weight: bold;">CONFIRMED EXOPLANET</span>
            </div>
            """, unsafe_allow_html=True)
        elif prediction_label == "CANDIDATE":
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); 
                       color: #92400E; padding: 1rem;text-align: center; 
                       font-weight: bold; border: 2px solid #F59E0B;">
                <span style="color: #F59E0B; font-weight: bold;">PLANETARY CANDIDATE</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); 
                       color: #991B1B; padding: 1rem; text-align: center; 
                       font-weight: bold; border: 2px solid #EF4444;">
                <span style="color: #EF4444; font-weight: bold;">FALSE POSITIVE</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Probability distribution
    st.subheader("Prediction Probabilities")
    
    # Create probability bar chart
    fig = go.Figure(data=[
        go.Bar(x=classes, y=probabilities, 
               marker_color=['orange', 'green', 'red'],
               text=[f"{p*100:.1f}%" for p in probabilities],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Classification",
        yaxis_title="Probability",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the results
    display_prediction_explanation(prediction_label)

def display_prediction_explanation(prediction_label):
    """Display explanation of prediction results"""
    st.subheader("Explanation of Results")
    
    if prediction_label == "CONFIRMED":
        st.markdown("""
        <div style="background: #D1FAE5; padding: 1rem; margin: 1rem 0;">
            <h4 style="color: #065F46; margin: 0 0 0.5rem 0;">CONFIRMED EXOPLANET</h4>
            <p style="margin: 0; color: #065F46;">
                This object has been <strong>validated through multiple observations</strong> and is considered a confirmed exoplanet. 
                The signal shows characteristics consistent with a real planetary transit, with proper U-shaped light curves 
                and stable orbital parameters.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif prediction_label == "CANDIDATE":
        st.markdown("""
        <div style="background: #FEF3C7; padding: 1rem; margin: 1rem 0;">
            <h4 style="color: #92400E; margin: 0 0 0.5rem 0;">PLANETARY CANDIDATE</h4>
            <p style="margin: 0; color: #92400E;">
                This object is a <strong>potential exoplanet that requires further study</strong>. The signal shows promising 
                characteristics but needs additional observations or analysis to confirm its planetary nature. 
                This is a high-priority target for follow-up observations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # FALSE POSITIVE
        st.markdown("""
        <div style="background: #FEE2E2; padding: 1rem; margin: 1rem 0;">
            <h4 style="color: #991B1B; margin: 0 0 0.5rem 0;">FALSE POSITIVE</h4>
            <p style="margin: 0; color: #991B1B;">
                This object is <strong>not a real exoplanet</strong>. The signal is likely caused by stellar activity, 
                binary star systems, instrumental noise, or other astrophysical phenomena that mimic planetary transits. 
                This helps filter out false signals from the dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # General explanation of all categories
    with st.expander("All Classification Categories Explained"):
        st.markdown("""
            - **CONFIRMED**: Validated through multiple observations with consistent planetary characteristics
            - **CANDIDATE**: Promising signal requiring further study and follow-up observations  
            - **FALSE POSITIVE**: Not a real exoplanet (stellar activity, binary stars, instrumental noise, etc.)
        """)
    
    # Add bottom spacing
    st.markdown("""
    <div style="height: 2rem;"></div>
    """, unsafe_allow_html=True)

def display_bulk_results():
    """Display bulk prediction results in main area"""
    if hasattr(st.session_state, 'bulk_results') and st.session_state.bulk_results is not None:
        st.markdown("---")
        st.subheader("Bulk Prediction Results")
        st.markdown("""
    <div style="color: red; margin-bottom: 1.5rem;">This feature is still under development and will be available soon. This is a sample for the bulk results.</div>
    """, unsafe_allow_html=True)
        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Objects", len(st.session_state.bulk_results))
        with col2:
            confirmed_count = sum(1 for p in st.session_state.bulk_predictions if p is not None and p == 'CONFIRMED')
            st.metric("Confirmed", confirmed_count)
        with col3:
            candidate_count = sum(1 for p in st.session_state.bulk_predictions if p is not None and p == 'CANDIDATE')
            st.metric("Candidates", candidate_count)
        with col4:
            fp_count = sum(1 for p in st.session_state.bulk_predictions if p is not None and p == 'FALSE POSITIVE')
            st.metric("False Positives", fp_count)
        
        # Display results table
        st.dataframe(st.session_state.bulk_results, use_container_width=True)
        
        # Download results
        csv = st.session_state.bulk_results.to_csv(index=False)
        st.download_button(
            label="Download Bulk Results as CSV",
            data=csv,
            file_name="bulk_predictions_results.csv",
            mime="text/csv"
        )
