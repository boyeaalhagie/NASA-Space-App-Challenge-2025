#!/usr/bin/env python3
"""
Model loader utilities for NASA Exoplanet Detection System
Handles model loading and prediction making
"""

import streamlit as st
import joblib
import os
from .feature_engineering import prepare_prediction_data

class ModelLoader:
    """Handles model loading and prediction operations"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ensemble model"""
        try:
            # Try to load the model
            if os.path.exists('trained_ensemble_model.pkl'):
                self.model = joblib.load('trained_ensemble_model.pkl')
                # Model loaded silently
            else:
                st.warning("Model not found. Please train the model first.")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def make_prediction(self, input_data):
        """Make prediction using the ensemble model"""
        if self.model is None:
            return None, None
        
        try:
            # Prepare data with proper feature engineering and ordering
            X = prepare_prediction_data(input_data)
            
            # Make prediction using the ensemble model
            prediction = self.model.ensemble_model.predict(X)[0]
            probabilities = self.model.ensemble_model.predict_proba(X)[0]
            
            return prediction, probabilities
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None
    
    def is_model_loaded(self):
        """Check if model is successfully loaded"""
        return self.model is not None
