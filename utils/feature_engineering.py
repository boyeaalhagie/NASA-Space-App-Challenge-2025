#!/usr/bin/env python3
"""
Feature engineering utilities for NASA Exoplanet Detection System
Handles feature creation and preprocessing for predictions
"""

import pandas as pd
import numpy as np

def create_engineered_features(input_data):
    """Create engineered features from input data"""
    df = pd.DataFrame([input_data])
    
    # Add engineered features (exactly as in training)
    df['radius_temp_ratio'] = df['koi_prad'] / df['koi_teq']
    df['period_depth_ratio'] = df['koi_period'] / (df['koi_depth'] + 1e-6)
    df['duration_impact_ratio'] = df['koi_duration'] / (df['koi_impact'] + 1e-6)
    df['snr_depth_interaction'] = df['koi_model_snr'] * np.log(df['koi_depth'] + 1)
    df['stellar_density'] = df['koi_smass'] / (df['koi_srad'] ** 3 + 1e-6)
    df['transit_probability'] = df['koi_srad'] / (df['koi_sma'] + 1e-6) if 'koi_sma' in df.columns else df['koi_srad'] / (df['koi_period'] + 1e-6)
    df['transit_duration_norm'] = df['koi_duration'] / df['koi_period']
    df['signal_strength'] = df['koi_model_snr'] * df['koi_depth']
    df['detection_confidence'] = df['koi_model_snr'] / (df['koi_model_snr'] + 10)
    
    # Log transformations
    df['koi_period_log'] = np.log(df['koi_period'] + 1)
    df['koi_depth_log'] = np.log(df['koi_depth'] + 1)
    df['koi_duration_log'] = np.log(df['koi_duration'] + 1)
    df['koi_model_snr_log'] = np.log(df['koi_model_snr'] + 1)
    
    return df

def get_feature_columns():
    """Get the exact feature columns used in training (in the same order)"""
    return [
        # Core transit parameters
        'koi_period', 'koi_duration', 'koi_depth', 'koi_impact', 'koi_ror',
        'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_insol',
        
        # Stellar properties
        'koi_steff', 'koi_srad', 'koi_smass', 'koi_slogg', 'koi_smet',
        
        # Quality flags
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        
        # Advanced engineered features
        'radius_temp_ratio', 'period_depth_ratio', 'duration_impact_ratio',
        'snr_depth_interaction', 'stellar_density', 'transit_probability',
        'transit_duration_norm', 'signal_strength', 'detection_confidence',
        
        # Log transformations
        'koi_period_log', 'koi_depth_log', 'koi_duration_log', 'koi_model_snr_log'
    ]

def prepare_prediction_data(input_data):
    """Prepare input data for prediction with proper feature engineering and ordering"""
    # Create engineered features
    df_engineered = create_engineered_features(input_data)
    
    # Get feature columns in exact training order
    feature_columns = get_feature_columns()
    
    # Filter to available features and ensure exact order
    available_features = [col for col in feature_columns if col in df_engineered.columns]
    X = df_engineered[available_features]
    
    # Ensure that we have all required features, fill missing ones with 0
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
            print(f"Warning: Missing feature {col}, filled with 0")
    
    # Reorder to match training exactly
    X = X[feature_columns]
    
    return X
