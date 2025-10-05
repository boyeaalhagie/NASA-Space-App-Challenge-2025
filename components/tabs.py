#!/usr/bin/env python3
"""
Tabs component for NASA Exoplanet Detection System
Handles all tab content including About, System Overview, Model Info, and User Guidance
"""

import streamlit as st
from .styling import get_tab_content_container_html, get_footer_html

def create_about_tab():
    """Create the About tab content"""
    st.markdown(get_tab_content_container_html(), unsafe_allow_html=True)
    
    st.header("What is an Exoplanet?")
    
    # Create two columns for laptop layout
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        An **exoplanet** (or extrasolar planet) is a planet that orbits a star outside our Solar System. 
        The first confirmed detection of an exoplanet was in 1992, and since then, astronomers have 
        discovered thousands of these distant worlds, revealing an incredible diversity of planetary systems.
        
        Studying exoplanets helps us understand:
        - How planets form and evolve
        - The variety of planetary systems in our galaxy
        - The potential for life beyond Earth
        - The distribution of planet types and sizes
        """)
    
    with col2:
        try:
            st.image("exoplanet.png", caption="Artist's concept of an exoplanet orbiting its host star")
        except FileNotFoundError:
            st.warning("Exoplanet image not found. Please ensure exoplanet.png is in the project directory.")
    
    st.header("How is an Exoplanet Detected?")
    st.markdown("""
    Detecting exoplanets is extremely challenging because they are:
    - **Incredibly far away** (light-years from Earth)
    - **Much smaller than stars** (planets are typically 1000x smaller)
    - **Very dim** compared to their host stars (planets don't produce their own light)
    
    ##### The Transit Method (Like an Eclipse in Space) - Used by [Kepler](https://science.nasa.gov/mission/kepler/)
    
    Imagine you're watching a bright light bulb (a star) from far away. If a small fly (a planet) flies directly in front of the light bulb, the light would get slightly dimmer for a moment. That's exactly how we detect exoplanets!
    
    **Simple Explanation:**
    - **What happens**: A planet blocks a tiny bit of the star's light as it passes in front
    - **What we measure**: The star gets very slightly dimmer (like turning down a light by 0.01%)
    - **Why it works**: We can detect this tiny change using very sensitive telescopes
    - **What we learn**: How big the planet is, how long it takes to orbit, and sometimes what's in its atmosphere
    
    **Why This Method is Perfect for Finding Earth-like Planets:**
    - **Sensitive enough**: Can detect planets as small as Earth
    - **Works on many stars**: Can watch 150,000+ stars at the same time
    - **Gives us data**: Creates the massive datasets needed for machine learning
    - **Proven success**: Kepler found thousands of planet candidates using this method
    """)
    
    # Embed the YouTube video with reduced size
    st.markdown("### Exoplanet Transit Animation - Planet Passing in Front of Star")
    st.markdown("""
    **This video shows:**
    - An exoplanet orbiting around its star
    - The planet passing directly in front of the star (transit)
    - The star's light dimming as the planet blocks it
    
    *See the transit method in action - the same technique that generated our training data!*
    """)
    st.video("https://youtu.be/TVJmC19juU0")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions for running predictions
    st.markdown("""
    <div style="background: #0B3D91; padding: 1rem; margin: 1rem 0;">
        <h4 style="color: white; margin: 0 0 0.5rem 0;">How to Run Predictions</h4>
        <p style="margin: 0; color: white;">
            <strong>Desktop:</strong> Use the input parameters in the sidebar to enter your exoplanet data.<br>
            <strong>Mobile:</strong> Click the <strong>>> menu icon</strong> in the top-left corner to access input parameters.<br>
            Then click <strong>"Run Prediction"</strong> to get instant classification results.
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_system_overview_tab():
    """Create the System Overview tab content"""
    st.markdown(get_tab_content_container_html(), unsafe_allow_html=True)
    st.header("System Overview")
    st.markdown("""
    This NASA Exoplanet Detection System uses advanced machine learning to automatically classify 
    planetary candidates from the Kepler mission data, helping astronomers prioritize which 
    candidates deserve further study.
    
    **Technical Features:**
    - **Ensemble Learning**: Combines XGBoost, LightGBM, and CatBoost models
    - **Feature Engineering**: 31 astronomical parameters optimized for exoplanet detection
    - **Real-time Analysis**: Instant predictions with confidence scores
    - **Interpretable Results**: Feature importance and probability distributions
    - **Bulk Processing**: Handle large datasets for research teams
    - **Hyperparameter Tuning**: Customize model weights for specific research goals
    
    **Classification Categories:**
    - <span style="color: #10B981; font-weight: bold;">**CONFIRMED**</span>: Validated exoplanet with high confidence
    - <span style="color: #F59E0B; font-weight: bold;">**CANDIDATE**</span>: Potential exoplanet requiring further study
    - <span style="color: #EF4444; font-weight: bold;">**FALSE POSITIVE**</span>: Not a real exoplanet (stellar activity, binary stars, etc.)
    """, unsafe_allow_html=True)
    
    st.header("Parameter Impact Analysis")
    st.markdown("""
    **How each parameter affects exoplanet classification:**
    
    **Orbital Parameters:**
    - **Period**: Longer periods (>100 days) often indicate false positives
    - **Duration**: Very short (<1 hour) or very long (>12 hours) durations suggest false signals
    - **Depth**: Unusually deep transits (>10,000 ppm) often indicate stellar eclipses
    
    **Stellar Properties:**
    - **Temperature**: Hotter stars (>7000K) have more stellar activity (false positives)
    - **Radius**: Large stellar radii can mask small planet signals
    - **Surface Gravity**: Low gravity stars show more variability
    
    **Signal Quality:**
    - **Signal-to-Noise Ratio**: Higher SNR (>20) indicates more reliable detections
    - **Impact Parameter**: Values near 1.0 suggest grazing transits (often false)
    
    **False Positive Flags:**
    - **Not Transit-like**: V-shaped transits vs. U-shaped planetary transits
    - **Stellar Eclipse**: Binary star systems mimicking planets
    - **Centroid Offset**: Background eclipsing binaries
    - **Ephemeris Match**: Known variable stars
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def create_model_info_tab(display_feature_importance_func):
    """Create the Model Information tab content"""
    st.markdown(get_tab_content_container_html(), unsafe_allow_html=True)
    st.header("Model Information")
    st.markdown("""
    **Model Architecture:**
    - **XGBoost**: Gradient boosting with advanced regularization
    - **LightGBM**: Fast gradient boosting with categorical support
    - **CatBoost**: Robust boosting with built-in categorical handling
    
    **Performance:**
    - **Ensemble Accuracy**: 92.5% on validation data
    - **XGBoost Accuracy**: 92.9%
    - **LightGBM Accuracy**: 92.8%
    - **CatBoost Accuracy**: 92.2%
    - **Training Time**: ~15 minutes on high-performance computing
    - **Prediction Speed**: <100ms per classification
    """)
    
    display_feature_importance_func()
    
    st.header("Dataset Statistics")
    st.markdown("""
    The model is trained on the Kepler Objects of Interest (KOI) dataset, which contains 
    observations from NASA's Kepler space telescope. This dataset represents one of the 
    most comprehensive exoplanet surveys ever conducted.
    """)
    
    # Enhanced statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Candidates", "9,564")
        st.metric("Confirmed Planets", "2,345")
        st.metric("Success Rate", "24.5%")
    
    with col2:
        st.metric("Planetary Candidates", "2,418")
        st.metric("False Positives", "4,801")
        st.metric("Pending Review", "56.6%")
    
    with col3:
        st.metric("Features Used", "31")
        st.metric("Model Accuracy", "92.5%")
        st.metric("Training Time", "5 min")
    
    st.markdown("""
    **Key Statistics from the Training Data:**
    - **Average Orbital Period**: ~30 days
    - **Average Transit Depth**: ~1000 ppm
    - **Average Stellar Temperature**: ~5500 K
    - **Mission Duration**: 4 years of continuous observations
    - **Stars Monitored**: ~150,000 stars
    
    These statistics provide context for the model's training and the distribution 
    of exoplanet types in the Kepler mission's observations.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def create_user_guidance_tab():
    """Create the User Guidance tab content"""
    st.markdown(get_tab_content_container_html(), unsafe_allow_html=True)
    st.header("User Guidance")
    st.markdown("""
    This tool is designed for both seasoned researchers and curious beginners interested in exoplanet discovery. 
    Choose your experience level below to get tailored guidance for using the system effectively.
    """)
    
    # Researcher vs Novice guidance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("For Researchers")
        st.markdown("""
        **Advanced Features:**
        - **Bulk Upload**: Process large datasets using the CSV upload feature
        - **Hyperparameter Tuning**: Adjust ensemble weights in Model Settings
        - **Feature Analysis**: View detailed feature importance in Model Information
        - **Export Results**: Download prediction results for further analysis
        
        **Recommended Workflow:**
        1. **Data Preparation**: Ensure your data matches the required input format
        2. **Bulk Processing**: Use the upload feature for large candidate lists
        3. **Parameter Adjustment**: Fine-tune model weights based on your research focus
        4. **Validation**: Cross-reference predictions with known exoplanet catalogs
        5. **Follow-up**: Prioritize high-confidence candidates for observational studies
        
        **Pro Tips:**
        - Focus on candidates with Signal-to-Noise Ratio > 20
        - Pay attention to false positive flags in your data
        - Use the feature importance analysis to guide parameter selection
        - Consider the ensemble confidence scores for candidate prioritization
        """)
    
    with col2:
        st.subheader("For Beginners")
        st.markdown("""
        **Getting Started:**
        - **Explore Parameters**: Use the sidebar sliders to understand each input parameter
        - **Start Simple**: Begin with typical values shown in the tooltips
        - **Watch the Video**: Check out the exoplanet explanation video in the About tab
        - **Learn the Science**: Read about parameter impacts in the About section
        
        **Understanding Results:**
        - <span style="color: #10B981; font-weight: bold;">**Green (CONFIRMED)**</span>: High-confidence exoplanet detection
        - <span style="color: #F59E0B; font-weight: bold;">**Yellow (CANDIDATE)**</span>: Potential planet requiring further study
        - <span style="color: #EF4444; font-weight: bold;">**Red (FALSE POSITIVE)**</span>: Not a real exoplanet
        - **Confidence Score**: Higher percentage = more reliable prediction
        
        **Learning Path:**
        1. **Watch the Video**: Start with the exoplanet explanation
        2. **Read About**: Understand the science behind the parameters
        3. **Experiment**: Try different parameter combinations
        4. **Analyze Results**: Learn to interpret the predictions
        5. **Explore Further**: Dive into the model information when ready    
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_footer():
    """Create the footer"""
    st.markdown(get_footer_html(), unsafe_allow_html=True)
