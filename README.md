
- **Author**: [Alhagie Boye](https://www.linkedin.com/in/alhagie-a-boye-0568771aa/)
- **Event**: [NASA Space Apps Challenge 2025 - Banjul, The Gambia](https://www.spaceappschallenge.org/2025/local-events/banjul/)
- **Challenge**: [A World Away: Hunting for Exoplanets with AI](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/)

## Overview

This project implements an ensemble machine learning system for classifying exoplanet candidates from NASA's Kepler mission data. The system combines XGBoost, LightGBM, and CatBoost models to achieve 92.5% accuracy in distinguishing between confirmed exoplanets, planetary candidates, and false positives.

## Features

- Advanced Ensemble Model: Combines 3 cutting-edge ML algorithms
- Domain-Aware Feature Engineering: 31 features based on astronomical knowledge
- Real-Time Predictions: <100ms classification with confidence scores
- Interactive Web Interface: User-friendly Streamlit application
- Comprehensive Analytics: Feature importance and model interpretability
- Bulk Data Upload: Process multiple candidates at once
- Model Settings: Adjust ensemble weights and parameters
- Dual User Support: Optimized for both researchers and beginners

## Performance

- Accuracy: 92.5% on validation data
- Training Time: 5 minutes
- Prediction Speed: <100ms per classification
- Dataset: 9,564 exoplanet candidates with 31 features

## Quick Start

### Option 1: Use Online Version
The application is hosted and ready to use at: **https://exonet.streamlit.app/**

### Option 2: Run Locally

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run the Application
```bash
streamlit run main.py
```

The web interface will start at `http://localhost:8501`

### 3. Train the Model (Optional)
```bash
python koi_ensemble_model.py
```

## Project Structure

```
├── main.py                    # Main Streamlit application
├── koi_ensemble_model.py      # Ensemble model training
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── trained_ensemble_model.pkl # Trained ensemble model
├── sample_exoplanet_data.csv  # Sample data for testing
├── styles.css                 # Custom styling
├── exoplanet.png             # Exoplanet illustration
├── nasa.png                  # NASA logo
├── components/                # UI components
│   ├── header.py
│   ├── sidebar.py
│   ├── tabs.py
│   ├── results.py
│   └── styling.py
├── utils/                     # Utility functions
│   ├── feature_engineering.py
│   └── model_loader.py
├── data/
│   └── koi.csv               # KOI dataset (NASA Kepler data)
├── Figures/                   # Generated model outputs
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── research papers/           # Extra materials
    ├── cnn+transformer.pdf
    ├── deep-cnn.pdf
    ├── new-cnn.pdf
    ├── transformer.pdf
    └── transformer2.pdf
```

## Model Architecture

### Why Ensemble Learning?

Ensemble learning was chosen for this exoplanet detection system because:

- **Improved Accuracy**: Combining multiple models reduces individual model bias and variance, leading to more reliable predictions
- **Robustness**: Different algorithms excel at different types of patterns in the data, making the system more resilient to various exoplanet characteristics
- **Reduced Overfitting**: The ensemble approach helps prevent overfitting to specific data patterns that might not generalize to new observations
- **Confidence Scoring**: Multiple model predictions can be combined to provide more reliable confidence scores for classification decisions
- **Handling Complex Relationships**: Exoplanet detection involves complex, non-linear relationships between stellar and planetary parameters that different algorithms capture differently

### Ensemble Components
1. XGBoost: Gradient boosting with advanced regularization
2. LightGBM: Fast gradient boosting with categorical support  
3. CatBoost: Robust boosting with built-in categorical handling

### Feature Engineering
- Physical Ratios: Radius-temperature, period-depth, duration-impact
- Stellar Properties: Density, luminosity, transit probability
- Signal Quality: SNR-depth interaction, detection confidence
- Statistical Transforms: Log and square root transformations

## Web Interface Features

- Interactive Input Form: Enter exoplanet parameters with real-time validation
- Real-Time Predictions: Instant classification results with confidence scores
- Probability Visualization: Interactive charts showing prediction confidence
- Feature Importance: Understanding which parameters matter most for classification
- Bulk Data Upload: Upload CSV files for batch processing
- Model Settings: Adjust ensemble weights and parameters
- User Guidance: Separate workflows for researchers vs. beginners
- Dataset Statistics: Comprehensive overview of the KOI dataset
- Data Variable Analysis: Detailed explanation of how each parameter affects classification

## Classification Types

- CONFIRMED: Validated exoplanet (2,345 in dataset)
- CANDIDATE: Potential exoplanet requiring study (2,418 in dataset)  
- FALSE POSITIVE: Not a real exoplanet (4,801 in dataset)

## Scientific Background

This system addresses the challenge of automatically classifying exoplanet candidates from transit photometry data. Traditional manual classification by astronomers is time-consuming and subjective. Our AI system:

- Automates the classification process
- Scales to large datasets from current and future missions
- Provides consistent, objective classifications
- Enables rapid follow-up observations of promising candidates

## System Optimization

The system is designed for efficient processing:

- Parallel Training: All 3 models train simultaneously
- Batch Processing: Efficient handling of large datasets
- Memory Optimization: Smart feature selection and preprocessing
- Scalable Architecture: Easy to extend to larger datasets

## Future Enhancements

- Light Curve Integration: Add raw time-series analysis
- Real-Time Updates: Continuous learning from new discoveries
- Advanced Model Settings: More granular control over ensemble parameters
- Enhanced Bulk Upload: Support for additional file formats
- Multi-Mission Support: Extend to TESS and K2 data

## Usage Terms

- This code was developed during the NASA Space Apps Challenge hackathon
- It is provided as-is for educational and research purposes
- While Ii strive for accuracy, this was a time-constrained project and may contain errors
- Results should be validated before use in scientific publications
- Feel free to use this code with proper attribution to the NASA Space Apps team
- **Credit**: Alhagie Boye & NASA Space Apps Challenge 2025 - Banjul, The Gambia

## License

This project is open source and developed for educational and research purposes as part of the NASA Space Apps Challenge. Please provide attribution when using this code.
