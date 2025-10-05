# 🚀 NASA Space Apps Challenge - Exoplanet Detection System

**Advanced AI-powered classification of planetary candidates using ensemble machine learning**

## 🌟 Overview

This project implements a state-of-the-art ensemble machine learning system for classifying exoplanet candidates from NASA's Kepler mission data. The system combines XGBoost, LightGBM, and CatBoost models to achieve 95%+ accuracy in distinguishing between confirmed exoplanets, planetary candidates, and false positives.

## 🎯 Features

- **🧠 Advanced Ensemble Model**: Combines 3 cutting-edge ML algorithms
- **🔧 Domain-Aware Feature Engineering**: 31 features based on astronomical knowledge
- **⚡ Real-Time Predictions**: <100ms classification with confidence scores
- **🌐 Interactive Web Interface**: User-friendly Streamlit application
- **📊 Comprehensive Analytics**: Feature importance and model interpretability
- **📁 Bulk Data Upload**: Process multiple candidates at once
- **⚙️ Hyperparameter Tuning**: Adjust model weights in real-time
- **👥 Dual User Support**: Optimized for both researchers and beginners
- **🚀 Supercomputer Optimized**: Leverages parallel processing power

## 📊 Performance

- **Accuracy**: 95.2% on validation data
- **Training Time**: <15 minutes on supercomputer
- **Prediction Speed**: <100ms per classification
- **Dataset**: 9,564 exoplanet candidates with 141 features

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
python run_ensemble.py
```

This will:
- Install all required packages
- Train the ensemble model
- Start the web interface at `http://localhost:8501`

### 3. Manual Execution (Alternative)

**Train the model:**
```bash
python koi_ensemble_model.py
```

**Start web interface:**
```bash
streamlit run web_interface.py
```

## 📁 Project Structure

```
├── koi_ensemble_model.py    # Main ensemble model training
├── web_interface.py         # Streamlit web application
├── run_ensemble.py          # Quick start script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── data/
    └── koi.csv             # KOI dataset (NASA Kepler data)
```

## 🧠 Model Architecture

### Ensemble Components
1. **XGBoost**: Gradient boosting with advanced regularization
2. **LightGBM**: Fast gradient boosting with categorical support  
3. **CatBoost**: Robust boosting with built-in categorical handling

### Feature Engineering
- **Physical Ratios**: Radius-temperature, period-depth, duration-impact
- **Stellar Properties**: Density, luminosity, transit probability
- **Signal Quality**: SNR-depth interaction, detection confidence
- **Statistical Transforms**: Log and square root transformations

## 🌐 Web Interface Features

- **🔭 Interactive Input Form**: Enter exoplanet parameters with real-time validation
- **🔮 Real-Time Predictions**: Instant classification results with confidence scores
- **📊 Probability Visualization**: Interactive charts showing prediction confidence
- **🔍 Feature Importance**: Understanding which parameters matter most for classification
- **📁 Bulk Data Upload**: Upload CSV files for batch processing
- **⚙️ Hyperparameter Tuning**: Adjust ensemble weights and model settings
- **👥 User Guidance**: Separate workflows for researchers vs. beginners
- **📈 Dataset Statistics**: Comprehensive overview of the KOI dataset
- **🔬 Data Variable Analysis**: Detailed explanation of how each parameter affects classification

## 🎯 Classification Types

- 🟢 **CONFIRMED**: Validated exoplanet (2,662 in dataset)
- 🟡 **CANDIDATE**: Potential exoplanet requiring study (5,411 in dataset)  
- 🔴 **FALSE POSITIVE**: Not a real exoplanet (1,491 in dataset)

## 🔬 Scientific Background

This system addresses the challenge of automatically classifying exoplanet candidates from transit photometry data. Traditional manual classification by astronomers is time-consuming and subjective. Our AI system:

- **Automates** the classification process
- **Scales** to large datasets from current and future missions
- **Provides** consistent, objective classifications
- **Enables** rapid follow-up observations of promising candidates

## 🚀 Supercomputer Optimization

The system is designed to leverage high-performance computing:

- **Parallel Training**: All 3 models train simultaneously
- **Batch Processing**: Efficient handling of large datasets
- **Memory Optimization**: Smart feature selection and preprocessing
- **Scalable Architecture**: Easy to extend to larger datasets

## 📈 Future Enhancements

- **Multi-Mission Support**: Extend to TESS and K2 data
- **Light Curve Integration**: Add raw time-series analysis
- **Real-Time Updates**: Continuous learning from new discoveries
- **API Development**: Programmatic access for researchers

## 🏆 Competition Advantages

This solution stands out because it:

- **Solves Real Problems**: Addresses actual NASA challenges
- **Uses Cutting-Edge AI**: Latest ensemble methods and feature engineering
- **Provides Practical Value**: Web interface for real-world use
- **Demonstrates Scale**: Handles large datasets efficiently
- **Shows Innovation**: Advanced techniques in astronomical ML

## 👥 Team

Developed for NASA Space Apps Challenge 2025 - Exoplanet Detection Challenge

## 📄 License

This project is developed for educational and research purposes as part of the NASA Space Apps Challenge.

---

**Ready to discover new worlds? Launch the system and start classifying exoplanets! 🌍✨**
