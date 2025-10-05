#!/usr/bin/env python3
"""
NASA Space Apps Challenge - Exoplanet Detection Ensemble Model
XGBoost + LightGBM + CatBoost Ensemble for KOI Dataset


DISCLAIMER AND USAGE TERMS:
- This code was developed during the NASA Space Apps Challenge hackathon
- It is provided as-is for educational and research purposes
- While I strive for accuracy, this was a time-constrained project and may contain errors
- Results should be validated before use in scientific publications
- Feel free to use this code with proper attribution to the NASA Space Apps team
- Credit:Alhagie Boye & NASA Space Apps Challenge 2025 - Banjul, The Gambia

LICENSE: Open source - please provide attribution when using this code.

Author: Alhagie Boye
NASA Space Apps Challenge 2025 - Banjul, The Gambia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

class KOIEnsembleModel:
    """
    Advanced ensemble model for exoplanet classification using KOI dataset
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def load_data(self, filepath):
        """Load and preprocess KOI dataset"""
        print("Loading KOI dataset...")
        
        # Load data with comment handling
        self.df = pd.read_csv(filepath, comment='#')
        print(f"Loaded {len(self.df)} observations with {len(self.df.columns)} features")
        
        return self.df
    
    def advanced_feature_engineering(self):
        """Create some extra features that might help"""
        print("Creating advanced features...")
        
        df = self.df.copy()
        
        # some ratios that might be useful
        df['radius_temp_ratio'] = df['koi_prad'] / df['koi_teq']
        df['period_depth_ratio'] = df['koi_period'] / (df['koi_depth'] + 1e-6)
        df['duration_impact_ratio'] = df['koi_duration'] / (df['koi_impact'] + 1e-6)
        df['snr_depth_interaction'] = df['koi_model_snr'] * np.log(df['koi_depth'] + 1)
        
        # star stuff
        df['stellar_density'] = df['koi_smass'] / (df['koi_srad'] ** 3 + 1e-6)
        df['stellar_luminosity'] = df['koi_srad'] ** 2 * df['koi_steff'] ** 4
        
        # transit stuff
        df['transit_probability'] = df['koi_srad'] / (df['koi_sma'] + 1e-6)
        df['transit_duration_norm'] = df['koi_duration'] / df['koi_period']
        
        # signal quality
        df['signal_strength'] = df['koi_model_snr'] * df['koi_depth']
        df['detection_confidence'] = df['koi_model_snr'] / (df['koi_model_snr'] + 10)
        
        # log and sqrt transforms
        numerical_cols = ['koi_period', 'koi_depth', 'koi_duration', 'koi_model_snr']
        for col in numerical_cols:
            if col in df.columns:
                df[f'{col}_log'] = np.log(df[col] + 1)
                df[f'{col}_sqrt'] = np.sqrt(df[col] + 1)
        
        # error ratios
        error_cols = [col for col in df.columns if '_err1' in col or '_err2' in col]
        for error_col in error_cols:
            base_col = error_col.replace('_err1', '').replace('_err2', '')
            if base_col in df.columns:
                df[f'{base_col}_error_ratio'] = df[error_col] / (df[base_col] + 1e-6)
        
        self.df_engineered = df
        print(f"Created {len(df.columns) - len(self.df.columns)} new features")
        
        return df
    
    def prepare_features(self):
        """Get features ready for training"""
        print("Preparing features for training...")
        
        df = self.df_engineered.copy()
        
        # pick the features we want to use
        key_features = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_impact', 'koi_ror',
            'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_insol',
            'koi_steff', 'koi_srad', 'koi_smass', 'koi_slogg', 'koi_smet',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
            'radius_temp_ratio', 'period_depth_ratio', 'duration_impact_ratio',
            'snr_depth_interaction', 'stellar_density', 'transit_probability',
            'transit_duration_norm', 'signal_strength', 'detection_confidence',
            'koi_period_log', 'koi_depth_log', 'koi_duration_log', 'koi_model_snr_log'
        ]
        
        # only use features that actually exist
        available_features = [f for f in key_features if f in df.columns]
        
        X = df[available_features].copy()
        
        # fill missing values
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
        
        y = df['koi_disposition'].copy()
        
        print(f"Selected {len(available_features)} features")
        print(f"Features: {available_features[:10]}...")  # Show first 10
        
        return X, y, available_features
    
    def train_individual_models(self, X, y, available_features):
        """Train the individual models"""
        print("Training individual models...")
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # XGBoost
        print("   Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train_encoded)
        
        # LightGBM
        print("   Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train_encoded)
        
        # CatBoost
        print("   Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        cb_model.fit(X_train, y_train_encoded)
        
        self.models = {
            'XGBoost': xgb_model,
            'LightGBM': lgb_model,
            'CatBoost': cb_model
        }
        
        # check how they did
        print("Evaluating individual models...")
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            print(f"   {name}: {accuracy:.4f} accuracy")
            
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(available_features, model.feature_importances_))
        
        return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded
    
    def create_ensemble(self, X_train, X_test, y_train_encoded, y_test_encoded):
        """Combine the models into an ensemble"""
        print("Creating ensemble model...")
        
        base_models = [
            ('xgb', self.models['XGBoost']),
            ('lgb', self.models['LightGBM']),
            ('cat', self.models['CatBoost'])
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        self.ensemble_model.fit(X_train, y_train_encoded)
        
        y_pred_ensemble = self.ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test_encoded, y_pred_ensemble)
        
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        return y_pred_ensemble, ensemble_accuracy
    
    def generate_report(self, X_test, y_test_encoded, y_pred_ensemble):
        """Generate the final report"""
        print("Generating evaluation report...")
        
        print("\nFINAL RESULTS:")
        print("="*50)
        print(classification_report(y_test_encoded, y_pred_ensemble, 
                                  target_names=self.label_encoder.classes_))
        
        cm = confusion_matrix(y_test_encoded, y_pred_ensemble)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Ensemble Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if self.feature_importance:
            self.plot_feature_importance()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (model_name, importance_dict) in enumerate(self.feature_importance.items()):
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
            features, importances = zip(*sorted_features)
            
            axes[i].barh(range(len(features)), importances)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_title(f'{model_name} - Feature Importance')
            axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_new_data(self, new_data):
        """Make predictions on new data"""
        if self.ensemble_model is None:
            raise ValueError("Model not trained yet!")
        
        processed_data = self._preprocess_new_data(new_data)
        prediction = self.ensemble_model.predict(processed_data)
        probability = self.ensemble_model.predict_proba(processed_data)
        prediction_label = self.label_encoder.inverse_transform(prediction)
        
        return prediction_label[0], probability[0]

def main():
    """Run the whole thing"""
    print("NASA Space Apps Challenge - Exoplanet Detection")
    print("="*60)
    
    model = KOIEnsembleModel()
    
    df = model.load_data('data/koi.csv')
    df_engineered = model.advanced_feature_engineering()
    X, y, features = model.prepare_features()
    
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = model.train_individual_models(X, y, features)
    y_pred_ensemble, ensemble_accuracy = model.create_ensemble(X_train, X_test, y_train_encoded, y_test_encoded)
    model.generate_report(X_test, y_test_encoded, y_pred_ensemble)
    
    print(f"\nMODEL TRAINING COMPLETE!")
    print(f"Final Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"Results saved as: confusion_matrix.png, feature_importance.png")
    
    return model

if __name__ == "__main__":
    model = main()
