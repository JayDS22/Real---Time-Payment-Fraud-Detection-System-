import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import yaml
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model combining Random Forest, XGBoost, and LightGBM"""
    
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        """Initialize ensemble model with configuration
        
        Args:
            config_path: Path to model configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.model_weights = {}
        self.feature_names = []
        self.is_fitted = False
        self.explainer = None
        
        # Initialize model weights
        if self.config['model']['ensemble']['enabled']:
            self.model_weights = self.config['model']['ensemble']['weights']
        
        logger.info("EnsembleModel initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            'model': {
                'ensemble': {
                    'enabled': True,
                    'weights': {'random_forest': 0.3, 'xgboost': 0.4, 'lightgbm': 0.3}
                },
                'random_forest': {
                    'n_estimators': 300,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 500,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'EnsembleModel':
        """Fit the ensemble model
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble model on {len(X)} samples with {X.shape[1]} features")
        
        self.feature_names = X.columns.tolist()
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_params = self.config['model']['random_forest']
        self.models['random_forest'] = RandomForestClassifier(**rf_params)
        self.models['random_forest'].fit(X, y)
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_params = self.config['model']['xgboost']
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.models['xgboost'] = xgb.XGBClassifier(
                **xgb_params,
                eval_metric='auc',
                early_stopping_rounds=50,
                verbosity=0
            )
            self.models['xgboost'].fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.models['xgboost'] = xgb.XGBClassifier(**xgb_params)
            self.models['xgboost'].fit(X, y)
        
        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_params = self.config['model']['lightgbm']
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.models['lightgbm'] = lgb.LGBMClassifier(
                **lgb_params,
                metric='auc',
                early_stopping_round=50,
                verbosity=-1
            )
            self.models['lightgbm'].fit(X, y, eval_set=eval_set, eval_names=['validation'])
        else:
            self.models['lightgbm'] = lgb.LGBMClassifier(**lgb_params)
            self.models['lightgbm'].fit(X, y)
        
        self.is_fitted = True
        
        # Initialize explainer for the best performing model (XGBoost)
        try:
            self.explainer = shap.TreeExplainer(self.models['xgboost'])
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
        
        logger.info("Ensemble model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary fraud labels
        
        Args:
            X: Features for prediction
            
        Returns:
            Binary predictions (0/1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of probabilities [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure feature alignment
        if self.feature_names:
            X = X[self.feature_names]
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(X)
                predictions[model_name] = pred_proba[:, 1]  # Fraud probability
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                predictions[model_name] = np.zeros(len(X))
        
        # Ensemble prediction using weighted average
        if self.config['model']['ensemble']['enabled'] and len(predictions) > 1:
            ensemble_pred = np.zeros(len(X))
            total_weight = 0
            
            for model_name, prob in predictions.items():
                weight = self.model_weights.get(model_name, 0.33)
                ensemble_pred += weight * prob
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            # Return in sklearn format
            return np.column_stack([1 - ensemble_pred, ensemble_pred])
        
        else:
            # Use single best model (XGBoost by default)
            best_model = 'xgboost' if 'xgboost' in predictions else list(predictions.keys())[0]
            single_pred = predictions[best_model]
            return np.column_stack([1 - single_pred, single_pred])
    
    def get_feature_importance(self, 
                             importance_type: str = 'gain',
                             normalize: bool = True) -> Dict[str, float]:
        """Get ensemble feature importance
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'weight')
            normalize: Whether to normalize importances to sum to 1
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        ensemble_importance = {}
        
        for model_name, model in self.models.items():
            weight = self.model_weights.get(model_name, 0.33)
            
            if hasattr(model, 'feature_importances_'):
                # For sklearn models (Random Forest)
                importance = model.feature_importances_
            elif hasattr(model, 'get_booster'):
                # For XGBoost
                importance_dict = model.get_booster().get_score(importance_type=importance_type)
                importance = np.array([importance_dict.get(f'f{i}', 0) for i in range(len(self.feature_names))])
            elif hasattr(model, 'booster_'):
                # For LightGBM
                importance = model.booster_.feature_importance(importance_type=importance_type)
            else:
                continue
            
            # Add weighted importance to ensemble
            for i, feature_name in enumerate(self.feature_names):
                if feature_name not in ensemble_importance:
                    ensemble_importance[feature_name] = 0
                if i < len(importance):
                    ensemble_importance[feature_name] += weight * importance[i]
        
        # Normalize if requested
        if normalize and ensemble_importance:
            total_importance = sum(ensemble_importance.values())
            if total_importance > 0:
                ensemble_importance = {k: v / total_importance for k, v in ensemble_importance.items()}
        
        return ensemble_importance
    
    def explain_prediction(self, 
                          X: pd.DataFrame, 
                          max_features: int = 10) -> List[Dict[str, Any]]:
        """Explain predictions using SHAP values
        
        Args:
            X: Features to explain
            max_features: Maximum number of features to include in explanation
