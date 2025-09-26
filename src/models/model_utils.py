import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
import joblib
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelUtils:
    """Utility functions for model management and evaluation"""
    
    @staticmethod
    def save_model_artifacts(model, 
                           model_path: str,
                           model_name: str,
                           feature_names: List[str],
                           metadata: Optional[Dict[str, Any]] = None):
        """Save model and related artifacts
        
        Args:
            model: Trained model object
            model_path: Directory to save artifacts
            model_name: Name of the model
            feature_names: List of feature names
            metadata: Additional metadata
        """
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_path, f"{model_name}.pkl")
        joblib.dump(model, model_file)
        
        # Save feature names
        features_file = os.path.join(model_path, f"{model_name}_features.pkl")
        joblib.dump(feature_names, features_file)
        
        # Save metadata
        if metadata:
            metadata_file = os.path.join(model_path, f"{model_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model artifacts saved for {model_name} in {model_path}")
    
    @staticmethod
    def load_model_artifacts(model_path: str, 
                           model_name: str) -> Tuple[Any, List[str], Dict[str, Any]]:
        """Load model and related artifacts
        
        Args:
            model_path: Directory containing artifacts
            model_name: Name of the model
            
        Returns:
            Tuple of (model, feature_names, metadata)
        """
        # Load model
        model_file = os.path.join(model_path, f"{model_name}.pkl")
        model = joblib.load(model_file)
        
        # Load feature names
        features_file = os.path.join(model_path, f"{model_name}_features.pkl")
        feature_names = joblib.load(features_file) if os.path.exists(features_file) else []
        
        # Load metadata
        metadata_file = os.path.join(model_path, f"{model_name}_metadata.json")
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Model artifacts loaded for {model_name} from {model_path}")
        return model, feature_names, metadata
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray, 
                                      y_pred: np.ndarray,
                                      y_pred_proba: Optional[np.ndarray] = None,
                                      pos_label: int = 1) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            pos_label: Positive class label
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        # Rates
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Probabilities-based metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = np.trapz(precision_curve, recall_curve)
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5,
                           scoring: str = 'roc_auc',
                           random_state: int = 42) -> Dict[str, float]:
        """Perform cross-validation on model
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            scoring: Scoring metric
            random_state: Random state for reproducibility
            
        Returns:
            Cross-validation results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            f'cv_{scoring}_mean': float(np.mean(scores)),
            f'cv_{scoring}_std': float(np.std(scores)),
            f'cv_{scoring}_min': float(np.min(scores)),
            f'cv_{scoring}_max': float(np.max(scores)),
            'cv_scores': scores.tolist()
        }
        
        return results
    
    @staticmethod
    def plot_model_performance(y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive model performance charts
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        axes[0, 1].plot(recall, precision, color='blue', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.2f})')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc="lower left")
        axes[0, 1].grid(True)
        
        # Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, 
                       label='Non-Fraud', color='green', density=True)
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5,
                       label='Fraud', color='red', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confusion Matrix
        y_pred = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def calculate_feature_stability(feature_df_train: pd.DataFrame,
                                  feature_df_test: pd.DataFrame,
                                  method: str = 'psi') -> Dict[str, float]:
        """Calculate feature stability between train and test sets
        
        Args:
            feature_df_train: Training features
            feature_df_test: Test features  
            method: Stability metric ('psi' for Population Stability Index)
            
        Returns:
            Dictionary of stability scores per feature
        """
        stability_scores = {}
        
        for feature in feature_df_train.columns:
            if feature in feature_df_test.columns:
                train_values = feature_df_train[feature].dropna()
                test_values = feature_df_test[feature].dropna()
                
                if method == 'psi':
                    # Population Stability Index
                    score = ModelUtils._calculate_psi(train_values, test_values)
                    stability_scores[feature] = score
        
        return stability_scores
    
    @staticmethod
    def _calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index
        
        Args:
            expected: Expected distribution (training)
            actual: Actual distribution (test)
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        try:
            # Create bins based on expected distribution
            _, bin_edges = pd.cut(expected, bins=bins, retbins=True, duplicates='drop')
            
            # Calculate distributions
            expected_dist = pd.cut(expected, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
            actual_dist = pd.cut(actual, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
            
            # Align distributions
            expected_dist = expected_dist.reindex(actual_dist.index, fill_value=0.001)
            actual_dist = actual_dist.fillna(0.001)
            
            # Calculate PSI
            psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
            
            return float(psi)
        except:
            return 0.0

class ModelValidator:
    """Model validation and testing utilities"""
    
    def __init__(self, performance_thresholds: Optional[Dict[str, float]] = None):
        """Initialize model validator
        
        Args:
            performance_thresholds: Minimum performance thresholds
        """
        self.performance_thresholds = performance_thresholds or {
            'precision': 0.90,
            'recall': 0.85,
            'f1_score': 0.87,
            'roc_auc': 0.90,
            'false_positive_rate': 0.05
        }
        
        self.validation_results = {}
        
    def validate_model_performance(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate model performance against thresholds
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Validation results
        """
        # Calculate metrics
        metrics = ModelUtils.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        
        # Check against thresholds
        validation_results = {
            'metrics': metrics,
            'threshold_checks': {},
            'overall_pass': True
        }
        
        for metric, threshold in self.performance_thresholds.items():
            if metric in metrics:
                actual_value = metrics[metric]
                
                if metric == 'false_positive_rate':
                    # For FPR, we want it to be below threshold
                    passes = actual_value <= threshold
                else:
                    # For other metrics, we want them to be above threshold
                    passes = actual_value >= threshold
                
                validation_results['threshold_checks'][metric] = {
                    'actual': actual_value,
                    'threshold': threshold,
                    'passes': passes
                }
                
                if not passes:
                    validation_results['overall_pass'] = False
        
        self.validation_results = validation_results
        return validation_results
    
    def validate_model_stability(self,
                               model,
                               X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               stability_threshold: float = 0.2) -> Dict[str, Any]:
        """Validate model stability across different datasets
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            stability_threshold: PSI threshold for stability
            
        Returns:
            Stability validation results
        """
        # Feature stability
        feature_stability = ModelUtils.calculate_feature_stability(X_train, X_test)
        
        # Prediction stability
        try:
            train_predictions = model.predict_proba(X_train)[:, 1]
            test_predictions = model.predict_proba(X_test)[:, 1]
            
            prediction_stability = ModelUtils._calculate_psi(
                pd.Series(train_predictions), 
                pd.Series(test_predictions)
            )
        except:
            prediction_stability = 0.0
        
        # Check stability
        unstable_features = [
            feature for feature, psi in feature_stability.items() 
            if psi > stability_threshold
        ]
        
        stability_results = {
            'feature_stability': feature_stability,
            'prediction_stability': prediction_stability,
            'unstable_features': unstable_features,
            'stability_threshold': stability_threshold,
            'overall_stable': len(unstable_features) == 0 and prediction_stability <= stability_threshold
        }
        
        return stability_results
    
    def validate_data_quality(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate data quality for model training
        
        Args:
            X: Features DataFrame
            y: Labels Series
            
        Returns:
            Data quality validation results
        """
        quality_results = {
            'data_shape': X.shape,
            'missing_values': {},
            'data_types': {},
            'class_distribution': {},
            'issues': [],
            'overall_quality': True
        }
        
        # Missing values check
        missing_counts = X.isnull().sum()
        quality_results['missing_values'] = missing_counts.to_dict()
        
        # High missing value features
        high_missing = missing_counts[missing_counts > len(X) * 0.3]  # >30% missing
        if len(high_missing) > 0:
            quality_results['issues'].append(f"Features with >30% missing values: {list(high_missing.index)}")
            quality_results['overall_quality'] = False
        
        # Data types
        quality_results['data_types'] = X.dtypes.astype(str).to_dict()
        
        # Class distribution
        class_dist = y.value_counts(normalize=True).to_dict()
        quality_results['class_distribution'] = class_dist
        
        # Class imbalance check
        minority_class_ratio = min(class_dist.values())
        if minority_class_ratio < 0.01:  # <1% minority class
            quality_results['issues'].append(f"Severe class imbalance: {minority_class_ratio:.3f}")
        
        # Feature variability
        low_variance_features = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].var() < 1e-8:
                low_variance_features.append(col)
        
        if low_variance_features:
            quality_results['issues'].append(f"Low variance features: {low_variance_features}")
        
        return quality_results
    
    def generate_validation_report(self, output_path: str):
        """Generate comprehensive validation report
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'summary': {
                'performance_validation': self.validation_results.get('overall_pass', False),
                'total_checks': len(self.validation_results.get('threshold_checks', {})),
                'failed_checks': sum(
                    1 for check in self.validation_results.get('threshold_checks', {}).values() 
                    if not check.get('passes', True)
                )
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def get_validation_summary(self) -> str:
        """Get a summary of validation results as string
        
        Returns:
            Formatted validation summary
        """
        if not self.validation_results:
            return "No validation results available"
        
        summary_lines = ["Model Validation Summary", "=" * 25]
        
        # Overall status
        overall_pass = self.validation_results.get('overall_pass', False)
        status = "✅ PASS" if overall_pass else "❌ FAIL"
        summary_lines.append(f"Overall Status: {status}")
        
        # Detailed results
        threshold_checks = self.validation_results.get('threshold_checks', {})
        for metric, check in threshold_checks.items():
            actual = check['actual']
            threshold = check['threshold']
            passes = check['passes']
            
            status_icon = "✅" if passes else "❌"
            summary_lines.append(f"{status_icon} {metric}: {actual:.4f} (threshold: {threshold:.4f})")
        
        return "\n".join(summary_lines)
