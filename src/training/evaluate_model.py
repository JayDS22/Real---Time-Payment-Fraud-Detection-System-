import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import joblib
import json
import os
from datetime import datetime
import shap

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, output_dir: str = 'reports'):
        """Initialize model evaluator
        
        Args:
            output_dir: Directory to save evaluation reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.evaluation_results = {}
        self.model_comparisons = {}
        
    def evaluate_single_model(self, 
                             model, 
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             model_name: str = "model") -> Dict[str, Any]:
        """Evaluate a single model comprehensively
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating {model_name} on {len(X_test)} test samples")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        results = {
            'model_name': model_name,
            'test_samples': len(X_test),
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': self._calculate_all_metrics(y_test, y_pred, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
            results['feature_importance'] = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20])  # Top 20 features
        
        # Store results
        self.evaluation_results[model_name] = results
        
        # Generate visualizations
        self._create_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)
        
        # Generate SHAP explanations if possible
        try:
            self._generate_shap_analysis(model, X_test.sample(min(1000, len(X_test))), model_name)
        except Exception as e:
            logger.warning(f"Could not generate SHAP analysis for {model_name}: {e}")
        
        logger.info(f"Evaluation completed for {model_name}")
        return results
    
    def evaluate_ensemble_models(self, 
                                models: Dict[str, Any],
                                X_test: pd.DataFrame, 
                                y_test: pd.Series,
                                ensemble_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Evaluate ensemble of models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            ensemble_weights: Weights for ensemble combination
            
        Returns:
            Ensemble evaluation results
        """
        logger.info(f"Evaluating ensemble of {len(models)} models")
        
        # Evaluate individual models
        individual_results = {}
        ensemble_predictions = []
        
        for model_name, model in models.items():
            individual_results[model_name] = self.evaluate_single_model(
                model, X_test, y_test, model_name
            )
            
            # Get predictions for ensemble
            pred_proba = model.predict_proba(X_test)[:, 1]
            ensemble_predictions.append(pred_proba)
        
        # Calculate ensemble prediction
        if ensemble_weights:
            weights = [ensemble_weights.get(name, 1.0) for name in models.keys()]
        else:
            weights = [1.0] * len(models)
        
        weights = np.array(weights) / sum(weights)  # Normalize weights
        ensemble_pred_proba = np.average(ensemble_predictions, axis=0, weights=weights)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        # Evaluate ensemble
        ensemble_results = {
            'model_name': 'ensemble',
            'individual_models': list(models.keys()),
            'ensemble_weights': dict(zip(models.keys(), weights)) if ensemble_weights else None,
            'test_samples': len(X_test),
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': self._calculate_all_metrics(y_test, ensemble_pred, ensemble_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred).tolist(),
            'classification_report': classification_report(y_test, ensemble_pred, output_dict=True),
            'individual_results': individual_results
        }
        
        # Create ensemble-specific plots
        self._create_ensemble_plots(y_test, ensemble_pred, ensemble_pred_proba, individual_results)
        
        # Model comparison analysis
        self._create_model_comparison_analysis(individual_results, ensemble_results)
        
        self.evaluation_results['ensemble'] = ensemble_results
        return ensemble_results
    
    def _calculate_all_metrics(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            
            # Probability-based metrics
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
        }
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics.update({
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'false_discovery_rate': fp / (fp + tp) if (fp + tp) > 0 else 0,
            'matthews_correlation_coeff': self._calculate_mcc(tp, tn, fp, fn)
        })
        
        # Business metrics for fraud detection
        metrics.update({
            'fraud_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'precision_at_1_percent': self._precision_at_k_percent(y_true, y_pred_proba, 0.01),
            'precision_at_5_percent': self._precision_at_k_percent(y_true, y_pred_proba, 0.05)
        })
        
        return metrics
    
    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Calculate Matthews Correlation Coefficient"""
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return 0.0
        return (tp * tn - fp * fn) / denominator
    
    def _precision_at_k_percent(self, y_true: np.ndarray, y_pred_proba: np.ndarray, k: float) -> float:
        """Calculate precision at top k% of predictions"""
        n_samples = int(len(y_true) * k)
        if n_samples == 0:
            return 0.0
        
        # Get indices of top k% predictions
        top_k_indices = np.argsort(y_pred_proba)[-n_samples:]
        
        # Calculate precision for top k%
        top_k_true = y_true[top_k_indices]
        return np.mean(top_k_true) if len(top_k_true) > 0 else 0.0
    
    def _create_evaluation_plots(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray,
                                model_name: str):
        """Create evaluation plots for a single model"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
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
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, color='blue', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc="lower left")
        axes[0, 1].grid(True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'])
        axes[0, 2].set_title('Confusion Matrix')
        axes[0, 2].set_ylabel('True Label')
        axes[0, 2].set_xlabel('Predicted Label')
        
        # Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7,
                       label='Non-Fraud', color='green', density=True)
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7,
                       label='Fraud', color='red', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Threshold Analysis
        thresholds = np.arange(0.1, 1.0, 0.05)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba > thresh).astype(int)
            precision_scores.append(precision_score(y_true, y_pred_thresh, zero_division=0))
            recall_scores.append(recall_score(y_true, y_pred_thresh, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
        
        axes[1, 1].plot(thresholds, precision_scores, label='Precision', marker='o')
        axes[1, 1].plot(thresholds, recall_scores, label='Recall', marker='s')
        axes[1, 1].plot(thresholds, f1_scores, label='F1 Score', marker='^')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Threshold Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Cumulative Gains Chart
        sorted_indices = np.argsort(y_pred_proba)[::-1]  # Sort by probability desc
        y_true_sorted = y_true[sorted_indices]
        gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        percentiles = np.arange(1, len(gains) + 1) / len(gains)
        
        axes[1, 2].plot(percentiles * 100, gains * 100, label='Model', linewidth=2)
        axes[1, 2].plot([0, 100], [0, 100], 'k--', label='Random')
        axes[1, 2].set_xlabel('% of Population')
        axes[1, 2].set_ylabel('% of Fraud Caught')
        axes[1, 2].set_title('Cumulative Gains Chart')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'{model_name}_evaluation_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved: {plot_path}")
    
    def _create_ensemble_plots(self, 
                              y_true: np.ndarray,
                              ensemble_pred: np.ndarray,
                              ensemble_pred_proba: np.ndarray,
                              individual_results: Dict[str, Any]):
        """Create ensemble-specific evaluation plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ensemble Model Analysis', fontsize=16)
        
        # Model Performance Comparison
        models = list(individual_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        comparison_data = []
        for metric in metrics:
            metric_values = [individual_results[model]['metrics'][metric] for model in models]
            # Add ensemble performance
            ensemble_metric = getattr(self, f'_calculate_{metric}', lambda x, y, z: 0)(
                y_true, ensemble_pred, ensemble_pred_proba
            ) if metric == 'roc_auc' else getattr(self, f'_calculate_{metric}', lambda x, y: 0)(
                y_true, ensemble_pred
            )
            metric_values.append(ensemble_metric)
            comparison_data.append(metric_values)
        
        model_names = models + ['Ensemble']
        comparison_df = pd.DataFrame(comparison_data, columns=model_names, index=metrics)
        
        sns.heatmap(comparison_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 0])
        axes[0, 0].set_title('Model Performance Comparison')
        
        # Ensemble vs Individual ROC Curves
        for model_name in models:
            model_metrics = individual_results[model_name]['metrics']
            # Simplified - would need actual predictions for full ROC curve
            axes[0, 1].plot([0, model_metrics['false_positive_rate'], 1], 
                           [0, model_metrics['true_positive_rate'], 1],
                           label=f'{model_name} (AUC={model_metrics["roc_auc"]:.3f})',
                           alpha=0.7)
        
        # Ensemble ROC
        fpr, tpr, _ = roc_curve(y_true, ensemble_pred_proba)
        roc_auc = roc_auc_score(y_true, ensemble_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='black', lw=3,
                       label=f'Ensemble (AUC={roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Prediction Agreement Analysis
        agreement_data = []
        for i, model1 in enumerate
