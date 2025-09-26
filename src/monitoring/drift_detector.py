import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime, timedelta
import redis
import joblib

logger = logging.getLogger(__name__)

class DataDriftDetector:
    """Detect data drift in incoming features"""
    
    def __init__(self, reference_data: pd.DataFrame, redis_client=None):
        self.reference_data = reference_data
        self.redis_client = redis_client
        self.drift_thresholds = {
            'ks_test': 0.05,
            'js_divergence': 0.15,
            'psi': 0.25
        }
        
    def detect_drift(self, current_data: pd.DataFrame, method='js_divergence') -> Dict[str, Any]:
        """Detect drift between reference and current data"""
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'features_analyzed': [],
            'drifted_features': [],
            'drift_scores': {},
            'overall_drift': False
        }
        
        common_features = set(self.reference_data.columns) & set(current_data.columns)
        
        for feature in common_features:
            if self.reference_data[feature].dtype in ['float64', 'int64']:
                
                if method == 'ks_test':
                    drift_score = self._ks_test_drift(
                        self.reference_data[feature], 
                        current_data[feature]
                    )
                elif method == 'js_divergence':
                    drift_score = self._js_divergence_drift(
                        self.reference_data[feature],
                        current_data[feature]
                    )
                elif method == 'psi':
                    drift_score = self._psi_drift(
                        self.reference_data[feature],
                        current_data[feature]
                    )
                else:
                    continue
                
                drift_results['features_analyzed'].append(feature)
                drift_results['drift_scores'][feature] = drift_score
                
                threshold = self.drift_thresholds.get(method, 0.15)
                if drift_score > threshold:
                    drift_results['drifted_features'].append(feature)
        
        drift_results['overall_drift'] = len(drift_results['drifted_features']) > 0
        
        # Store in Redis if available
        if self.redis_client:
            self.redis_client.setex(
                f"drift_detection:{datetime.now().strftime('%Y%m%d_%H')}",
                3600,  # 1 hour TTL
                json.dumps(drift_results, default=str)
            )
        
        return drift_results
    
    def _ks_test_drift(self, ref_data: pd.Series, current_data: pd.Series) -> float:
        """Kolmogorov-Smirnov test for distribution drift"""
        try:
            statistic, p_value = stats.ks_2samp(
                ref_data.dropna(), 
                current_data.dropna()
            )
            return float(statistic)
        except:
            return 0.0
    
    def _js_divergence_drift(self, ref_data: pd.Series, current_data: pd.Series) -> float:
        """Jensen-Shannon divergence for drift detection"""
        try:
            # Create histograms
            min_val = min(ref_data.min(), current_data.min())
            max_val = max(ref_data.max(), current_data.max())
            bins = np.linspace(min_val, max_val, 50)
            
            ref_hist, _ = np.histogram(ref_data.dropna(), bins=bins, density=True)
            current_hist, _ = np.histogram(current_data.dropna(), bins=bins, density=True)
            
            # Normalize
            ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
            current_hist = current_hist / (current_hist.sum() + 1e-10)
            
            # Add small epsilon to avoid log(0)
            ref_hist = ref_hist + 1e-10
            current_hist = current_hist + 1e-10
            
            # Jensen-Shannon divergence
            m = (ref_hist + current_hist) / 2
            js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(current_hist, m)
            
            return float(np.sqrt(js_div))
        except:
            return 0.0
    
    def _psi_drift(self, ref_data: pd.Series, current_data: pd.Series) -> float:
        """Population Stability Index for drift detection"""
        try:
            # Create bins based on reference data quantiles
            _, bins = pd.qcut(ref_data.dropna(), q=10, retbins=True, duplicates='drop')
            
            # Calculate expected and actual distributions
            expected_dist = pd.cut(ref_data, bins=bins, include_lowest=True).value_counts(normalize=True)
            actual_dist = pd.cut(current_data, bins=bins, include_lowest=True).value_counts(normalize=True)
            
            # Align and handle missing values
            expected_dist = expected_dist.reindex(actual_dist.index, fill_value=0.001)
            actual_dist = actual_dist.fillna(0.001)
            
            # Calculate PSI
            psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()
            
            return float(abs(psi))
        except:
            return 0.0

class ModelDriftDetector:
    """Detect model performance drift"""
    
    def __init__(self, model, reference_performance: Dict[str, float], redis_client=None):
        self.model = model
        self.reference_performance = reference_performance
        self.redis_client = redis_client
        self.performance_threshold = 0.05  # 5% degradation threshold
        
    def detect_performance_drift(self, X_current: pd.DataFrame, y_current: pd.Series) -> Dict[str, Any]:
        """Detect model performance drift"""
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            # Get current predictions
            y_pred = self.model.predict(X_current)
            y_pred_proba = self.model.predict_proba(X_current)[:, 1]
            
            # Calculate current performance
            current_performance = {
                'accuracy': accuracy_score(y_current, y_pred),
                'precision': precision_score(y_current, y_pred, zero_division=0),
                'recall': recall_score(y_current, y_pred, zero_division=0),
                'f1_score': f1_score(y_current, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_current, y_pred_proba)
            }
            
            # Compare with reference performance
            drift_results = {
                'timestamp': datetime.now().isoformat(),
                'reference_performance': self.reference_performance,
                'current_performance': current_performance,
                'performance_degradation': {},
                'drifted_metrics': [],
                'overall_drift': False
            }
            
            for metric in current_performance:
                if metric in self.reference_performance:
                    ref_value = self.reference_performance[metric]
                    current_value = current_performance[metric]
                    degradation = (ref_value - current_value) / ref_value
                    
                    drift_results['performance_degradation'][metric] = degradation
                    
                    if degradation > self.performance_threshold:
                        drift_results['drifted_metrics'].append(metric)
            
            drift_results['overall_drift'] = len(drift_results['drifted_metrics']) > 0
            
            # Store results
            if self.redis_client:
                self.redis_client.setex(
                    f"model_drift:{datetime.now().strftime('%Y%m%d_%H')}",
                    3600,
                    json.dumps(drift_results, default=str)
                )
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting model drift: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def should_retrain(self, drift_results: Dict[str, Any]) -> bool:
        """Determine if model should be retrained based on drift"""
        
        if drift_results.get('overall_drift', False):
            # Check severity of drift
            drifted_metrics = drift_results.get('drifted_metrics', [])
            
            # Critical metrics that require immediate retraining
            critical_metrics = ['precision', 'recall', 'f1_score']
            
            critical_drift = any(metric in critical_metrics for metric in drifted_metrics)
            
            if critical_drift:
                return True
            
            # Check if multiple metrics are drifting
            if len(drifted_metrics) >= 3:
                return True
        
        return False
