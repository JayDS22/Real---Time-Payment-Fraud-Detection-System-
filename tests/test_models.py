import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import tempfile
import shutil
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ensemble_model import EnsembleModel, ModelPredictor
from models.model_utils import ModelUtils, ModelValidator

class TestEnsembleModel:
    """Test ensemble model functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            flip_y=0.01,
            class_sep=0.8,
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(20)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        return X_df, y_series
    
    @pytest.fixture
    def ensemble_model(self):
        """Create ensemble model instance"""
        return EnsembleModel()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_initialization(self, ensemble_model):
        """Test model initialization"""
        assert ensemble_model.models == {}
        assert ensemble_model.feature_names == []
        assert not ensemble_model.is_fitted
        assert isinstance(ensemble_model.model_weights, dict)
    
    def test_model_training(self, ensemble_model, sample_data):
        """Test model training"""
        X, y = sample_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        ensemble_model.fit(X_train, y_train, X_test, y_test)
        
        # Verify training
        assert ensemble_model.is_fitted
        assert len(ensemble_model.models) == 3  # RF, XGB, LGB
        assert ensemble_model.feature_names == X_train.columns.tolist()
        
        # Check individual models
        assert 'random_forest' in ensemble_model.models
        assert 'xgboost' in ensemble_model.models  
        assert 'lightgbm' in ensemble_model.models
    
    def test_predictions(self, ensemble_model, sample_data):
        """Test model predictions"""
        X, y = sample_data
        
        # Train first
        ensemble_model.fit(X, y)
        
        # Test predictions
        predictions = ensemble_model.predict(X[:10])
        probabilities = ensemble_model.predict_proba(X[:10])
        
        # Verify prediction format
        assert len(predictions) == 10
        assert predictions.dtype in [np.int64, np.int32, int]
        assert all(pred in [0, 1] for pred in predictions)
        
        # Verify probability format
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
    
    def test_feature_importance(self, ensemble_model, sample_data):
        """Test feature importance extraction"""
        X, y = sample_data
        
        # Train first
        ensemble_model.fit(X, y)
        
        # Get feature importance
        importance = ensemble_model.get_feature_importance()
        
        # Verify importance structure
        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
        assert all(isinstance(imp, float) for imp in importance.values())
        assert all(imp >= 0 for imp in importance.values())
    
    def test_explanation_generation(self, ensemble_model, sample_data):
        """Test prediction explanations"""
        X, y = sample_data
        
        # Train first
        ensemble_model.fit(X, y)
        
        # Get explanations for small sample
        explanations = ensemble_model.explain_prediction(X[:5])
        
        # Verify explanation structure
        assert len(explanations) == 5
        for explanation in explanations:
            assert 'top_features' in explanation
            assert isinstance(explanation['top_features'], list)
            if explanation['top_features']:
                assert 'feature' in explanation['top_features'][0]
                assert 'contribution' in explanation['top_features'][0]
    
    def test_model_evaluation(self, ensemble_model, sample_data):
        """Test model evaluation"""
        X, y = sample_data
        
        # Train first
        ensemble_model.fit(X, y)
        
        # Evaluate
        metrics = ensemble_model.evaluate(X, y)
        
        # Verify metrics structure
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0 <= metrics[metric] <= 1
    
    def test_model_save_load(self, ensemble_model, sample_data, temp_dir):
        """Test model saving and loading"""
        X, y = sample_data
        
        # Train model
        ensemble_model.fit(X, y)
        original_predictions = ensemble_model.predict_proba(X[:10])
        
        # Save model
        ensemble_model.save_model(temp_dir, save_to_mlflow=False)
        
        # Verify files are created
        assert os.path.exists(os.path.join(temp_dir, 'random_forest_model.pkl'))
        assert os.path.exists(os.path.join(temp_dir, 'ensemble_metadata.pkl'))
        
        # Load model
        new_ensemble = EnsembleModel()
        new_ensemble.load_model(temp_dir)
        
        # Verify loaded model
        assert new_ensemble.is_fitted
        assert new_ensemble.feature_names == ensemble_model.feature_names
        
        # Test predictions are consistent
        new_predictions = new_ensemble.predict_proba(X[:10])
        np.testing.assert_array_almost_equal(original_predictions, new_predictions, decimal=5)

class TestModelPredictor:
    """Test model predictor functionality"""
    
    @pytest.fixture
    def trained_predictor(self, temp_dir, sample_data):
        """Create trained predictor"""
        X, y = sample_data
        
        # Train and save model
        ensemble = EnsembleModel()
        ensemble.fit
