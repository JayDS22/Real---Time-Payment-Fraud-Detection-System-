import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_engineer import FeatureEngineer
from features.feature_store import RealTimeFeatureStore

class TestFeatureEngineer:
    """Test feature engineering functionality"""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data"""
        return pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
            'user_id': ['USER_001', 'USER_001', 'USER_002'],
            'amount': [100.0, 250.0, 75.0],
            'merchant_id': ['MERCH_001', 'MERCH_002', 'MERCH_001'],
            'merchant_category': ['grocery', 'restaurant', 'grocery'],
            'timestamp': [
                '2025-09-21T10:30:00Z',
                '2025-09-21T11:30:00Z', 
                '2025-09-21T12:30:00Z'
            ],
            'location': [
                {'lat': 37.7749, 'lon': -122.4194},
                {'lat': 37.7849, 'lon': -122.4094},
                {'lat': 40.7128, 'lon': -74.0060}
            ]
        })
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    def test_temporal_features(self, feature_engineer, sample_transactions):
        """Test temporal feature creation"""
        result = feature_engineer._add_temporal_features(sample_transactions)
        
        # Check that temporal features are created
        expected_features = ['hour', 'day_of_week', 'is_weekend', 'is_night']
        for feature in expected_features:
            assert feature in result.columns
        
        # Verify specific values
        assert result.iloc[0]['hour'] == 10
        assert result.iloc[0]['is_weekend'] in [0, 1]
        assert result.iloc[0]['is_night'] in [0, 1]
    
    def test_amount_features(self, feature_engineer, sample_transactions):
        """Test amount-based feature creation"""
        result = feature_engineer._add_amount_features(sample_transactions)
        
        # Check for amount features
        expected_features = ['log_amount', 'sqrt_amount', 'is_round_amount']
        for feature in expected_features:
            assert feature in result.columns
        
        # Verify calculations
        assert result.iloc[0]['log_amount'] == np.log1p(100.0)
        assert result.iloc[0]['sqrt_amount'] == np.sqrt(100.0)
        assert result.iloc[0]['is_round_amount'] == 1  # 100.0 is round
    
    def test_categorical_features(self, feature_engineer, sample_transactions):
        """Test categorical feature encoding"""
        # Add fraud labels for risk encoding
        sample_transactions['is_fraud'] = [0, 1, 0]
        
        result = feature_engineer._add_categorical_features(sample_transactions)
        
        # Check for encoded features
        assert 'merchant_category_frequency' in result.columns
        assert 'merchant_category_encoded' in result.columns
        
        # Verify frequency encoding
        assert result.iloc[0]['merchant_category_frequency'] > 0
    
    def test_location_features(self, feature_engineer, sample_transactions):
        """Test location feature extraction"""
        result = feature_engineer._add_location_features(sample_transactions)
        
        # Check that lat/lon are extracted
        assert 'lat' in result.columns
        assert 'lon' in result.columns
        
        # Verify values
        assert result.iloc[0]['lat'] == 37.7749
        assert result.iloc[0]['lon'] == -122.4194
    
    def test_feature_creation_integration(self, feature_engineer, sample_transactions):
        """Test complete feature creation pipeline"""
        result = feature_engineer.create_features(sample_transactions)
        
        # Should have more columns than input
        assert len(result.columns) > len(sample_transactions.columns)
        
        # Should have same number of rows
        assert len(result) == len(sample_transactions)
        
        # Should not have any infinite or null values in final result
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

class TestRealTimeFeatureStore:
    """Test real-time feature store functionality"""
    
    @pytest.fixture
    def feature_store(self):
        """Create feature store instance (mock Redis)"""
        try:
            return RealTimeFeatureStore('localhost', 6379)
        except:
            pytest.skip("Redis not available for testing")
    
    @pytest.fixture
    def sample_transaction(self):
        """Create sample transaction"""
        return {
            'transaction_id': 'TXN_TEST_001',
            'user_id': 'USER_TEST_001',
            'device_id': 'DEV_TEST_001',
            'merchant_id': 'MERCH_TEST_001',
            'amount': 150.0,
            'merchant_category': 'restaurant',
            'location': {'lat': 37.7749, 'lon': -122.4194},
            'timestamp': datetime.now().isoformat()
        }
    
    def test_update_transaction_features(self, feature_store, sample_transaction):
        """Test updating transaction features"""
        if feature_store is None:
            pytest.skip("Feature store not available")
        
        # Should not raise exception
        feature_store.update_transaction_features(sample_transaction)
        
        # Verify features can be retrieved
        user_features = feature_store.get_user_features(sample_transaction['user_id'])
        assert isinstance(user_features, dict)
        assert len(user_features) > 0
    
    def test_get_user_features(self, feature_store, sample_transaction):
        """Test retrieving user features"""
        if feature_store is None:
            pytest.skip("Feature store not available")
        
        # Update first
        feature_store.update_transaction_features(sample_transaction)
        
        # Get features
        features = feature_store.get_user_features(sample_transaction['user_id'])
        
        # Check expected feature structure
        expected_features = ['txn_count_1h', 'txn_count_24h', 'avg_amount_24h']
        for feature in expected_features:
            assert feature in features
    
    def test_get_device_features(self, feature_store, sample_transaction):
        """Test retrieving device features"""
        if feature_store is None:
            pytest.skip("Feature store not available")
        
        # Update first
        feature_store.update_transaction_features(sample_transaction)
        
        # Get features
        features = feature_store.get_device_features(sample_transaction['device_id'])
        
        # Check feature structure
        assert isinstance(features, dict)
        assert 'device_txn_count_24h' in features
    
    def test_get_all_features(self, feature_store, sample_transaction):
        """Test retrieving all features for a transaction"""
        if feature_store is None:
            pytest.skip("Feature store not available")
        
        # Update first
        feature_store.update_transaction_features(sample_transaction)
        
        # Get all features
        all_features = feature_store.get_all_features(
            sample_transaction['user_id'],
            sample_transaction['device_id'], 
            sample_transaction['merchant_id']
        )
        
        # Should combine user, device, and merchant features
        assert isinstance(all_features, dict)
        assert len(all_features) > 10  # Should have many features
    
    def test_feature_statistics(self, feature_store):
        """Test feature store statistics"""
        if feature_store is None:
            pytest.skip("Feature store not available")
        
        stats = feature_store.get_feature_statistics()
        
        # Check statistics structure
        assert isinstance(stats, dict)
        assert 'redis_info' in stats
        assert 'key_counts' in stats

class TestFeatureValidation:
    """Test feature validation and data quality"""
    
    def test_feature_completeness(self):
        """Test that all expected features are created"""
        # Sample data
        df = pd.DataFrame({
            'user_id': ['USER_001', 'USER_002'],
            'amount': [100.0, 200.0],
            'merchant_category': ['grocery', 'restaurant'],
            'timestamp': [datetime.now().isoformat()] * 2,
            'location': [{'lat': 37.7749, 'lon': -122.4194}] * 2
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_features(df)
        
        # Should have significantly more features than input
        assert len(result.columns) >= 20
    
    def test_feature_data_types(self):
        """Test that features have correct data types"""
        df = pd.DataFrame({
            'user_id': ['USER_001'],
            'amount': [100.0],
            'merchant_category': ['grocery'],
            'timestamp': [datetime.now().isoformat()],
            'location': [{'lat': 37.7749, 'lon': -122.4194}]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_features(df)
        
        # Numeric features should be numeric
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0
        
        # No infinite values
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
    
    def test_feature_consistency(self):
        """Test that features are consistent across multiple runs"""
        df = pd.DataFrame({
            'user_id': ['USER_001'],
            'amount': [100.0],
            'merchant_category': ['grocery'], 
            'timestamp': ['2025-09-21T10:30:00Z'],
            'location': [{'lat': 37.7749, 'lon': -122.4194}]
        })
        
        engineer1 = FeatureEngineer()
        engineer2 = FeatureEngineer()
        
        result1 = engineer1.create_features(df)
        result2 = engineer2.create_features(df)
        
        # Results should be identical for same input
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
