import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from geopy.distance import geodesic
import hashlib

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for fraud detection"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.feature_names = []
        
    def create_features(self, 
                       transactions_df: pd.DataFrame,
                       user_history: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Create comprehensive feature set for fraud detection
        
        Args:
            transactions_df: Current transactions DataFrame
            user_history: Historical transactions per user
            
        Returns:
            DataFrame with engineered features
        """
        
        logger.info(f"Engineering features for {len(transactions_df)} transactions")
        
        df = transactions_df.copy()
        
        # Basic temporal features
        df = self._add_temporal_features(df)
        
        # Amount-based features
        df = self._add_amount_features(df)
        
        # Categorical features
        df = self._add_categorical_features(df)
        
        # Location features
        df = self._add_location_features(df)
        
        # User behavioral features
        if user_history:
            df = self._add_user_behavioral_features(df, user_history)
        
        # Merchant features
        df = self._add_merchant_features(df)
        
        # Device features
        df = self._add_device_features(df)
        
        # Interaction features
        df = self._add_interaction_features(df)
        
        # Advanced features
        df = self._add_advanced_features(df)
        
        # Select final feature set
        df = self._select_features(df)
        
        logger.info(f"Created {len(df.columns)} features")
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        # Parse timestamp if string
        if 'timestamp' in df.columns:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        else:
            df['timestamp_dt'] = datetime.now()
            
        # Basic time features
        df['hour'] = df['timestamp_dt'].dt.hour
        df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
        df['day_of_month'] = df['timestamp_dt'].dt.day
        df['month'] = df['timestamp_dt'].dt.month
        df['quarter'] = df['timestamp_dt'].dt.quarter
        df['year'] = df['timestamp_dt'].dt.year
        
        # Binary time indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday'] = self._is_holiday(df['timestamp_dt']).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 2) & (df['hour'] <= 6)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 1)).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time since features (if reference date available)
        if 'account_creation_date' in df.columns:
            df['account_age_days'] = (df['timestamp_dt'] - pd.to_datetime(df['account_creation_date'])).dt.days
            df['account_age_months'] = df['account_age_days'] / 30
        
        return df
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add amount-based features"""
        
        # Log transformations
        df['log_amount'] = np.log1p(df['amount'])
        df['sqrt_amount'] = np.sqrt(df['amount'])
        
        # Amount categories
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=[0, 10, 50, 100, 500, 1000, np.inf],
            labels=['micro', 'small', 'medium', 'large', 'xlarge', 'xxlarge']
        )
        
        # Round number indicators
        df['is_round_amount'] = (df['amount'] % 1 == 0).astype(int)
        df['is_round_10'] = (df['amount'] % 10 == 0).astype(int)
        df['is_round_100'] = (df['amount'] % 100 == 0).astype(int)
        
        # Statistical features per user (if multiple transactions)
        if 'user_id' in df.columns and len(df) > 1:
            user_stats = df.groupby('user_id')['amount'].agg([
                'mean', 'std', 'min', 'max', 'count', 'median'
            ]).add_prefix('user_amount_')
            
            df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
            
            # Derived features
            df['amount_vs_user_mean'] = df['amount'] / (df['user_amount_mean'] + 1e-5)
            df['amount_std_score'] = (df['amount'] - df['user_amount_mean']) / (df['user_amount_std'] + 1e-5)
            df['amount_vs_user_max'] = df['amount'] / (df['user_amount_max'] + 1e-5)
            df['is_user_max_amount'] = (df['amount'] == df['user_amount_max']).astype(int)
        
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add and encode categorical features"""
        
        categorical_cols = ['merchant_category', 'card_type', 'device_type']
        
        for col in categorical_cols:
            if col in df.columns:
                # Frequency encoding
                freq_encoding = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_frequency'] = df[col].map(freq_encoding)
                
                # Risk encoding based on fraud rates (if is_fraud available)
                if 'is_fraud' in df.columns:
                    risk_encoding = df.groupby(col)['is_fraud'].mean().to_dict()
                    df[f'{col}_risk'] = df[col].map(risk_encoding)
                
                # Label encoding
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].fillna('unknown'))
                else:
                    # Handle new categories
                    known_categories = set(self.encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_categories else 'unknown')
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].fillna('unknown'))
        
        return df
    
    def _add_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location-based features"""
        
        if 'location' not in df.columns and ('lat' not in df.columns or 'lon' not in df.columns):
            return df
        
        # Extract lat/lon if in dict format
        if 'location' in df.columns:
            df['lat'] = df['location'].apply(lambda x: x.get('lat', 0) if isinstance(x, dict) else 0)
            df['lon'] = df['location'].apply(lambda x: x.get('lon', 0) if isinstance(x, dict) else 0)
        
        # Distance from major cities
        major_cities = [
            ('NYC', 40.7128, -74.0060),
            ('LA', 34.0522, -118.2437),
            ('Chicago', 41.8781, -87.6298),
            ('Houston', 29.7604, -95.3698)
        ]
        
        for city_name, city_lat, city_lon in major_cities:
            df[f'distance_to_{city_name}'] = df.apply(
                lambda row: self._calculate_distance(row['lat'], row['lon'], city_lat, city_lon),
                axis=1
            )
        
        # Geographic clusters
        df['lat_rounded'] = np.round(df['lat'], 1)
        df['lon_rounded'] = np.round(df['lon'], 1)
        df['location_cluster'] = df['lat_rounded'].astype(str) + '_' + df['lon_rounded'].astype(str)
        
        # User location features (if multiple transactions per user)
        if 'user_id' in df.columns and len(df) > 1:
            # Home location (most frequent location)
            user_locations = df.groupby('user_id').agg({
                'lat': ['mean', 'std', 'count'],
                'lon': ['mean', 'std', 'count']
            }).reset_index()
            
            user_locations.columns = ['user_id', 'user_lat_mean', 'user_lat_std', 'user_lat_count',
                                    'user_lon_mean', 'user_lon_std', 'user_lon_count']
            
            df = df.merge(user_locations, on='user_id', how='left')
            
            # Distance from user's typical location
            df['distance_from_home'] = df.apply(
                lambda row: self._calculate_distance(
                    row['lat'], row['lon'], 
                    row.get('user_lat_mean', row['lat']), 
                    row.get('user_lon_mean', row['lon'])
                ), axis=1
            )
            
            # Location novelty
            df['is_new_location'] = (df['distance_from_home'] > 100).astype(int)  # >100km
            df['location_risk_score'] = np.minimum(df['distance_from_home'] / 1000, 1.0)  # Normalize to 0-1
        
        return df
    
    def _add_user_behavioral_features(self, df: pd.DataFrame, user_history: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add user behavioral features based on historical data"""
        
        behavioral_features = []
        
        for _, row in df.iterrows():
            user_id = row.get('user_id')
            current_time = pd.to_datetime(row.get('timestamp', datetime.now()))
            
            if user_id in user_history:
                history = user_history[user_id]
                
                # Transaction velocity features
                history['timestamp_dt'] = pd.to_datetime(history['timestamp'])
                
                # Transactions in different time windows
                windows = [1, 6, 24, 168]  # 1h, 6h, 24h, 7d in hours
                velocity_features = {}
                
                for window_hours in windows:
                    cutoff_time = current_time - timedelta(hours=window_hours)
                    recent_txns = history[history['timestamp_dt'] >= cutoff_time]
                    
                    velocity_features[f'txn_count_{window_hours}h'] = len(recent_txns)
                    velocity_features[f'txn_amount_sum_{window_hours}h'] = recent_txns['amount'].sum()
                    velocity_features[f'txn_amount_mean_{window_hours}h'] = recent_txns['amount'].mean()
                    
                    if len(recent_txns) > 1:
                        velocity_features[f'txn_velocity_{window_hours}h'] = len(recent_txns) / window_hours
                    else:
                        velocity_features[f'txn_velocity_{window_hours}h'] = 0
                
                # Time since last transaction
                if len(history) > 0:
                    last_txn_time = history['timestamp_dt'].max()
                    velocity_features['hours_since_last_txn'] = (current_time - last_txn_time).total_seconds() / 3600
                else:
                    velocity_features['hours_since_last_txn'] = 999999  # Large number for new users
                
                # Category and merchant diversity
                recent_1d = history[history['timestamp_dt'] >= current_time - timedelta(days=1)]
                velocity_features['unique_categories_24h'] = recent_1d['merchant_category'].nunique()
                velocity_features['unique_merchants_24h'] = recent_1d['merchant_id'].nunique()
                
                # Amount patterns
                if len(history) >= 5:
                    amounts = history['amount'].values
                    velocity_features['amount_pattern_std'] = np.std(amounts)
                    velocity_features['amount_pattern_cv'] = np.std(amounts) / (np.mean(amounts) + 1e-5)
                    velocity_features['amount_trend'] = self._calculate_trend(amounts[-10:])  # Last 10 transactions
                
                # Cyclical patterns (hour of day, day of week)
                if len(history) >= 10:
                    history_hours = history['timestamp_dt'].dt.hour
                    current_hour = current_time.hour
                    velocity_features['hour_consistency'] = (history_hours == current_hour).mean()
                    
                    history_days = history['timestamp_dt'].dt.dayofweek
                    current_day = current_time.dayofweek
                    velocity_features['day_consistency'] = (history_days == current_day).mean()
                
                behavioral_features.append(velocity_features)
            else:
                # New user - default features
                default_features = {
                    'txn_count_1h': 0, 'txn_count_6h': 0, 'txn_count_24h': 0, 'txn_count_168h': 0,
                    'txn_amount_sum_1h': 0, 'txn_amount_sum_6h': 0, 'txn_amount_sum_24h': 0, 'txn_amount_sum_168h': 0,
                    'txn_amount_mean_1h': 0, 'txn_amount_mean_6h': 0, 'txn_amount_mean_24h': 0, 'txn_amount_mean_168h': 0,
                    'txn_velocity_1h': 0, 'txn_velocity_6h': 0, 'txn_velocity_24h': 0, 'txn_velocity_168h': 0,
                    'hours_since_last_txn': 999999,
                    'unique_categories_24h': 0, 'unique_merchants_24h': 0,
                    'amount_pattern_std': 0, 'amount_pattern_cv': 0, 'amount_trend': 0,
                    'hour_consistency': 0, 'day_consistency': 0
                }
                behavioral_features.append(default_features)
        
        # Add behavioral features to DataFrame
        behavioral_df = pd.DataFrame(behavioral_features)
        df = pd.concat([df, behavioral_df], axis=1)
        
        return df
    
    def _add_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add merchant-based features"""
        
        if 'merchant_id' not in df.columns:
            return df
        
        # Merchant transaction statistics
        merchant_stats = df.groupby('merchant_id').agg({
            'amount': ['count', 'mean', 'std', 'min', 'max'],
            'user_id': 'nunique'
        }).reset_index()
        
        merchant_stats.columns = ['merchant_id', 'merchant_txn_count', 'merchant_amount_mean',
                                'merchant_amount_std', 'merchant_amount_min', 'merchant_amount_max',
                                'merchant_unique_users']
        
        df = df.merge(merchant_stats, on='merchant_id', how='left')
        
        # Merchant risk indicators
        df['merchant_amount_vs_mean'] = df['amount'] / (df['merchant_amount_mean'] + 1e-5)
        df['is_merchant_amount_outlier'] = (
            np.abs(df['amount'] - df['merchant_amount_mean']) > 
            2 * (df['merchant_amount_std'] + 1e-5)
        ).astype(int)
        
        # Merchant popularity
        df['merchant_popularity'] = df['merchant_txn_count'] / df['merchant_txn_count'].max()
        
        return df
    
    def _add_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add device-based features"""
        
        if 'device_id' not in df.columns:
            return df
        
        # Device usage statistics
        device_stats = df.groupby('device_id').agg({
            'user_id': 'nunique',
            'amount': ['count', 'sum', 'mean'],
            'merchant_id': 'nunique'
        }).reset_index()
        
        device_stats.columns = ['device_id', 'device_unique_users', 'device_txn_count',
                              'device_amount_sum', 'device_amount_mean', 'device_unique_merchants']
        
        df = df.merge(device_stats, on='device_id', how='left')
        
        # Device risk indicators
        df['device_user_ratio'] = df['device_unique_users'] / (df['device_txn_count'] + 1e-5)
        df['is_shared_device'] = (df['device_unique_users'] > 1).astype(int)
        df['device_merchant_diversity'] = df['device_unique_merchants'] / (df['device_txn_count'] + 1e-5)
        
        # Device fingerprinting features (simplified)
        if 'user_agent' in df.columns:
            df['browser_type'] = df['user_agent'].str.extract(r'(Chrome|Firefox|Safari|Edge|Opera)', expand=False)
            df['is_mobile_browser'] = df['user_agent'].str.contains('Mobile', na=False).astype(int)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different variables"""
        
        # Amount × Time interactions
        if 'amount' in df.columns and 'hour' in df.columns:
            df['amount_hour_interaction'] = df['amount'] * df['hour']
            df['amount_weekend_interaction'] = df['amount'] * df['is_weekend']
            df['amount_night_interaction'] = df['amount'] * df['is_night']
        
        # Velocity × Amount interactions
        velocity_cols = [col for col in df.columns if 'velocity' in col or 'txn_count' in col]
        for vel_col in velocity_cols[:3]:  # Limit to avoid too many features
            if vel_col in df.columns and 'amount' in df.columns:
                df[f'{vel_col}_amount_interaction'] = df[vel_col] * df['amount']
        
        # Location × Time interactions
        if 'distance_from_home' in df.columns:
            df['distance_hour_interaction'] = df['distance_from_home'] * df['hour']
            df['distance_weekend_interaction'] = df['distance_from_home'] * df['is_weekend']
        
        # Category × Amount interactions
        if 'merchant_category_encoded' in df.columns:
            df['category_amount_interaction'] = df['merchant_category_encoded'] * df['log_amount']
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistical and derived features"""
        
        # Anomaly scores using isolation forest on amount patterns
        from sklearn.ensemble import IsolationForest
        
        amount_features = ['amount', 'log_amount']
        if len(df) > 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            amount_data = df[amount_features].fillna(0)
            df['amount_anomaly_score'] = iso_forest.decision_function(amount_data)
            df['is_amount_anomaly'] = (iso_forest.predict(amount_data) == -1).astype(int)
        
        # Rolling statistics (if sorted by time)
        if 'timestamp_dt' in df.columns and len(df) > 1:
            df = df.sort_values('timestamp_dt')
            
            # Rolling amount statistics
            df['amount_rolling_mean_5'] = df['amount'].rolling(window=5, min_periods=1).mean()
            df['amount_rolling_std_5'] = df['amount'].rolling(window=5, min_periods=1).std().fillna(0)
            df['amount_vs_rolling_mean'] = df['amount'] / (df['amount_rolling_mean_5'] + 1e-5)
        
        # Entropy-based features for categorical variables
        categorical_cols = ['merchant_category', 'card_type', 'device_type']
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_entropy'] = self._calculate_entropy(df[col])
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final feature set and handle missing values"""
        
        # Exclude non-feature columns
        exclude_cols = [
            'transaction_id', 'user_id', 'merchant_id', 'device_id',
            'timestamp', 'timestamp_dt', 'location', 'is_fraud'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_df = df[feature_cols].copy()
        
        # Handle infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values
        for col in feature_df.columns:
            if feature_df[col].dtype in ['float64', 'int64']:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
            else:
                feature_df[col] = feature_df[col].fillna(feature_df[col].mode().iloc[0] if len(feature_df[col].mode()) > 0 else 'unknown')
        
        self.feature_names = feature_cols
        return feature_df
    
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Simple holiday detection (extend with actual holiday calendar)"""
        # Simplified - just weekends for now
        return (dates.dt.dayofweek >= 5)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        try:
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except:
            # Fallback to Haversine formula
            R = 6371  # Earth's radius in km
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        try:
            trend = np.polyfit(x, values, 1)[0]
            return trend
        except:
            return 0
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a categorical series"""
        value_counts = series.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
        return entropy
    
    def save_feature_config(self, filepath: str):
        """Save feature engineering configuration"""
        config = {
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_stats': self.feature_stats
        }
        joblib.dump(config, filepath)
        logger.info(f"Feature configuration saved to {filepath}")
    
    def load_feature_config(self, filepath: str):
        """Load feature engineering configuration"""
        config = joblib.load(filepath)
        self.feature_names = config['feature_names']
        self.scalers = config['scalers']
        self.encoders = config['encoders']  
        self.feature_stats = config['feature_stats']
        logger.info(f"Feature configuration loaded from {filepath}")
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_names.copy() if self.feature_names else []
