import redis
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Any, Optional
import asyncio
import aioredis
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature computation"""
    name: str
    ttl: int  # Time to live in seconds
    computation_window: str  # '1h', '24h', '7d', etc.
    aggregation_type: str  # 'count', 'sum', 'avg', 'std', 'min', 'max'

class RealTimeFeatureStore:
    """Real-time feature store using Redis for fraud detection"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True,
            health_check_interval=30
        )
        
        # Feature configurations
        self.feature_configs = self._setup_feature_configs()
        
        # Time windows in seconds
        self.time_windows = {
            '1h': 3600,
            '24h': 86400,
            '7d': 604800,
            '30d': 2592000
        }
        
        logger.info("RealTimeFeatureStore initialized")
    
    def _setup_feature_configs(self) -> List[FeatureConfig]:
        """Setup feature computation configurations"""
        
        configs = [
            # Transaction count features
            FeatureConfig('user_txn_count_1h', 3600, '1h', 'count'),
            FeatureConfig('user_txn_count_24h', 86400, '24h', 'count'),
            FeatureConfig('user_txn_count_7d', 604800, '7d', 'count'),
            
            # Amount-based features
            FeatureConfig('user_total_amount_1h', 3600, '1h', 'sum'),
            FeatureConfig('user_total_amount_24h', 86400, '24h', 'sum'),
            FeatureConfig('user_avg_amount_7d', 604800, '7d', 'avg'),
            FeatureConfig('user_std_amount_7d', 604800, '7d', 'std'),
            FeatureConfig('user_max_amount_7d', 604800, '7d', 'max'),
            FeatureConfig('user_min_amount_7d', 604800, '7d', 'min'),
            
            # Device features
            FeatureConfig('device_txn_count_24h', 86400, '24h', 'count'),
            FeatureConfig('device_unique_users_24h', 86400, '24h', 'count_unique'),
            
            # Merchant features
            FeatureConfig('merchant_txn_count_1h', 3600, '1h', 'count'),
            FeatureConfig('merchant_avg_amount_24h', 86400, '24h', 'avg'),
            
            # Location features
            FeatureConfig('user_unique_locations_24h', 86400, '24h', 'count_unique'),
            
            # Category features
            FeatureConfig('user_category_count_24h', 86400, '24h', 'count'),
        ]
        
        return configs
    
    def update_transaction_features(self, transaction: Dict[str, Any]):
        """Update real-time features for a new transaction"""
        
        user_id = transaction.get('user_id')
        device_id = transaction.get('device_id')
        merchant_id = transaction.get('merchant_id')
        amount = float(transaction.get('amount', 0))
        category = transaction.get('merchant_category', 'other')
        location = transaction.get('location', {})
        timestamp = datetime.fromisoformat(transaction.get('timestamp', datetime.utcnow().isoformat()).replace('Z', '+00:00'))
        
        current_time = int(timestamp.timestamp())
        
        # Use pipeline for atomic operations
        pipeline = self.redis_client.pipeline()
        
        # Update user features
        self._update_user_features(pipeline, user_id, amount, category, location, current_time)
        
        # Update device features
        self._update_device_features(pipeline, device_id, user_id, current_time)
        
        # Update merchant features
        self._update_merchant_features(pipeline, merchant_id, amount, current_time)
        
        # Update location features
        self._update_location_features(pipeline, user_id, location, current_time)
        
        # Execute all operations
        pipeline.execute()
        
        logger.debug(f"Updated features for transaction {transaction.get('transaction_id')}")
    
    def _update_user_features(self, pipeline, user_id: str, amount: float, category: str, location: Dict, timestamp: int):
        """Update user-specific features"""
        
        base_key = f"user:{user_id}"
        
        # Transaction count features with time-based keys
        for window in ['1h', '24h', '7d']:
            window_seconds = self.time_windows[window]
            count_key = f"{base_key}:txn_count:{window}"
            
            # Add current transaction to sorted set with timestamp as score
            pipeline.zadd(count_key, {timestamp: timestamp})
            
            # Remove old entries outside the window
            cutoff_time = timestamp - window_seconds
            pipeline.zremrangebyscore(count_key, 0, cutoff_time)
            
            # Set expiry
            pipeline.expire(count_key, window_seconds * 2)
        
        # Amount-based features
        for window in ['1h', '24h', '7d']:
            window_seconds = self.time_windows[window]
            amount_key = f"{base_key}:amounts:{window}"
            
            # Store amounts with timestamps
            pipeline.zadd(amount_key, {f"{timestamp}:{amount}": timestamp})
            
            # Clean old entries
            cutoff_time = timestamp - window_seconds
            pipeline.zremrangebyscore(amount_key, 0, cutoff_time)
            pipeline.expire(amount_key, window_seconds * 2)
        
        # Running statistics (simplified moving averages)
        stats_key = f"{base_key}:stats"
        pipeline.hset(stats_key, {
            'last_amount': amount,
            'last_timestamp': timestamp,
            'last_category': category
        })
        pipeline.expire(stats_key, self.time_windows['30d'])
        
        # Category diversity
        category_key = f"{base_key}:categories:24h"
        pipeline.zadd(category_key, {f"{category}:{timestamp}": timestamp})
        cutoff_time = timestamp - self.time_windows['24h']
        pipeline.zremrangebyscore(category_key, 0, cutoff_time)
        pipeline.expire(category_key, self.time_windows['24h'] * 2)
    
    def _update_device_features(self, pipeline, device_id: str, user_id: str, timestamp: int):
        """Update device-specific features"""
        
        base_key = f"device:{device_id}"
        
        # Transaction count
        txn_key = f"{base_key}:txns:24h"
        pipeline.zadd(txn_key, {f"{user_id}:{timestamp}": timestamp})
        
        cutoff_time = timestamp - self.time_windows['24h']
        pipeline.zremrangebyscore(txn_key, 0, cutoff_time)
        pipeline.expire(txn_key, self.time_windows['24h'] * 2)
        
        # Unique users
        users_key = f"{base_key}:users:24h"
        pipeline.zadd(users_key, {user_id: timestamp})
        pipeline.zremrangebyscore(users_key, 0, cutoff_time)
        pipeline.expire(users_key, self.time_windows['24h'] * 2)
    
    def _update_merchant_features(self, pipeline, merchant_id: str, amount: float, timestamp: int):
        """Update merchant-specific features"""
        
        base_key = f"merchant:{merchant_id}"
        
        # Transaction count and amounts
        for window in ['1h', '24h']:
            window_seconds = self.time_windows[window]
            
            # Transaction count
            count_key = f"{base_key}:txns:{window}"
            pipeline.zadd(count_key, {timestamp: timestamp})
            
            # Amounts for average calculation
            amount_key = f"{base_key}:amounts:{window}"
            pipeline.zadd(amount_key, {f"{timestamp}:{amount}": timestamp})
            
            # Clean old entries
            cutoff_time = timestamp - window_seconds
            pipeline.zremrangebyscore(count_key, 0, cutoff_time)
            pipeline.zremrangebyscore(amount_key, 0, cutoff_time)
            
            pipeline.expire(count_key, window_seconds * 2)
            pipeline.expire(amount_key, window_seconds * 2)
    
    def _update_location_features(self, pipeline, user_id: str, location: Dict, timestamp: int):
        """Update location-based features"""
        
        if not location or 'lat' not in location or 'lon' not in location:
            return
        
        lat = location['lat']
        lon = location['lon']
        
        # Create location hash for uniqueness detection
        location_hash = hashlib.md5(f"{lat:.3f},{lon:.3f}".encode()).hexdigest()[:8]
        
        base_key = f"user:{user_id}"
        locations_key = f"{base_key}:locations:24h"
        
        pipeline.zadd(locations_key, {location_hash: timestamp})
        
        cutoff_time = timestamp - self.time_windows['24h']
        pipeline.zremrangebyscore(locations_key, 0, cutoff_time)
        pipeline.expire(locations_key, self.time_windows['24h'] * 2)
    
    def get_user_features(self, user_id: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Get all features for a user at a specific timestamp"""
        
        if timestamp is None:
            timestamp = int(time.time())
        
        features = {}
        base_key = f"user:{user_id}"
        
        # Transaction count features
        for window in ['1h', '24h', '7d']:
            count_key = f"{base_key}:txn_count:{window}"
            window_seconds = self.time_windows[window]
            cutoff_time = timestamp - window_seconds
            
            count = self.redis_client.zcount(count_key, cutoff_time, timestamp)
            features[f'txn_count_{window}'] = count
        
        # Amount-based features
        for window in ['1h', '24h', '7d']:
            amount_key = f"{base_key}:amounts:{window}"
            window_seconds = self.time_windows[window]
            cutoff_time = timestamp - window_seconds
            
            # Get amounts within window
            amounts_raw = self.redis_client.zrangebyscore(amount_key, cutoff_time, timestamp)
            amounts = []
            
            for item in amounts_raw:
                try:
                    _, amount_str = item.split(':', 1)
                    amounts.append(float(amount_str))
                except:
                    continue
            
            if amounts:
                features[f'total_amount_{window}'] = sum(amounts)
                features[f'avg_amount_{window}'] = np.mean(amounts)
                features[f'std_amount_{window}'] = np.std(amounts)
                features[f'max_amount_{window}'] = max(amounts)
                features[f'min_amount_{window}'] = min(amounts)
            else:
                features[f'total_amount_{window}'] = 0
                features[f'avg_amount_{window}'] = 0
                features[f'std_amount_{window}'] = 0
                features[f'max_amount_{window}'] = 0
                features[f'min_amount_{window}'] = 0
        
        # Category diversity
        category_key = f"{base_key}:categories:24h"
        cutoff_time = timestamp - self.time_windows['24h']
        categories = self.redis_client.zrangebyscore(category_key, cutoff_time, timestamp)
        unique_categories = len(set([cat.split(':')[0] for cat in categories]))
        features['unique_categories_24h'] = unique_categories
        
        # Location diversity
        locations_key = f"{base_key}:locations:24h"
        unique_locations = self.redis_client.zcount(locations_key, cutoff_time, timestamp)
        features['unique_locations_24h'] = unique_locations
        
        # Last transaction stats
        stats_key = f"{base_key}:stats"
        stats = self.redis_client.hgetall(stats_key)
        
        features['last_amount'] = float(stats.get('last_amount', 0))
        features['last_timestamp'] = int(stats.get('last_timestamp', timestamp))
        features['last_category'] = stats.get('last_category', 'other')
        
        # Time since last transaction
        if features['last_timestamp']:
            features['time_since_last_txn'] = (timestamp - features['last_timestamp']) / 3600  # hours
        else:
            features['time_since_last_txn'] = 24  # default
        
        return features
    
    def get_merchant_features(self, merchant_id: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Get merchant-specific features"""
        
        if timestamp is None:
            timestamp = int(time.time())
        
        features = {}
        base_key = f"merchant:{merchant_id}"
        
        for window in ['1h', '24h']:
            window_seconds = self.time_windows[window]
            cutoff_time = timestamp - window_seconds
            
            # Transaction count
            count_key = f"{base_key}:txns:{window}"
            txn_count = self.redis_client.zcount(count_key, cutoff_time, timestamp)
            features[f'merchant_txn_count_{window}'] = txn_count
            
            # Average amount
            amount_key = f"{base_key}:amounts:{window}"
            amounts_raw = self.redis_client.zrangebyscore(amount_key, cutoff_time, timestamp)
            amounts = []
            
            for item in amounts_raw:
                try:
                    _, amount_str = item.split(':', 1)
                    amounts.append(float(amount_str))
                except:
                    continue
            
            if amounts:
                features[f'merchant_avg_amount_{window}'] = np.mean(amounts)
                features[f'merchant_total_amount_{window}'] = sum(amounts)
            else:
                features[f'merchant_avg_amount_{window}'] = 0
                features[f'merchant_total_amount_{window}'] = 0
        
        # Risk score based on transaction velocity
        velocity_1h = features['merchant_txn_count_1h']
        velocity_24h = features['merchant_txn_count_24h']
        
        # Simple risk heuristic
        if velocity_1h > 100 or velocity_24h > 1000:
            features['merchant_risk_score'] = 0.8
        elif velocity_1h > 50 or velocity_24h > 500:
            features['merchant_risk_score'] = 0.6
        else:
            features['merchant_risk_score'] = 0.3
        
        return features
    
    def get_all_features(self, user_id: str, device_id: str, merchant_id: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Get all features for a transaction"""
        
        all_features = {}
        
        # Get user features
        user_features = self.get_user_features(user_id, timestamp)
        all_features.update(user_features)
        
        # Get device features
        device_features = self.get_device_features(device_id, timestamp)
        all_features.update(device_features)
        
        # Get merchant features
        merchant_features = self.get_merchant_features(merchant_id, timestamp)
        all_features.update(merchant_features)
        
        return all_features
    
    def cleanup_expired_keys(self):
        """Clean up expired keys (maintenance task)"""
        
        patterns_to_clean = [
            'user:*:txn_count:*',
            'user:*:amounts:*',
            'device:*:txns:*',
            'device:*:users:*',
            'merchant:*:txns:*',
            'merchant:*:amounts:*'
        ]
        
        cleaned_count = 0
        for pattern in patterns_to_clean:
            keys = self.redis_client.keys(pattern)
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiry set
                    # Set default expiry based on key pattern
                    if ':1h' in key:
                        self.redis_client.expire(key, 7200)  # 2 hours
                    elif ':24h' in key:
                        self.redis_client.expire(key, 172800)  # 2 days
                    elif ':7d' in key:
                        self.redis_client.expire(key, 1209600)  # 14 days
                    cleaned_count += 1
        
        logger.info(f"Set expiry for {cleaned_count} keys")
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about the feature store"""
        
        info = self.redis_client.info()
        
        # Count keys by pattern
        key_counts = {}
        patterns = ['user:*', 'device:*', 'merchant:*']
        
        for pattern in patterns:
            keys = self.redis_client.keys(pattern)
            key_counts[pattern] = len(keys)
        
        stats = {
            'redis_info': {
                'used_memory': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            },
            'key_counts': key_counts,
            'feature_configs': len(self.feature_configs)
        }
        
        return stats

class FeatureStoreAPI:
    """API wrapper for the feature store"""
    
    def __init__(self, feature_store: RealTimeFeatureStore):
        self.feature_store = feature_store
        
    async def process_transaction_async(self, transaction: Dict[str, Any]):
        """Asynchronously process transaction for features"""
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            self.feature_store.update_transaction_features, 
            transaction
        )
    
    async def get_features_async(self, user_id: str, device_id: str, merchant_id: str) -> Dict[str, Any]:
        """Asynchronously get features"""
        
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            None,
            self.feature_store.get_all_features,
            user_id,
            device_id,
            merchant_id
        )
        
        return features

def create_sample_transactions(n_samples: int = 1000) -> List[Dict[str, Any]]:
    """Create sample transactions for testing"""
    
    np.random.seed(42)
    
    transactions = []
    base_time = datetime.now()
    
    for i in range(n_samples):
        transaction = {
            'transaction_id': f'TXN_{i:06d}',
            'user_id': f'USER_{np.random.randint(1, 100):03d}',
            'device_id': f'DEV_{np.random.randint(1, 50):03d}',
            'merchant_id': f'MERCH_{np.random.randint(1, 200):03d}',
            'amount': np.random.lognormal(3, 1),
            'merchant_category': np.random.choice(['grocery', 'restaurant', 'online', 'gas_station']),
            'location': {
                'lat': np.random.uniform(37.0, 38.0),
                'lon': np.random.uniform(-122.5, -121.5)
            },
            'timestamp': (base_time - timedelta(minutes=np.random.randint(0, 1440))).isoformat()
        }
        transactions.append(transaction)
    
    return transactions

def main():
    """Main function for testing feature store"""
    
    # Initialize feature store
    feature_store = RealTimeFeatureStore()
    
    # Create sample transactions
    logger.info("Creating sample transactions...")
    transactions = create_sample_transactions(1000)
    
    # Process transactions
    logger.info("Processing transactions...")
    start_time = time.time()
    
    for i, transaction in enumerate(transactions):
        feature_store.update_transaction_features(transaction)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1} transactions")
    
    processing_time = time.time() - start_time
    logger.info(f"Processed {len(transactions)} transactions in {processing_time:.2f} seconds")
    logger.info(f"Rate: {len(transactions)/processing_time:.1f} transactions/second")
    
    # Test feature retrieval
    logger.info("Testing feature retrieval...")
    sample_transaction = transactions[0]
    
    features = feature_store.get_all_features(
        sample_transaction['user_id'],
        sample_transaction['device_id'],
        sample_transaction['merchant_id']
    )
    
    logger.info("Sample features:")
    for key, value in list(features.items())[:10]:
        logger.info(f"  {key}: {value}")
    
    # Show statistics
    stats = feature_store.get_feature_statistics()
    logger.info("Feature store statistics:")
    logger.info(f"  Memory used: {stats['redis_info']['used_memory']}")
    logger.info(f"  Key counts: {stats['key_counts']}")
    
    # Cleanup test
    logger.info("Running cleanup...")
    feature_store.cleanup_expired_keys()
    
    logger.info("Feature store test completed!")

if __name__ == "__main__":
    main()
    
    def get_device_features(self, device_id: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Get device-specific features"""
        
        if timestamp is None:
            timestamp = int(time.time())
        
        features = {}
        base_key = f"device:{device_id}"
        cutoff_time = timestamp - self.time_windows['24h']
        
        # Transaction count
        txn_key = f"{base_key}:txns:24h"
        txn_count = self.redis_client.zcount(txn_key, cutoff_time, timestamp)
        features['device_txn_count_24h'] = txn_count
        
        # Unique users
        users_key = f"{base_key}:users:24h"
        unique_users = self.redis_client.zcount(users_key, cutoff_time, timestamp)
        features['device_unique_users_24h'] = unique_users
        
        # Risk score (simple heuristic)
        if txn_count > 0:
            features['device_risk_score'] = min(unique_users / max(txn_count, 1), 1.0)
        else:
            features['device_risk_score'] = 0.5
