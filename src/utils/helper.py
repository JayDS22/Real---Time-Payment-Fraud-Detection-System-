import os
import yaml
import json
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self, config_dir: str = 'config'):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files"""
        if not self.config_dir.exists():
            return
        
        for config_file in self.config_dir.glob('*.yaml'):
            config_name = config_file.stem
            self.configs[config_name] = self.load_config(config_file)
    
    def load_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a single configuration file"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading config {file_path}: {e}")
            return {}
    
    def get_config(self, config_name: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration by name"""
        return self.configs.get(config_name, default or {})
    
    def get_nested_config(self, config_name: str, *keys, default=None):
        """Get nested configuration value"""
        config = self.get_config(config_name)
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        
        return config

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_transaction(transaction: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate transaction data"""
        errors = []
        
        # Required fields
        required_fields = ['transaction_id', 'user_id', 'amount', 'merchant_id', 'timestamp']
        for field in required_fields:
            if field not in transaction or transaction[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Amount validation
        if 'amount' in transaction:
            try:
                amount = float(transaction['amount'])
                if amount <= 0:
                    errors.append("Amount must be positive")
                if amount > 1000000:  # 1M limit
                    errors.append("Amount exceeds maximum limit")
            except (ValueError, TypeError):
                errors.append("Amount must be a valid number")
        
        # Timestamp validation
        if 'timestamp' in transaction:
            try:
                datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                errors.append("Invalid timestamp format")
        
        # Location validation
        if 'location' in transaction:
            location = transaction['location']
            if isinstance(location, dict):
                lat = location.get('lat')
                lon = location.get('lon')
                
                if lat is not None:
                    try:
                        lat_val = float(lat)
                        if not (-90 <= lat_val <= 90):
                            errors.append("Latitude must be between -90 and 90")
                    except (ValueError, TypeError):
                        errors.append("Invalid latitude format")
                
                if lon is not None:
                    try:
                        lon_val = float(lon)
                        if not (-180 <= lon_val <= 180):
                            errors.append("Longitude must be between -180 and 180")
                    except (ValueError, TypeError):
                        errors.append("Invalid longitude format")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> tuple[bool, List[str]]:
        """Validate DataFrame structure and data quality"""
        errors = []
        
        if df is None or df.empty:
            errors.append("DataFrame is empty or None")
            return False, errors
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {list(missing_cols)}")
        
        # Check for all null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            errors.append(f"Columns with all null values: {null_cols}")
        
        # Check data types
        for col in df.select_dtypes(include=['object']).columns:
            if col in ['amount'] and col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except:
                    errors.append(f"Column {col} contains non-numeric values")
        
        return len(errors) == 0, errors

class PerformanceTimer:
    """Performance timing utility"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.get_duration_ms()
        logging.debug(f"{self.name} completed in {duration:.2f}ms")
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop timing"""
        self.end_time = time.time()
        return self.get_duration_ms()
    
    def get_duration_ms(self) -> float:
        """Get duration in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def hash_features(data: Union[str, Dict[str, Any]], algorithm: str = 'sha256') -> str:
    """Create hash of features for anonymization"""
    
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    hasher = hashlib.new(algorithm)
    hasher.update(data_str.encode('utf-8'))
    return hasher.hexdigest()

def normalize_amount(amount: float, method: str = 'log') -> float:
    """Normalize transaction amount"""
    
    if method == 'log':
        return np.log1p(max(0, amount))
    elif method == 'sqrt':
        return np.sqrt(max(0, amount))
    elif method == 'minmax':
        # Assuming typical range 0-10000
        return min(1.0, max(0.0, amount / 10000.0))
    else:
        return amount

def calculate_business_hours(timestamp: datetime, timezone: str = 'UTC') -> Dict[str, Any]:
    """Calculate business hours features"""
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    
    return {
        'is_business_hours': 9 <= hour <= 17 and day_of_week < 5,
        'is_weekend': day_of_week >= 5,
        'is_night': hour < 6 or hour > 22,
        'is_early_morning': 2 <= hour <= 6,
        'hour_category': (
            'night' if hour < 6 or hour > 22 else
            'morning' if 6 <= hour < 12 else
            'afternoon' if 12 <= hour < 18 else
            'evening'
        )
    }

def get_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    r = 6371
    
    return r * c

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format amount as currency string"""
    
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'EUR':
        return f"€{amount:,.2f}"
    elif currency == 'GBP':
        return f"£{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def create_time_windows(df: pd.DataFrame, 
                       timestamp_col: str = 'timestamp',
                       windows: List[str] = ['1H', '1D', '7D']) -> pd.DataFrame:
    """Create time-based windows for feature engineering"""
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    
    for window in windows:
        window_col = f'{timestamp_col}_{window}'
        df[window_col] = df[timestamp_col].dt.floor(window)
    
    return df

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator"""
    return numerator / denominator if denominator != 0 else default

def sanitize_string(text: str, max_length: int = 100) -> str:
    """Sanitize string input"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove potentially harmful characters
    sanitized = ''.join(char for char in text if char.isalnum() or char in '._-')
    
    # Truncate if too long
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized

def retry_operation(operation, max_retries: int = 3, delay: float = 1.0):
    """Retry operation with exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = delay * (2 ** attempt)
            logging.warning(f"Operation failed (attempt {attempt + 1}/{max_retries}), "
                          f"retrying in {wait_time}s: {e}")
            time.sleep(wait_time)

class MemoryCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str, default=None):
        """Get value from cache"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            else:
                del self.cache[key]
        return default
    
    def set(self, key: str, value, ttl: Optional[int] = None):
        """Set value in cache with TTL"""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if current_time >= expiry
        ]
        
        for key in expired_keys:
            del self.cache[key]
