from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum

class TransactionStatus(str, Enum):
    """Transaction status enumeration"""
    PENDING = "pending"
    APPROVED = "approved" 
    DECLINED = "declined"
    FLAGGED = "flagged"

class MerchantCategory(str, Enum):
    """Merchant category enumeration"""
    GROCERY = "grocery"
    RESTAURANT = "restaurant"
    GAS_STATION = "gas_station"
    ONLINE = "online"
    RETAIL = "retail"
    PHARMACY = "pharmacy"
    ENTERTAINMENT = "entertainment"
    TRAVEL = "travel"
    UTILITIES = "utilities"
    GAMBLING = "gambling"
    CASH_ADVANCE = "cash_advance"
    OTHER = "other"

class CardType(str, Enum):
    """Card type enumeration"""
    CREDIT = "credit"
    DEBIT = "debit"
    PREPAID = "prepaid"

class DeviceType(str, Enum):
    """Device type enumeration"""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    POS_TERMINAL = "pos_terminal"

# Location Schema
class LocationSchema(BaseModel):
    """Geographic location schema"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    city: Optional[str] = Field(None, description="City name")
    country: Optional[str] = Field("US", description="Country code")
    
    class Config:
        schema_extra = {
            "example": {
                "lat": 37.7749,
                "lon": -122.4194,
                "city": "San Francisco",
                "country": "US"
            }
        }

# Transaction Request Schemas
class TransactionBase(BaseModel):
    """Base transaction schema"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp: str = Field(..., description="Transaction timestamp (ISO format)")
    device_id: str = Field(..., description="Device identifier")
    location: LocationSchema = Field(..., description="Transaction location")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO format.')
        return v

class TransactionRequest(TransactionBase):
    """Transaction prediction request"""
    merchant_category: Optional[MerchantCategory] = Field(MerchantCategory.OTHER, description="Merchant category")
    card_type: Optional[CardType] = Field(CardType.CREDIT, description="Card type")
    device_type: Optional[DeviceType] = Field(DeviceType.MOBILE, description="Device type")
    
    # Additional optional fields
    currency: Optional[str] = Field("USD", description="Transaction currency")
    payment_method: Optional[str] = Field("card", description="Payment method")
    channel: Optional[str] = Field("online", description="Transaction channel")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_123456789",
                "amount": 299.99,
                "merchant_id": "MERCH_001234",
                "user_id": "USER_567890",
                "timestamp": "2025-09-21T10:30:00Z",
                "device_id": "DEVICE_ABC123",
                "location": {
                    "lat": 37.7749,
                    "lon": -122.4194,
                    "city": "San Francisco",
                    "country": "US"
                },
                "merchant_category": "online",
                "card_type": "credit",
                "device_type": "mobile"
            }
        }

# Response Schemas
class FeatureExplanation(BaseModel):
    """Feature explanation schema"""
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="Feature contribution to prediction")
    
class PredictionExplanation(BaseModel):
    """Prediction explanation schema"""
    top_features: List[FeatureExplanation] = Field(..., description="Top contributing features")
    model_version: Optional[str] = Field(None, description="Model version used")
    explanation_method: Optional[str] = Field("shap", description="Explanation method")

class FraudPredictionResponse(BaseModel):
    """Fraud prediction response"""
    transaction_id: str = Field(..., description="Transaction identifier")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability score")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    risk_level: str = Field(..., description="Risk level category")
    explanation: PredictionExplanation = Field(..., description="Prediction explanation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: Optional[str] = Field("1.0", description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    @validator('risk_level', pre=True, always=True)
    def set_risk_level(cls, v, values):
        """Set risk level based on fraud probability"""
        fraud_prob = values.get('fraud_probability', 0)
        if fraud_prob >= 0.8:
            return "VERY_HIGH"
        elif fraud_prob >= 0.6:
            return "HIGH"
        elif fraud_prob >= 0.4:
            return "MEDIUM"
        elif fraud_prob >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_123456789",
                "is_fraud": False,
                "fraud_probability": 0.234,
                "risk_score": 23.4,
                "risk_level": "LOW",
                "explanation": {
                    "top_features": [
                        {"feature": "amount_deviation", "contribution": 0.15},
                        {"feature": "transaction_velocity", "contribution": 0.08},
                        {"feature": "location_risk", "contribution": 0.05}
                    ],
                    "model_version": "1.0",
                    "explanation_method": "shap"
                },
                "processing_time_ms": 87.5,
                "model_version": "1.0",
                "timestamp": "2025-09-21T10:30:01Z"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    transactions: List[TransactionRequest] = Field(..., description="List of transactions to predict")
    include_explanations: Optional[bool] = Field(False, description="Include explanations for each prediction")
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError('Batch must contain at least 1 transaction')
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[FraudPredictionResponse] = Field(..., description="List of predictions")
    batch_id: str = Field(..., description="Batch identifier")
    total_processed: int = Field(..., description="Number of transactions processed")
    total_fraud_detected: int = Field(..., description="Number of fraud cases detected")
    processing_time_ms: float = Field(..., description="Total processing time")
    timestamp: str = Field(..., description="Batch processing timestamp")

# Health Check Schemas
class ServiceStatus(BaseModel):
    """Service status schema"""
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    response_time_ms: Optional[float] = Field(None, description="Response time")
    last_check: str = Field(..., description="Last check timestamp")

class HealthResponse(BaseModel):
    """System health response"""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    redis_connected: bool = Field(..., description="Redis connection status")
    services: List[ServiceStatus] = Field([], description="Individual service statuses")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-09-21T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "models_loaded": True,
                "redis_connected": True,
                "services": [
                    {
                        "name": "redis",
                        "status": "healthy",
                        "response_time_ms": 2.3,
                        "last_check": "2025-09-21T10:29:58Z"
                    }
                ]
            }
        }

# Statistics Schemas
class SystemStats(BaseModel):
    """System statistics response"""
    total_predictions: int = Field(..., description="Total predictions made")
    fraud_detected: int = Field(..., description="Total fraud detected")
    fraud_rate: float = Field(..., description="Current fraud detection rate")
    average_latency_ms: float = Field(..., description="Average prediction latency")
    requests_per_minute: float = Field(..., description="Current requests per minute")
    model_accuracy: Optional[float] = Field(None, description="Current model accuracy")
    last_retrain: Optional[str] = Field(None, description="Last model retrain timestamp")

# Model Management Schemas
class ModelInfo(BaseModel):
    """Model information schema"""
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    algorithm: str = Field(..., description="Algorithm type")
    training_date: str = Field(..., description="Training date")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")

class EnsembleModelInfo(BaseModel):
    """Ensemble model information"""
    ensemble_type: str = Field("weighted_voting", description="Ensemble type")
    models: List[ModelInfo] = Field(..., description="Individual models in ensemble")
    weights: Dict[str, float] = Field(..., description="Model weights")
    overall_performance: Dict[str, float] = Field(..., description="Overall ensemble performance")

# Feature Schemas
class FeatureImportance(BaseModel):
    """Feature importance schema"""
    feature_name: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")
    rank: int = Field(..., description="Importance rank")

class FeatureStats(BaseModel):
    """Feature statistics schema"""
    feature_name: str = Field(..., description="Feature name")
    mean: Optional[float] = Field(None, description="Mean value")
    std: Optional[float] = Field(None, description="Standard deviation")
    min: Optional[float] = Field(None, description="Minimum value")
    max: Optional[float] = Field(None, description="Maximum value")
    missing_count: int = Field(..., description="Number of missing values")
    data_type: str = Field(..., description="Data type")

# Alert Schemas
class AlertLevel(str, Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class Alert(BaseModel):
    """System alert schema"""
    alert_id: str = Field(..., description="Alert identifier")
    level: AlertLevel = Field(..., description="Alert level")
    message: str = Field(..., description="Alert message")
    timestamp: str = Field(..., description="Alert timestamp")
    service: Optional[str] = Field(None, description="Related service")
    metric_value: Optional[float] = Field(None, description="Metric value that triggered alert")
    threshold: Optional[float] = Field(None, description="Alert threshold")

# Monitoring Schemas
class MetricValue(BaseModel):
    """Time series metric value"""
    timestamp: str = Field(..., description="Metric timestamp")
    value: float = Field(..., description="Metric value")

class TimeSeriesMetric(BaseModel):
    """Time series metric data"""
    metric_name: str = Field(..., description="Metric name")
    values: List[MetricValue] = Field(..., description="Metric values over time")
    unit: Optional[str] = Field(None, description="Metric unit")

class DashboardData(BaseModel):
    """Dashboard data response"""
    fraud_rate_24h: float = Field(..., description="Fraud rate last 24 hours")
    total_transactions_24h: int = Field(..., description="Total transactions last 24 hours")
    average_latency_ms: float = Field(..., description="Average response latency")
    system_alerts: List[Alert] = Field([], description="Recent system alerts")
    top_risk_features: List[FeatureImportance] = Field(..., description="Top risk features")
    performance_metrics: Dict[str, float] = Field(..., description="Current performance metrics")

# Error Schemas
class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field name with error")
    message: str = Field(..., description="Error message")
    invalid_value: Any = Field(None, description="Invalid value provided")

class ErrorResponse(BaseModel):
    """API error response"""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    validation_errors: Optional[List[ValidationError]] = Field(None, description="Validation errors")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")

# Configuration Schemas
class ModelConfig(BaseModel):
    """Model configuration schema"""
    fraud_threshold: float = Field(0.5, ge=0, le=1, description="Fraud probability threshold")
    batch_size: int = Field(100, ge=1, le=1000, description="Maximum batch size")
    enable_explanations: bool = Field(True, description="Enable prediction explanations")
    max_explanation_features: int = Field(10, ge=1, le=50, description="Max features in explanation")

class APIConfig(BaseModel):
    """API configuration schema"""
    rate_limit: int = Field(1000, description="Requests per minute limit")
    timeout_seconds: int = Field(30, description="Request timeout")
    enable_caching: bool = Field(True, description="Enable response caching")
    log_predictions: bool = Field(True, description="Log all predictions")

# User Management Schemas (for future expansion)
class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_CLIENT = "api_client"

class User(BaseModel):
    """User schema"""
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    role: UserRole = Field(..., description="User role")
    created_at: str = Field(..., description="Account creation timestamp")
    last_login: Optional[str] = Field(None, description="Last login timestamp")
    is_active: bool = Field(True, description="Account active status")

class APIKey(BaseModel):
    """API key schema"""
    key_id: str = Field(..., description="API key identifier")
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(..., description="API key permissions")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    last_used: Optional[str] = Field(None, description="Last usage timestamp")

# Audit Schemas
class AuditLog(BaseModel):
    """Audit log entry"""
    log_id: str = Field(..., description="Log entry identifier")
    timestamp: str = Field(..., description="Action timestamp")
    user_id: Optional[str] = Field(None, description="User who performed action")
    action: str = Field(..., description="Action performed")
    resource: str = Field(..., description="Resource affected")
    details: Dict[str, Any] = Field({}, description="Additional details")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    success: bool = Field(..., description="Whether action was successful")

# Feedback Schemas
class PredictionFeedback(BaseModel):
    """Prediction feedback schema"""
    transaction_id: str = Field(..., description="Transaction identifier")
    predicted_fraud: bool = Field(..., description="Original prediction")
    actual_fraud: bool = Field(..., description="Actual fraud status")
    feedback_timestamp: str = Field(..., description="Feedback timestamp")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Feedback confidence")
    notes: Optional[str] = Field(None, description="Additional notes")

# Export/Import Schemas
class DataExportRequest(BaseModel):
    """Data export request"""
    start_date: str = Field(..., description="Start date for export")
    end_date: str = Field(..., description="End date for export")
    include_predictions: bool = Field(True, description="Include prediction data")
    include_features: bool = Field(False, description="Include feature data")
    format: str = Field("csv", description="Export format (csv, json, parquet)")
    
class DataExportResponse(BaseModel):
    """Data export response"""
    export_id: str = Field(..., description="Export job identifier")
    status: str = Field(..., description="Export status")
    download_url: Optional[str] = Field(None, description="Download URL when ready")
    created_at: str = Field(..., description="Export creation timestamp")
    expires_at: str = Field(..., description="Download expiration timestamp")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")

# Common response wrapper
class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = Field(True, description="Request success status")
    data: Any = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {"result": "example"},
                "message": "Request processed successfully",
                "timestamp": "2025-09-21T10:30:00Z"
            }
        }
