"""
Streaming data processing components.

This module handles real-time transaction processing using
Kafka and Spark Streaming for fraud detection pipeline.
"""

from .kafka_consumer import FraudDetectionConsumer
from .transaction_producer import TransactionProducer, TransactionSimulator
from .spark_processor import SparkFraudProcessor

__all__ = [
    'FraudDetectionConsumer',
    'TransactionProducer', 
    'TransactionSimulator',
    'SparkFraudProcessor'
]
