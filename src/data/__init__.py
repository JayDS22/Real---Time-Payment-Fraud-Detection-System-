"""
Data generation and simulation modules for fraud detection system.

This module contains components for generating realistic synthetic
transaction data and simulating various fraud scenarios.
"""

from .data_generator import FraudDataGenerator
from .transaction_simulator import TransactionSimulator

__all__ = ['FraudDataGenerator', 'TransactionSimulator']
