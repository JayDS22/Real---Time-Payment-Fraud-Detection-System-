import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from kafka import KafkaProducer
import logging
from typing import Dict, Any, List
import uuid
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionProducer:
    """Generates and sends synthetic transaction data to Kafka"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='transactions'):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            batch_size=16384,
            linger_ms=10,
            compression_type='gzip'
        )
        
        # Sample data pools
        self.users = [f"USER_{i:06d}" for i in range(1, 10001)]
        self.devices = [f"DEV_{i:05d}" for i in range(1, 5001)]
        self.merchants = [f"MERCH_{i:05d}" for i in range(1, 2001)]
        
        self.merchant_categories = [
            'grocery', 'restaurant', 'gas_station', 'online', 'retail',
            'pharmacy', 'entertainment', 'travel', 'utilities', 'other',
            'gambling', 'cash_advance', 'wire_transfer'
        ]
        
        self.category_weights = [0.20, 0.15, 0.10, 0.15, 0.12, 0.08, 0.05, 0.04, 0.03, 0.05, 0.01, 0.01, 0.01]
        
        # Location pools (major cities)
        self.locations = [
            {'lat': 37.7749, 'lon': -122.4194, 'city': 'San Francisco'},
            {'lat': 40.7128, 'lon': -74.0060, 'city': 'New York'},
            {'lat': 34.0522, 'lon': -118.2437, 'city': 'Los Angeles'},
            {'lat': 41.8781, 'lon': -87.6298, 'city': 'Chicago'},
            {'lat': 29.7604, 'lon': -95.3698, 'city': 'Houston'},
            {'lat': 33.4484, 'lon': -112.0740, 'city': 'Phoenix'},
            {'lat': 39.7392, 'lon': -104.9903, 'city': 'Denver'},
            {'lat': 47.6062, 'lon': -122.3321, 'city': 'Seattle'},
            {'lat': 25.7617, 'lon': -80.1918, 'city': 'Miami'},
            {'lat': 32.7767, 'lon': -96.7970, 'city': 'Dallas'}
        ]
        
        logger.info(f"TransactionProducer initialized for topic: {topic}")
    
    def generate_transaction(self, fraud_probability=0.025) -> Dict[str, Any]:
        """Generate a single synthetic transaction"""
        
        # Basic transaction details
        transaction_id = f"TXN_{uuid.uuid4().hex[:12].upper()}"
        user_id = random.choice(self.users)
        device_id = random.choice(self.devices)
        merchant_id = random.choice(self.merchants)
        
        # Time-based features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        # Amount generation with realistic distribution
        if random.random() < 0.7:  # 70% normal transactions
            amount = np.random.lognormal(3.0, 1.0)
        elif random.random() < 0.2:  # 20% small transactions
            amount = np.random.uniform(1, 50)
        else:  # 10% large transactions
            amount = np.random.uniform(500, 5000)
        
        amount = round(max(0.01, amount), 2)
        
        # Category selection
        category = np.random.choice(self.merchant_categories, p=self.category_weights)
        
        # Location
        location = random.choice(self.locations).copy()
        # Add some noise to coordinates
        location['lat'] += np.random.normal(0, 0.01)
        location['lon'] += np.random.normal(0, 0.01)
        location['lat'] = round(location['lat'], 4)
        location['lon'] = round(location['lon'], 4)
        
        # Card type
        card_type = np.random.choice(['credit', 'debit', 'prepaid'], p=[0.6, 0.35, 0.05])
        
        # Generate fraud indicator based on risk factors
        fraud_risk = 0
        
        # Risk factors
        if amount > 2000:
            fraud_risk += 0.3
        if category in ['gambling', 'cash_advance', 'wire_transfer']:
            fraud_risk += 0.4
        if hour < 6 or hour > 22:  # Night transactions
            fraud_risk += 0.2
        if day_of_week >= 5:  # Weekend
            fraud_risk += 0.1
        if card_type == 'prepaid':
            fraud_risk += 0.2
        
        # Add random component
        fraud_risk += np.random.uniform(0, 0.1)
        
        # Determine if fraud
        is_fraud = fraud_risk > (1 - fraud_probability * 20)  # Adjust threshold
        
        # If marked as fraud, enhance some risk factors
        if is_fraud:
            if random.random() < 0.5:
                amount *= np.random.uniform(2, 5)  # Make amount suspicious
            if random.random() < 0.3:
                category = random.choice(['gambling', 'cash_advance', 'online'])
        
        transaction = {
            'transaction_id': transaction_id,
            'user_id': user_id,
            'device_id': device_id,
            'merchant_id': merchant_id,
            'amount': round(amount, 2),
            'merchant_category': category,
            'card_type': card_type,
            'location': location,
            'timestamp': now.isoformat(),
            'is_fraud': is_fraud,  # Ground truth (remove in production)
            'metadata': {
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5,
                'is_night': hour < 6 or hour > 22
            }
        }
        
        return transaction
    
    def generate_user_session(self, user_id: str, session_length: int = None) -> List[Dict[str, Any]]:
        """Generate a session of transactions for a specific user"""
        
        if session_length is None:
            session_length = np.random.poisson(3) + 1  # Average 3-4 transactions per session
        
        transactions = []
        base_time = datetime.now()
        
        # Select consistent location and device for session
        session_location = random.choice(self.locations).copy()
        session_device = random.choice(self.devices)
        
        for i in range(session_length):
            # Transactions within session are close in time
            time_offset = np.random.exponential(5)  # Minutes between transactions
            transaction_time = base_time + timedelta(minutes=i * time_offset)
            
            transaction = self.generate_transaction()
            
            # Override with session-specific data
            transaction['user_id'] = user_id
            transaction['device_id'] = session_device
            transaction['timestamp'] = transaction_time.isoformat()
            
            # Keep location consistent within session (with small variations)
            transaction['location'] = {
                'lat': session_location['lat'] + np.random.normal(0, 0.005),
                'lon': session_location['lon'] + np.random.normal(0, 0.005),
                'city': session_location['city']
            }
            transaction['location']['lat'] = round(transaction['location']['lat'], 4)
            transaction['location']['lon'] = round(transaction['location']['lon'], 4)
            
            transactions.append(transaction)
        
        return transactions
    
    def send_transaction(self, transaction: Dict[str, Any]):
        """Send a single transaction to Kafka"""
        
        try:
            # Remove ground truth before sending (simulate production)
            production_transaction = transaction.copy()
            if 'is_fraud' in production_transaction:
                del production_transaction['is_fraud']
            
            future = self.producer.send(self.topic, value=production_transaction)
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Sent transaction {transaction['transaction_id']} to {record_metadata.topic}[{record_metadata.partition}]")
            
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
    
    def send_batch(self, transactions: List[Dict[str, Any]]):
        """Send a batch of transactions"""
        
        for transaction in transactions:
            self.send_transaction(transaction)
        
        # Flush to ensure all messages are
