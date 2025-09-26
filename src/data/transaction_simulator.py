import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import numpy as np
import pandas as pd
from kafka import KafkaProducer
import redis
from .data_generator import FraudDataGenerator

logger = logging.getLogger(__name__)

class TransactionSimulator:
    """Simulates realistic transaction patterns and fraud scenarios"""
    
    def __init__(self, 
                 kafka_servers: str = 'localhost:9092',
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 topic: str = 'transactions'):
        """Initialize the transaction simulator
        
        Args:
            kafka_servers: Kafka bootstrap servers
            redis_host: Redis host for state management
            redis_port: Redis port
            topic: Kafka topic to send transactions to
        """
        self.kafka_servers = kafka_servers
        self.topic = topic
        
        # Initialize components
        self.data_generator = FraudDataGenerator()
        self.kafka_producer = None
        self.redis_client = None
        
        # Simulation state
        self.is_running = False
        self.simulation_stats = {
            'total_transactions': 0,
            'fraud_transactions': 0,
            'start_time': None,
            'scenarios_executed': []
        }
        
        # User session management
        self.active_sessions = {}
        self.user_profiles = {}
        
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            
        self._setup_kafka_producer()
        
    def _setup_kafka_producer(self):
        """Setup Kafka producer"""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
                batch_size=16384,
                linger_ms=10,
                compression_type='gzip',
                acks=1,
                retries=3
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.kafka_producer = None
    
    def _send_transaction(self, transaction: Dict[str, Any]):
        """Send transaction to Kafka and update stats"""
        try:
            if self.kafka_producer:
                # Remove ground truth before sending (simulate production)
                production_txn = transaction.copy()
                is_fraud = production_txn.pop('is_fraud', False)
                
                # Send to Kafka
                self.kafka_producer.send(self.topic, value=production_txn)
                
                # Update stats
                self.simulation_stats['total_transactions'] += 1
                if is_fraud:
                    self.simulation_stats['fraud_transactions'] += 1
                    
                # Store in Redis for monitoring
                if self.redis_client:
                    self.redis_client.setex(
                        f"sim_txn:{transaction['transaction_id']}", 
                        3600, 
                        json.dumps(transaction, default=str)
                    )
                    
            logger.debug(f"Sent transaction {transaction['transaction_id']}")
            
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
    
    def simulate_normal_traffic(self, 
                              duration_minutes: int = 60,
                              base_tps: float = 10.0,
                              callback: Optional[Callable] = None):
        """Simulate normal transaction traffic with realistic patterns
        
        Args:
            duration_minutes: Simulation duration in minutes
            base_tps: Base transactions per second
            callback: Optional callback function for each transaction
        """
        logger.info(f"Starting normal traffic simulation: {duration_minutes}m at {base_tps} TPS")
        
        self.is_running = True
        self.simulation_stats['start_time'] = datetime.now()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Generate user profiles for simulation
        n_users = min(1000, int(base_tps * 100))  # Scale users with TPS
        user_ids = [f"SIM_USER_{i:06d}" for i in range(1, n_users + 1)]
        
        for user_id in user_ids:
            self.user_profiles[user_id] = self.data_generator.generate_user_profile(user_id)
        
        transaction_count = 0
        
        try:
            while time.time() < end_time and self.is_running:
                current_time = datetime.now()
                
                # Adjust TPS based on time of day (realistic pattern)
                time_multiplier = self._get_traffic_multiplier(current_time.hour)
                current_tps = base_tps * time_multiplier
                
                # Add some randomness
                current_tps *= np.random.uniform(0.8, 1.2)
                
                # Generate transactions for this second
                transactions_this_second = np.random.poisson(current_tps)
                
                for _ in range(transactions_this_second):
                    if not self.is_running:
                        break
                        
                    # Select random user
                    user_id = np.random.choice(user_ids)
                    user_profile = self.user_profiles[user_id]
                    
                    # Generate transaction
                    transaction = self.data_generator.generate_transaction(
                        user_profile, 
                        current_time
                    )
                    
                    # Send transaction
                    self._send_transaction(transaction)
                    transaction_count += 1
                    
                    # Call callback if provided
                    if callback:
                        callback(transaction)
                
                # Log progress periodically
                if transaction_count > 0 and transaction_count % (int(base_tps * 60)) == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    current_fraud_rate = (self.simulation_stats['fraud_transactions'] / 
                                        max(self.simulation_stats['total_transactions'], 1)) * 100
                    logger.info(
                        f"Simulation progress: {elapsed_minutes:.1f}m, "
                        f"{transaction_count} transactions, "
                        f"{current_fraud_rate:.2f}% fraud rate"
                    )
                
                # Sleep for remainder of second
                time.sleep(max(0, 1.0 - (time.time() % 1)))
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.is_running = False
            if self.kafka_producer:
                self.kafka_producer.flush()
            
        elapsed_time = time.time() - start_time
        avg_tps = transaction_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"Normal traffic simulation completed:")
        logger.info(f"  Duration: {elapsed_time/60:.1f} minutes")
        logger.info(f"  Total transactions: {transaction_count}")
        logger.info(f"  Average TPS: {avg_tps:.1f}")
        logger.info(f"  Fraud rate: {(self.simulation_stats['fraud_transactions']/max(transaction_count, 1))*100:.2f}%")
    
    def simulate_fraud_attack(self,
                            duration_minutes: int = 10,
                            attack_intensity: float = 5.0,
                            attack_type: str = 'account_takeover'):
        """Simulate a coordinated fraud attack
        
        Args:
            duration_minutes: Attack duration in minutes
            attack_intensity: Fraudulent transactions per second
            attack_type: Type of fraud attack to simulate
        """
        logger.info(f"Simulating {attack_type} fraud attack: {duration_minutes}m at {attack_intensity} fraud TPS")
        
        self.is_running = True
        attack_scenario = {
            'type': attack_type,
            'start_time': datetime.now(),
            'duration_minutes': duration_minutes,
            'intensity': attack_intensity
        }
        self.simulation_stats['scenarios_executed'].append(attack_scenario)
        
        # Setup attack parameters based on type
        attack_config = self._get_attack_config(attack_type)
        
        # Generate compromised accounts
        compromised_users = self._generate_compromised_users(attack_config)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        fraud_count = 0
        
        try:
            while time.time() < end_time and self.is_running:
                current_time = datetime.now()
                
                # Generate burst of fraudulent transactions
                fraud_burst_size = np.random.poisson(attack_intensity)
                
                for _ in range(fraud_burst_size):
                    if not self.is_running:
                        break
                    
                    # Select compromised user
                    user_id = np.random.choice(list(compromised_users.keys()))
                    user_profile = compromised_users[user_id]
                    
                    # Generate fraudulent transaction
                    transaction = self.data_generator.generate_transaction(
                        user_profile,
                        current_time,
                        force_fraud=True
                    )
                    
                    # Apply attack-specific modifications
                    transaction = self._apply_attack_pattern(transaction, attack_config)
                    
                    self._send_transaction(transaction)
                    fraud_count += 1
                
                # Also generate some normal traffic to mix in
                normal_tps = np.random.poisson(2.0)  # Lower normal traffic during attack
                for _ in range(normal_tps):
                    user_id = np.random.choice(list(self.user_profiles.keys()))
                    user_profile = self.user_profiles[user_id]
                    
                    transaction = self.data_generator.generate_transaction(
                        user_profile,
                        current_time
                    )
                    self._send_transaction(transaction)
                
                # Log attack progress
                if fraud_count > 0 and fraud_count % 50 == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    logger.info(f"Attack progress: {elapsed_minutes:.1f}m, {fraud_count} fraud transactions")
                
                # Random delay between bursts
                time.sleep(np.random.uniform(0.5, 2.0))
                
        except KeyboardInterrupt:
            logger.info("Fraud attack simulation interrupted")
        finally:
            self.is_running = False
            if self.kafka_producer:
                self.kafka_producer.flush()
        
        elapsed_time = time.time() - start_time
        attack_scenario['end_time'] = datetime.now()
        attack_scenario['fraud_transactions_generated'] = fraud_count
        
        logger.info(f"Fraud attack simulation completed:")
        logger.info(f"  Attack type: {attack_type}")
        logger.info(f"  Duration: {elapsed_time/60:.1f} minutes") 
        logger.info(f"  Fraud transactions: {fraud_count}")
        logger.info(f"  Fraud rate during attack: {(fraud_count/(fraud_count+100))*100:.1f}%")
    
    def simulate_user_session(self, 
                            user_id: str,
                            session_length: int = 5,
                            include_fraud: bool = False) -> List[Dict[str, Any]]:
        """Simulate a realistic user session with multiple transactions
        
        Args:
            user_id: User identifier
            session_length: Number of transactions in session
            include_fraud: Whether to include fraudulent transactions
            
        Returns:
            List of transactions in the session
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self.data_generator.generate_user_profile(user_id)
            
        user_profile = self.user_profiles[user_id]
        
        # Generate session with realistic timing
        session_transactions = []
        session_start = datetime.now()
        
        # Session characteristics
        session_location = user_profile['home_location']
        session_device = f"DEV_{np.random.randint(1, 1000):06d}"
        
        for i in range(session_length):
            # Transactions within session are close in time
            time_offset_minutes = np.random.exponential(15)  # Average 15 min between
            transaction_time = session_start + timedelta(minutes=i * time_offset_minutes)
            
            # Determine if this transaction should be fraudulent
            force_fraud = include_fraud and (i == session_length - 1)  # Last transaction
            
            transaction = self.data_generator.generate_transaction(
                user_profile,
                transaction_time,
                force_fraud=force_fraud
            )
            
            # Apply session consistency
            transaction['device_id'] = session_device
            transaction['location'] = {
                'lat': session_location['lat'] + np.random.normal(0, 0.002),
                'lon': session_location['lon'] + np.random.normal(0, 0.002),
                'city': session_location['city']
            }
            
            session_transactions.append(transaction)
            self._send_transaction(transaction)
        
        # Store session in Redis
        if self.redis_client:
            session_key = f"session:{user_id}:{session_start.strftime('%Y%m%d_%H%M%S')}"
            self.redis_client.setex(
                session_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'user_id': user_id,
                    'start_time': session_start.isoformat(),
                    'transactions': [t['transaction_id'] for t in session_transactions],
                    'includes_fraud': include_fraud
                }, default=str)
            )
        
        logger.info(f"Generated session for {user_id}: {len(session_transactions)} transactions")
        return session_transactions
    
    def _get_traffic_multiplier(self, hour: int) -> float:
        """Get traffic multiplier based on hour of day"""
        # Realistic hourly traffic patterns
        hourly_multipliers = {
            0: 0.1, 1: 0.05, 2: 0.03, 3: 0.02, 4: 0.02, 5: 0.05,
            6: 0.2, 7: 0.5, 8: 0.8, 9: 1.0, 10: 1.1, 11: 1.2,
            12: 1.3, 13: 1.2, 14: 1.1, 15: 1.0, 16: 0.9, 17: 0.8,
            18: 0.7, 19: 0.6, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.2
        }
        return hourly_multipliers.get(hour, 0.5)
    
    def _get_attack_config(self, attack_type: str) -> Dict[str, Any]:
        """Get configuration for different attack types"""
        attack_configs = {
            'account_takeover': {
                'target_users': 50,
                'new_locations': True,
                'high_amounts': True,
                'unusual_times': True,
                'rapid_succession': True
            },
            'card_testing': {
                'target_users': 100,
                'small_amounts': True,
                'online_merchants': True,
                'rapid_succession': True,
                'multiple_cards': True
            },
            'synthetic_identity': {
                'target_users': 30,
                'new_accounts': True,
                'gradual_buildup': True,
                'diverse_categories': True
            },
            'merchant_compromise': {
                'target_merchants': 5,
                'multiple_users': 200,
                'specific_location': True,
                'normal_amounts': True
            }
        }
        return attack_configs.get(attack_type, attack_configs['account_takeover'])
    
    def _generate_compromised_users(self, attack_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate compromised user profiles for attack simulation"""
        n_users = attack_config.get('target_users', 50)
        compromised_users = {}
        
        for i in range(n_users):
            user_id = f"COMPROMISED_{i:04d}"
            user_profile = self.data_generator.generate_user_profile(user_id)
            
            # Modify profile based on attack type
            if attack_config.get('new_accounts'):
                user_profile['registration_date'] = datetime.now() - timedelta(days=np.random.randint(1, 30))
                
            # Increase fraud propensity for attack
            user_profile['fraud_propensity'] = 0.8
            
            compromised_users[user_id] = user_profile
            
        return compromised_users
    
    def _apply_attack_pattern(self, transaction: Dict[str, Any], attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply attack-specific patterns to transaction"""
        
        if attack_config.get('high_amounts'):
            transaction['amount'] *= np.random.uniform(5, 15)
            
        if attack_config.get('small_amounts'):
            transaction['amount'] = np.random.uniform(1, 10)
            
        if attack_config.get('online_merchants'):
            transaction['merchant_category'] = 'online'
            
        if attack_config.get('unusual_times'):
            # Set to unusual hour
            timestamp = datetime.fromisoformat(transaction['timestamp'])
            unusual_hour = np.random.choice([2, 3, 4, 5])
            timestamp = timestamp.replace(hour=unusual_hour)
            transaction['timestamp'] = timestamp.isoformat()
            
        if attack_config.get('new_locations'):
            # Use random distant location
            new_location = np.random.choice(self.data_generator.locations)
            transaction['location'] = {
                'lat': new_location['lat'],
                'lon': new_location['lon'], 
                'city': new_location['city']
            }
            
        return transaction
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get current simulation statistics"""
        stats = self.simulation_stats.copy()
        
        if stats['start_time']:
            elapsed_time = (datetime.now() - stats['start_time']).total_seconds()
            stats['elapsed_minutes'] = elapsed_time / 60
            stats['avg_tps'] = stats['total_transactions'] / max(elapsed_time, 1)
            stats['fraud_rate'] = (stats['fraud_transactions'] / max(stats['total_transactions'], 1)) * 100
            
        return stats
    
    def stop_simulation(self):
        """Stop the running simulation"""
        logger.info("Stopping simulation...")
        self.is_running = False
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("Simulation cleanup completed")

# CLI interface
def main():
    """Main CLI interface for transaction simulator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transaction Simulator for Fraud Detection')
    parser.add_argument('--mode', choices=['normal', 'attack', 'session'], default='normal')
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    parser.add_argument('--tps', type=float, default=10.0, help='Transactions per second')
    parser.add_argument('--attack-type', default='account_takeover')
    parser.add_argument('--attack-intensity', type=float, default=5.0)
    parser.add_argument('--user-id', help='User ID for session simulation')
    parser.add_argument('--kafka-servers', default='localhost:9092')
    parser.add_argument('--redis-host', default='localhost')
    parser.add_argument('--topic', default='transactions')
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = TransactionSimulator(
        kafka_servers=args.kafka_servers,
        redis_host=args.redis_host,
        topic=args.topic
    )
    
    try:
        if args.mode == 'normal':
            simulator.simulate_normal_traffic(args.duration, args.tps)
        elif args.mode == 'attack':
            simulator.simulate_fraud_attack(
                args.duration, 
                args.attack_intensity, 
                args.attack_type
            )
        elif args.mode == 'session':
            user_id = args.user_id or 'TEST_USER_001'
            simulator.simulate_user_session(user_id, include_fraud=True)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted")
    finally:
        simulator.cleanup()

if __name__ == "__main__":
    main()
