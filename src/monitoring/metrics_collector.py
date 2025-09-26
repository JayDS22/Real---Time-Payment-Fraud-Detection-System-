import time
import psutil
import redis
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
import json
import threading

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect system and application metrics"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.registry = CollectorRegistry()
        
        # Prometheus metrics
        self.prediction_count = Counter('fraud_predictions_total', 'Total predictions made', registry=self.registry)
        self.fraud_detected = Counter('fraud_detected_total', 'Total fraud detected', registry=self.registry)
        self.prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', registry=self.registry)
        self.system_cpu = Gauge('system_cpu_percent', 'CPU usage', registry=self.registry)
        self.system_memory = Gauge('system_memory_percent', 'Memory usage', registry=self.registry)
        
        self.start_time = time.time()
        
    def record_prediction(self, is_fraud: bool, latency_ms: float):
        """Record a prediction event"""
        self.prediction_count.inc()
        if is_fraud:
            self.fraud_detected.inc()
        self.prediction_latency.observe(latency_ms / 1000.0)
        
        # Store in Redis for dashboard
        metrics_key = f"metrics:{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.redis_client.hincrby(metrics_key, 'total_predictions', 1)
        if is_fraud:
            self.redis_client.hincrby(metrics_key, 'fraud_detected', 1)
        self.redis_client.expire(metrics_key, 3600)  # 1 hour TTL
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self.system_cpu.set(cpu_percent)
        self.system_memory.set(memory.percent)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100,
            'disk_used_gb': disk.used / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'uptime_seconds': time.time() - self.start_time
        }
        
        # Store in Redis
        self.redis_client.setex(
            f"system_metrics:{datetime.now().strftime('%Y%m%d_%H%M')}",
            300,  # 5 minutes TTL
            json.dumps(metrics)
        )
        
        return metrics
    
    def get_fraud_rate(self, hours=1) -> float:
        """Calculate fraud rate over specified hours"""
        try:
            now = datetime.now()
            total_predictions = 0
            total_fraud = 0
            
            for i in range(hours * 60):  # Minutes
                time_key = (now - timedelta(minutes=i)).strftime('%Y%m%d_%H%M')
                metrics_key = f"metrics:{time_key}"
                
                predictions = int(self.redis_client.hget(metrics_key, 'total_predictions') or 0)
                fraud = int(self.redis_client.hget(metrics_key, 'fraud_detected') or 0)
                
                total_predictions += predictions
                total_fraud += fraud
            
            return (total_fraud / total_predictions * 100) if total_predictions > 0 else 0.0
        except:
            return 0.0
    
    def get_throughput(self, minutes=60) -> float:
        """Calculate throughput (predictions per minute)"""
        try:
            now = datetime.now()
            total_predictions = 0
            
            for i in range(minutes):
                time_key = (now - timedelta(minutes=i)).strftime('%Y%m%d_%H%M')
                metrics_key = f"metrics:{time_key}"
                predictions = int(self.redis_client.hget(metrics_key, 'total_predictions') or 0)
                total_predictions += predictions
            
            return total_predictions / minutes if minutes > 0 else 0.0
        except:
            return 0.0

class SystemMonitor:
    """Comprehensive system monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector, check_interval=30):
        self.metrics_collector = metrics_collector
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        
        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'fraud_rate': 5.0,
            'low_throughput': 1.0  # predictions per minute
        }
        
        self.alerts = []
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                # Check thresholds and generate alerts
                self._check_thresholds(system_metrics)
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_thresholds(self, system_metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        
        current_time = datetime.now()
        
        # CPU threshold
        if system_metrics['cpu_percent'] > self.thresholds['cpu_percent']:
            self._create_alert(
                'HIGH_CPU',
                f"CPU usage: {system_metrics['cpu_percent']:.1f}%",
                system_metrics['cpu_percent'],
                self.thresholds['cpu_percent']
            )
        
        # Memory threshold
        if system_metrics['memory_percent'] > self.thresholds['memory_percent']:
            self._create_alert(
                'HIGH_MEMORY',
                f"Memory usage: {system_metrics['memory_percent']:.1f}%",
                system_metrics['memory_percent'],
                self.thresholds['memory_percent']
            )
        
        # Disk threshold
        if system_metrics['disk_percent'] > self.thresholds['disk_percent']:
            self._create_alert(
                'HIGH_DISK',
                f"Disk usage: {system_metrics['disk_percent']:.1f}%",
                system_metrics['disk_percent'],
                self.thresholds['disk_percent']
            )
        
        # Fraud rate threshold
        fraud_rate = self.metrics_collector.get_fraud_rate(hours=1)
        if fraud_rate > self.thresholds['fraud_rate']:
            self._create_alert(
                'HIGH_FRAUD_RATE',
                f"Fraud rate: {fraud_rate:.2f}%",
                fraud_rate,
                self.thresholds['fraud_rate']
            )
        
        # Throughput threshold
        throughput = self.metrics_collector.get_throughput(minutes=10)
        if throughput < self.thresholds['low_throughput']:
            self._create_alert(
                'LOW_THROUGHPUT',
                f"Throughput: {throughput:.1f} predictions/min",
                throughput,
                self.thresholds['low_throughput']
            )
    
    def _create_alert(self, alert_type: str, message: str, current_value: float, threshold: float):
        """Create and store an alert"""
        
        alert = {
            'id': f"{alert_type}_{int(time.time())}",
            'type': alert_type,
            'message': message,
            'current_value': current_value,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat(),
            'severity': self._get_alert_severity(current_value, threshold)
        }
        
        # Store alert
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Store in Redis
        self.metrics_collector.redis_client.lpush('system_alerts', json.dumps(alert))
        self.metrics_collector.redis_client.ltrim('system_alerts', 0, 99)  # Keep last 100
        
        logger.warning(f"ALERT: {message}")
    
    def _get_alert_severity(self, current_value: float, threshold: float) -> str:
        """Determine alert severity"""
        ratio = current_value / threshold
        
        if ratio >= 1.5:
            return 'CRITICAL'
        elif ratio >= 1.2:
            return 'HIGH'
        elif ratio >= 1.0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        try:
            alerts_data = self.metrics_collector.redis_client.lrange('system_alerts', 0, count - 1)
            alerts = [json.loads(alert_data) for alert_data in alerts_data]
            return alerts
        except:
            return []
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        
        # Get latest metrics
        system_metrics = self.metrics_collector.collect_system_metrics()
        fraud_rate = self.metrics_collector.get_fraud_rate(hours=1)
        throughput = self.metrics_collector.get_throughput(minutes=10)
        recent_alerts = self.get_recent_alerts(5)
        
        # Determine overall health status
        critical_alerts = [alert for alert in recent_alerts if alert.get('severity') == 'CRITICAL']
        high_alerts = [alert for alert in recent_alerts if alert.get('severity') == 'HIGH']
        
        if critical_alerts:
            health_status = 'CRITICAL'
        elif high_alerts:
            health_status = 'DEGRADED'
        elif len(recent_alerts) > 0:
            health_status = 'WARNING'
        else:
            health_status = 'HEALTHY'
        
        return {
            'status': health_status,
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system_metrics,
            'fraud_rate_1h': fraud_rate,
            'throughput_10m': throughput,
            'recent_alerts_count': len(recent_alerts),
            'uptime_hours': system_metrics['uptime_seconds'] / 3600
        }
