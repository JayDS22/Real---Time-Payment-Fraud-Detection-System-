import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import redis
import json
import time
from datetime import datetime, timedelta
import psycopg2
import os
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .success-card {
        border-left-color: #51cf66;
    }
    .warning-card {
        border-left-color: #ffd43b;
    }
    .info-card {
        border-left-color: #339af0;
    }
</style>
""", unsafe_allow_html=True)

class FraudMonitoringDashboard:
    def __init__(self):
        # Initialize connections
        self.redis_client = self._init_redis()
        self.postgres_conn = self._init_postgres()
        
        # Cache for data
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = 0
        
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            client.ping()
            return client
        except Exception as e:
            st.error(f"Failed to connect to Redis: {e}")
            return None
    
    def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            postgres_url = os.getenv('POSTGRES_URL', 'postgresql://frauddetection:frauddetection123@localhost:5432/fraud_detection')
            conn = psycopg2.connect(postgres_url)
            return conn
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL: {e}")
            return None
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics from Redis"""
        
        if not self.redis_client:
            return self._get_mock_metrics()
        
        try:
            # Get basic metrics
            fraud_detections_24h = self.redis_client.get('metrics:fraud_detections_24h') or 0
            total_transactions_24h = self.redis_client.get('metrics:total_transactions_24h') or 0
            avg_latency = self.redis_client.get('metrics:avg_latency_ms') or 0
            
            # Get hourly fraud rates
            hourly_fraud_rates = []
            for i in range(24):
                hour_key = f"metrics:fraud_rate_hour_{i}"
                rate = float(self.redis_client.get(hour_key) or 0)
                hourly_fraud_rates.append(rate)
            
            # Get top fraudulent features
            top_features = []
            feature_keys = self.redis_client.keys('feature_importance:*')
            for key in feature_keys[:10]:
                feature_name = key.split(':')[1]
                importance = float(self.redis_client.get(key) or 0)
                top_features.append({'feature': feature_name, 'importance': importance})
            
            top_features = sorted(top_features, key=lambda x: x['importance'], reverse=True)[:5]
            
            # Get recent predictions
            prediction_keys = self.redis_client.keys('prediction:*')
            recent_predictions = []
            
            for key in prediction_keys[-50:]:  # Get last 50 predictions
                try:
                    pred_data = json.loads(self.redis_client.get(key))
                    recent_predictions.append(pred_data)
                except:
                    continue
            
            # Calculate derived metrics
            fraud_rate = float(fraud_detections_24h) / max(float(total_transactions_24h), 1) * 100
            
            return {
                'total_transactions_24h': int(total_transactions_24h),
                'fraud_detections_24h': int(fraud_detections_24h),
                'fraud_rate': round(fraud_rate, 2),
                'avg_latency': round(float(avg_latency), 2),
                'hourly_fraud_rates': hourly_fraud_rates,
                'top_features': top_features,
                'recent_predictions': recent_predictions
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics from Redis: {e}")
            return self._get_mock_metrics()
    
    def _get_mock_metrics(self) -> Dict[str, Any]:
        """Get mock metrics for demo purposes"""
        
        np.random.seed(int(time.time()) % 1000)
        
        return {
            'total_transactions_24h': np.random.randint(45000, 55000),
            'fraud_detections_24h': np.random.randint(800, 1200),
            'fraud_rate': round(np.random.uniform(1.8, 2.5), 2),
            'avg_latency': round(np.random.uniform(65, 95), 2),
            'hourly_fraud_rates': [np.random.uniform(0.5, 4.0) for _ in range(24)],
            'top_features': [
                {'feature': 'amount_deviation', 'importance': 0.23},
                {'feature': 'transaction_velocity', 'importance': 0.18},
                {'feature': 'distance_from_home', 'importance': 0.15},
                {'feature': 'device_risk_score', 'importance': 0.12},
                {'feature': 'merchant_risk_score', 'importance': 0.10}
            ],
            'recent_predictions': self._generate_mock_predictions(50)
        }
    
    def _generate_mock_predictions(self, n: int) -> List[Dict[str, Any]]:
        """Generate mock predictions for demo"""
        
        predictions = []
        base_time = datetime.now()
        
        for i in range(n):
            pred_time = base_time - timedelta(minutes=np.random.randint(0, 60))
            
            is_fraud = np.random.choice([True, False], p=[0.03, 0.97])
            fraud_prob = np.random.uniform(0.7, 0.95) if is_fraud else np.random.uniform(0.01, 0.3)
            
            prediction = {
                'transaction_id': f'TXN_{np.random.randint(100000, 999999)}',
                'user_id': f'USER_{np.random.randint(1, 10000):06d}',
                'amount': round(np.random.lognormal(3, 1), 2),
                'is_fraud': is_fraud,
                'fraud_probability': round(fraud_prob, 3),
                'risk_score': round(fraud_prob * 100, 1),
                'processing_time_ms': round(np.random.uniform(50, 150), 2),
                'timestamp': pred_time.isoformat()
            }
            predictions.append(prediction)
        
        return sorted(predictions, key=lambda x: x['timestamp'], reverse=True)
    
    def get_model_performance_data(self) -> Dict[str, Any]:
        """Get model performance data"""
        
        # Generate mock performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        performance_data = {
            'dates': dates.tolist(),
            'precision': [np.random.uniform(0.92, 0.96) for _ in dates],
            'recall': [np.random.uniform(0.88, 0.94) for _ in dates],
            'f1_score': [np.random.uniform(0.90, 0.95) for _ in dates],
            'auc': [np.random.uniform(0.94, 0.98) for _ in dates],
            'false_positive_rate': [np.random.uniform(0.02, 0.05) for _ in dates]
        }
        
        return performance_data
    
    def render_header(self):
        """Render dashboard header"""
        
        st.title("🛡️ Real-Time Fraud Detection Monitoring")
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if self.redis_client:
                st.success("✅ Redis Connected")
            else:
                st.error("❌ Redis Disconnected")
        
        with col2:
            if self.postgres_conn:
                st.success("✅ Database Connected")
            else:
                st.warning("⚠️ Database Disconnected")
        
        with col3:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"🕒 Last Updated: {current_time}")
    
    def render_key_metrics(self, metrics: Dict[str, Any]):
        """Render key performance metrics"""
        
        st.subheader("📊 Key Metrics (Last 24 Hours)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value=f"{metrics['total_transactions_24h']:,}",
                delta="+5.2%"
            )
        
        with col2:
            st.metric(
                label="Fraud Detections",
                value=f"{metrics['fraud_detections_24h']:,}",
                delta="+2.1%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="Fraud Rate",
                value=f"{metrics['fraud_rate']}%",
                delta="-0.3%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="Avg Latency",
                value=f"{metrics['avg_latency']} ms",
                delta="-5 ms"
            )
    
    def render_fraud_trends(self, metrics: Dict[str, Any]):
        """Render fraud detection trends"""
        
        st.subheader("📈 Fraud Detection Trends")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Hourly fraud rate chart
            hours = list(range(24))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=metrics['hourly_fraud_rates'],
                mode='lines+markers',
                name='Fraud Rate %',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Fraud Rate by Hour (Last 24 Hours)",
                xaxis_title="Hour of Day",
                yaxis_title="Fraud Rate (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Fraud Rate Analysis")
            
            current_rate = metrics['fraud_rate']
            if current_rate < 2.0:
                st.success(f"✅ Normal: {current_rate}%")
            elif current_rate < 3.0:
                st.warning(f"⚠️ Elevated: {current_rate}%")
            else:
                st.error(f"🚨 High: {current_rate}%")
            
            st.markdown("#### 📊 Peak Hours")
            peak_hour = np.argmax(metrics['hourly_fraud_rates'])
            st.write(f"Peak: {peak_hour:02d}:00")
            st.write(f"Rate: {metrics['hourly_fraud_rates'][peak_hour]:.1f}%")
    
    def render_feature_importance(self, metrics: Dict[str, Any]):
        """Render feature importance chart"""
        
        st.subheader("🔍 Top Contributing Features")
        
        if metrics['top_features']:
            features_df = pd.DataFrame(metrics['top_features'])
            
            fig = px.bar(
                features_df,
                x='importance',
                y='feature',
                orientation='h',
                title="Most Important Features for Fraud Detection",
                color='importance',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available")
    
    def render_model_performance(self, performance_data: Dict[str, Any]):
        """Render model performance over time"""
        
        st.subheader("🎯 Model Performance Over Time")
        
        df = pd.DataFrame(performance_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision & Recall', 'F1 Score & AUC', 'False Positive Rate', 'Performance Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "indicator"}]]
        )
        
        # Precision & Recall
        fig.add_trace(
            go.Scatter(x=df['dates'], y=df['precision'], name='Precision', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['dates'], y=df['recall'], name='Recall', line=dict(color='green')),
            row=1, col=1
        )
        
        # F1 Score & AUC
        fig.add_trace(
            go.Scatter(x=df['dates'], y=df['f1_score'], name='F1 Score', line=dict(color='orange')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['dates'], y=df['auc'], name='AUC', line=dict(color='purple')),
            row=1, col=2
        )
        
        # False Positive Rate
        fig.add_trace(
            go.Scatter(x=df['dates'], y=df['false_positive_rate'], name='FPR', line=dict(color='red')),
            row=2, col=1
        )
        
        # Performance Summary Gauge
        current_f1 = df['f1_score'].iloc[-1]
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_f1,
                delta={'reference': 0.90},
                title={'text': "Current F1 Score"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.95
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_predictions(self, metrics: Dict[str, Any]):
        """Render recent predictions table"""
        
        st.subheader("📋 Recent Predictions")
        
        if metrics['recent_predictions']:
            df = pd.DataFrame(metrics['recent_predictions'])
            
            # Format the dataframe for display
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
            df['amount'] = df['amount'].apply(lambda x: f"${x:,.2f}")
            df['fraud_probability'] = df['fraud_probability'].apply(lambda x: f"{x:.1%}")
            df['processing_time_ms'] = df['processing_time_ms'].apply(lambda x: f"{x:.1f} ms")
            
            # Color code fraud predictions
            def highlight_fraud(row):
                if row['is_fraud']:
                    return ['background-color: #ffebee'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = df[['timestamp', 'transaction_id', 'user_id', 'amount', 
                          'is_fraud', 'fraud_probability', 'risk_score', 'processing_time_ms']].head(20)
            
            st.dataframe(
                styled_df.style.apply(highlight_fraud, axis=1),
                use_container_width=True
            )
        else:
            st.info("No recent predictions available")
    
    def render_system_health(self):
        """Render system health metrics"""
        
        st.subheader("🏥 System Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Redis Metrics")
            if self.redis_client:
                try:
                    info = self.redis_client.info()
                    st.write(f"Memory Used: {info.get('used_memory_human', 'N/A')}")
                    st.write(f"Connected Clients: {info.get('connected_clients', 'N/A')}")
                    st.write(f"Commands/sec: {info.get('instantaneous_ops_per_sec', 'N/A')}")
                    
                    # Memory usage gauge
                    memory_pct = (info.get('used_memory', 0) / info.get('maxmemory', 1)) * 100 if info.get('maxmemory', 0) > 0 else 25
                    
                    fig_memory = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=memory_pct,
                        title={'text': "Redis Memory Usage %"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 90}}
                    ))
                    fig_memory.update_layout(height=300)
                    st.plotly_chart(fig_memory, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error getting Redis metrics: {e}")
            else:
                st.error("Redis not connected")
        
        with col2:
            st.markdown("#### API Performance")
            
            # Mock API metrics
            api_metrics = {
                'requests_per_second': np.random.randint(800, 1200),
                'average_response_time': np.random.uniform(80, 120),
                'error_rate': np.random.uniform(0.1, 0.5),
                'uptime': "99.9%"
            }
            
            st.write(f"Requests/sec: {api_metrics['requests_per_second']}")
            st.write(f"Avg Response Time: {api_metrics['average_response_time']:.1f} ms")
            st.write(f"Error Rate: {api_metrics['error_rate']:.2f}%")
            st.write(f"Uptime: {api_metrics['uptime']}")
            
            # Response time gauge
            fig_response = go.Figure(go.Indicator(
                mode="gauge+number",
                value=api_metrics['average_response_time'],
                title={'text': "Avg Response Time (ms)"},
                gauge={'axis': {'range': [0, 200]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 100], 'color': "lightgray"},
                                {'range': [100, 150], 'color': "yellow"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 150}}
