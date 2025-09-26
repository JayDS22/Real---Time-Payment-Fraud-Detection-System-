import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import joblib
import os
from datetime import datetime
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudModelTrainer:
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # MLflow setup
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("fraud_detection")
    
    def generate_synthetic_data(self, n_samples=100000):
        """Generate synthetic transaction data for training"""
        
        logger.info(f"Generating {n_samples} synthetic transactions...")
        
        np.random.seed(42)
        
        # Generate base features
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
            'user_id': [f'USER_{np.random.randint(1, 10000):06d}' for _ in range(n_samples)],
            'amount': np.random.lognormal(3, 1.5, n_samples),  # Log-normal distribution
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'merchant_category': np.random.choice(
                ['grocery', 'restaurant', 'gas_station', 'online', 'gambling', 'cash_advance', 'other'],
                n_samples,
                p=[0.25, 0.2, 0.15, 0.15, 0.05, 0.05, 0.15]
            ),
            'device_id': [f'DEV_{np.random.randint(1, 5000):06d}' for _ in range(n_samples)],
            'merchant_id': [f'MERCH_{np.random.randint(1, 2000):06d}' for _ in range(n_samples)],
        }
        
        df = pd.DataFrame(data)
        
        # Feature engineering
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        df['log_amount'] = np.log1p(df['amount'])
        
        # User-level features (simplified aggregation)
        user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
        user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_txn_count']
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Amount deviation
        df['amount_deviation'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-5)
        
        # Merchant-level features
        merchant_stats = df.groupby('merchant_id')['amount'].agg(['mean', 'count']).reset_index()
        merchant_stats.columns = ['merchant_id', 'merchant_avg_amount', 'merchant_txn_count']
        df = df.merge(merchant_stats, on='merchant_id', how='left')
        
        # Device features
        device_stats = df.groupby('device_id').size().reset_index()
        device_stats.columns = ['device_id', 'device_txn_count']
        df = df.merge(device_stats, on='device_id', how='left')
        
        # Category risk scores
        category_risk = {
            'grocery': 0.02, 'restaurant': 0.03, 'gas_station': 0.02,
            'online': 0.08, 'gambling': 0.25, 'cash_advance': 0.15, 'other': 0.05
        }
        df['category_risk'] = df['merchant_category'].map(category_risk)
        
        # Velocity features (simplified)
        df['txn_velocity_1h'] = np.random.poisson(0.5, n_samples)
        df['txn_velocity_24h'] = np.random.poisson(3, n_samples)
        df['txn_velocity_7d'] = np.random.poisson(15, n_samples)
        
        # Distance features
        df['distance_from_home'] = np.random.exponential(20, n_samples)
        
        # Device and merchant risk scores
        df['device_risk_score'] = np.random.beta(2, 8, n_samples)  # Skewed towards low risk
        df['merchant_risk_score'] = np.random.beta(2, 8, n_samples)
        
        # Time-based features
        df['time_since_last_txn'] = np.random.exponential(12, n_samples)  # Hours
        
        # Interaction features
        df['amount_hour_interaction'] = df['amount'] * df['hour']
        df['amount_weekend_interaction'] = df['amount'] * df['is_weekend']
        df['velocity_amount_interaction'] = df['txn_velocity_24h'] * df['amount_deviation']
        
        # Additional behavioral features
        df['amount_percentile_user'] = df.groupby('user_id')['amount'].rank(pct=True)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Generate fraud labels with realistic patterns
        fraud_prob = (
            df['category_risk'] * 0.3 +
            (df['amount'] > df['amount'].quantile(0.95)).astype(int) * 0.2 +
            df['is_night'] * 0.1 +
            (df['amount_deviation'] > 3).astype(int) * 0.2 +
            (df['distance_from_home'] > 100).astype(int) * 0.1 +
            df['device_risk_score'] * 0.1 +
            df['merchant_risk_score'] * 0.1 +
            (df['txn_velocity_24h'] > 10).astype(int) * 0.15 +
            np.random.normal(0, 0.05, n_samples)  # Add noise
        )
        
        # Clip probabilities
        fraud_prob = np.clip(fraud_prob, 0, 1)
        
        # Generate labels
        df['is_fraud'] = np.random.binomial(1, fraud_prob)
        
        # Ensure realistic fraud rate (around 2-3%)
        fraud_rate = df['is_fraud'].mean()
        if fraud_rate > 0.05:  # If too high, randomly flip some positives to negatives
            excess_frauds = df[df['is_fraud'] == 1].sample(frac=0.5).index
            df.loc[excess_frauds, 'is_fraud'] = 0
        
        logger.info(f"Generated data with fraud rate: {df['is_fraud'].mean():.3f}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        
        # Select numeric features
        numeric_features = [
            'amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'log_amount', 'user_avg_amount', 'user_std_amount', 'user_txn_count',
            'amount_deviation', 'merchant_avg_amount', 'merchant_txn_count',
            'device_txn_count', 'category_risk', 'txn_velocity_1h',
            'txn_velocity_24h', 'txn_velocity_7d', 'distance_from_home',
            'device_risk_score', 'merchant_risk_score', 'time_since_last_txn',
            'amount_hour_interaction', 'amount_weekend_interaction',
            'velocity_amount_interaction', 'amount_percentile_user',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Encode categorical features
        categorical_features = ['merchant_category']
        
        X = df[numeric_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Add encoded categorical features
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col])
            
            encoded_col = f"{col}_encoded"
            X[encoded_col] = self.label_encoders[col].transform(df[col])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        y = df['is_fraud']
        
        return X, y
    
    def optimize_xgboost(self, X_train, y_train):
        """Optimize XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train):
        """Optimize LightGBM hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbosity': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble models"""
        
        logger.info("Training ensemble models...")
        
        with mlflow.start_run(run_name=f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Handle class imbalance with SMOTE
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            # Log dataset info
            mlflow.log_params({
                'n_samples_train': len(X_train),
                'n_samples_train_balanced': len(X_train_balanced),
                'n_features': X_train.shape[1],
                'fraud_rate_original': y_train.mean(),
                'fraud_rate_balanced': y_train_balanced.mean()
            })
            
            # 1. Random Forest
            logger.info("Training Random Forest...")
            rf_params = {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
            
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train_balanced, y_train_balanced)
            
            rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            rf_auc = roc_auc_score(y_test, rf_pred_proba)
            
            self.models['rf'] = rf_model
            mlflow.sklearn.log_model(rf_model, "random_forest")
            mlflow.log_metric("rf_auc", rf_auc)
            
            # 2. XGBoost with optimization
            logger.info("Optimizing and training XGBoost...")
            xgb_best_params = self.optimize_xgboost(X_train_balanced, y_train_balanced)
            
            xgb_model = xgb.XGBClassifier(**xgb_best_params)
            xgb_model.fit(X_train_balanced, y_train_balanced)
            
            xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
            
            self.models['xgb'] = xgb_model
            mlflow.xgboost.log_model(xgb_model, "xgboost")
            mlflow.log_metric("xgb_auc", xgb_auc)
            mlflow.log_params({f"xgb_{k}": v for k, v in xgb_best_params.items()})
            
            # 3. LightGBM with optimization
            logger.info("Optimizing and training LightGBM...")
            lgb_best_params = self.optimize_lightgbm(X_train_balanced, y_train_balanced)
            
            lgb_model = lgb.LGBMClassifier(**lgb_best_params)
            lgb_model.fit(X_train_balanced, y_train_balanced)
            
            lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
            lgb_auc = roc_auc_score(y_test, lgb_pred_proba)
            
            self.models['lgb'] = lgb_model
            mlflow.lightgbm.log_model(lgb_model, "lightgbm")
            mlflow.log_metric("lgb_auc", lgb_auc)
            mlflow.log_params({f"lgb_{k}": v for k, v in lgb_best_params.items()})
            
            # Ensemble prediction
            model_weights = {'rf': 0.3, 'xgb': 0.4, 'lgb': 0.3}
            ensemble_pred_proba = (
                model_weights['rf'] * rf_pred_proba +
                model_weights['xgb'] * xgb_pred_proba +
                model_weights['lgb'] * lgb_pred_proba
            )
            
            ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
            
            # Log ensemble metrics
            mlflow.log_metric("ensemble_auc", ensemble_auc)
            mlflow.log_params(model_weights)
            
            logger.info(f"Model AUC scores - RF: {rf_auc:.4f}, XGB: {xgb_auc:.4f}, LGB: {lgb_auc:.4f}, Ensemble: {ensemble_auc:.4f}")
            
            # Detailed evaluation
            self.evaluate_models(y_test, ensemble_pred, ensemble_pred_proba)
            
            # Feature importance analysis
            self.analyze_feature_importance()
            
            # Generate SHAP explanations
            self.generate_shap_explanations(X_test.iloc[:1000])  # Use subset for speed
            
        return self.models
    
    def evaluate_models(self, y_true, y_pred, y_pred_proba):
        """Comprehensive model evaluation"""
        
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        # False positive rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'auc': auc,
            'false_positive_rate': fpr
        }
        
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            mlflow.log_metric(f"ensemble_{metric}", value)
        
        # Classification report
        report = classification_report(y_true, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('reports/confusion_matrix.png')
        mlflow.log_artifact('reports/confusion_matrix.png')
        plt.close()
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig('reports/precision_recall_curve.png')
        mlflow.log_artifact('reports/precision_recall_curve.png')
        plt.close()
        
        return metrics
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        
        importance_data = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=self.feature_names)
            importance_df['avg_importance'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('avg_importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 10))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features.reset_index(), x='avg_importance', y='index')
            plt.title('Top 20 Feature Importances (Average)')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('reports/feature_importance.png')
            mlflow.log_artifact('reports/feature_importance.png')
            plt.close()
            
            # Save feature importance data
            importance_df.to_csv('reports/feature_importance.csv')
            mlflow.log_artifact('reports/feature_importance.csv')
            
            logger.info("Top 10 most important features:")
            for i, (feature, importance) in enumerate(importance_df['avg_importance'].head(10).items()):
                logger.info(f"{i+1}. {feature}: {importance:.4f}")
    
    def generate_shap_explanations(self, X_sample):
        """Generate SHAP explanations for model interpretability"""
        
        try:
            # Use XGBoost model for SHAP (fastest for tree models)
            explainer = shap.TreeExplainer(self.models['xgb'])
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('reports/shap_summary.png')
            plt.close()
            
            # Feature importance plot
            plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('reports/shap_importance.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('reports/shap_importance.png')
            plt.close()
            
            logger.info("SHAP explanations generated and saved")
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP explanations: {e}")
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}_model.pkl')
        
        # Save preprocessing objects
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        logger.info("Models and preprocessing objects saved successfully")
    
    def train_pipeline(self):
        """Complete training pipeline"""
        
        logger.info("Starting fraud detection model training pipeline...")
        
        # Generate or load data
        if os.path.exists('data/transactions.csv'):
            logger.info("Loading existing transaction data...")
            df = pd.read_csv('data/transactions.csv')
        else:
            logger.info("Generating synthetic transaction data...")
            df = self.generate_synthetic_data(n_samples=100000)
            df.to_csv('data/transactions.csv', index=False)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.4f}")
        
        # Prepare features
        X, y = self.prepare_features(df)
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Train models
        self.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Save models
        self.save_models()
        
        logger.info("Training pipeline completed successfully!")
        logger.info("Models saved to 'models/' directory")
        logger.info("Reports saved to 'reports/' directory")

def main():
    trainer = FraudModelTrainer()
    trainer.train_pipeline()

if __name__ == "__main__":
    main()
