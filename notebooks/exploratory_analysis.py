# %% [markdown]
# # Fraud Detection - Exploratory Data Analysis
#
# This notebook provides exploratory data analysis for the fraud detection system.
#
# ## Contents
# 1. Data Loading and Overview
# 2. Fraud Distribution Analysis
# 3. Feature Analysis
# 4. Transaction Patterns
# 5. Geographic Analysis
# 6. Correlation Analysis
# 7. Model Performance Analysis
# 8. Business Impact Analysis

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, roc_curve, precision_recall_curve, classification_report
)
import sys
import os

# Add src to path for imports
sys.path.append('../src')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("✅ Imports successful")

# %% [markdown]
# ## 1. Data Loading and Overview

# %%
# Generate sample data using our data generator
from data.data_generator import FraudDataGenerator

# Initialize data generator
generator = FraudDataGenerator(seed=42)

# Generate synthetic dataset
print("Generating synthetic fraud detection dataset...")
df = generator.generate_dataset(n_samples=10000, n_users=1000, date_range_days=90)

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
print(f"Total fraudulent amount: ${df[df['is_fraud']==1]['amount'].sum():,.2f}")

# Display first few rows
df.head()

# %%
# Dataset information and statistics
print("=== DATASET OVERVIEW ===")
print(f"Total transactions: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Unique merchants: {df['merchant_id'].nunique():,}")
print(f"Unique devices: {df['device_id'].nunique():,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

print("\n=== DATA QUALITY ===")
print("Missing values:")
print(df.isnull().sum())

print("\n=== BASIC STATISTICS ===")
print("Amount statistics:")
print(df['amount'].describe())

# %% [markdown]
# ## 2. Fraud Distribution Analysis

# %%
# Analyze fraud distribution
fraud_counts = df['is_fraud'].value_counts()
fraud_pct = df['is_fraud'].value_counts(normalize=True) * 100

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Count plot
fraud_counts.plot(kind='bar', ax=ax1, color=['lightblue', 'salmon'])
ax1.set_title('Fraud Distribution (Count)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Transaction Type')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['Normal', 'Fraud'], rotation=0)
for i, v in enumerate(fraud_counts.values):
    ax1.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

# Percentage pie chart
colors = ['lightgreen', 'lightcoral']
ax2.pie(fraud_pct.values, labels=['Normal', 'Fraud'], autopct='%1.2f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12})
ax2.set_title('Fraud Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"📊 FRAUD DISTRIBUTION SUMMARY:")
print(f"Normal transactions: {fraud_counts[0]:,} ({fraud_pct[0]:.2f}%)")
print(f"Fraudulent transactions: {fraud_counts[1]:,} ({fraud_pct[1]:.2f}%)")

# %%
# Fraud by merchant category
category_analysis = df.groupby('merchant_category').agg({
    'is_fraud': ['count', 'sum', 'mean'],
    'amount': ['sum', 'mean']
}).round(3)

category_analysis.columns = ['Total_Txns', 'Fraud_Count', 'Fraud_Rate', 'Total_Amount', 'Avg_Amount']
category_analysis = category_analysis.sort_values('Fraud_Rate', ascending=False)

print("📈 FRAUD BY MERCHANT CATEGORY:")
print(category_analysis)

# Visualize fraud rate by category
plt.figure(figsize=(12, 6))
category_analysis['Fraud_Rate'].plot(kind='bar', color='coral')
plt.title('Fraud Rate by Merchant Category', fontsize=14, fontweight='bold')
plt.ylabel('Fraud Rate')
plt.xlabel('Merchant Category')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(category_analysis['Fraud_Rate'].values):
    plt.text(i, v + 0.001, f'{v:.1%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Feature Analysis

# %%
# Amount analysis by fraud status
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Box plot: Amount distribution by fraud status
df.boxplot(column='amount', by='is_fraud', ax=axes[0,0])
axes[0,0].set_title('Amount Distribution by Fraud Status')
axes[0,0].set_xlabel('Is Fraud (0=Normal, 1=Fraud)')
axes[0,0].set_ylabel('Transaction Amount ($)')

# Log amount distribution
df['log_amount'] = np.log1p(df['amount'])
axes[0,1].hist(df[df['is_fraud']==0]['log_amount'], bins=50, alpha=0.7, 
               label='Normal', color='lightblue', density=True)
axes[0,1].hist(df[df['is_fraud']==1]['log_amount'], bins=50, alpha=0.7, 
               label='Fraud', color='salmon', density=True)
axes[0,1].set_title('Log Amount Distribution')
axes[0,1].set_xlabel('Log(Amount + 1)')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# Amount statistics by fraud status
amount_stats = df.groupby('is_fraud')['amount'].describe().round(2)
axes[1,0].axis('off')
table_data = []
for idx, row in amount_stats.iterrows():
    fraud_type = 'Normal' if idx == 0 else 'Fraud'
    table_data.append([fraud_type] + [f'${x:,.2f}' for x in row.values])

axes[1,0].table(cellText=table_data,
                colLabels=['Type'] + [col.title() for col in amount_stats.columns],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])
axes[1,0].set_title('Amount Statistics by Fraud Status', pad=20)

# Fraud amount vs count scatter
fraud_by_amount = df.groupby(pd.cut(df['amount'], bins=20)).agg({
    'is_fraud': ['count', 'sum', 'mean'],
    'amount': 'mean'
})
fraud_by_amount.columns = ['Total_Count', 'Fraud_Count', 'Fraud_Rate', 'Avg_Amount']
fraud_by_amount = fraud_by_amount.dropna()

axes[1,1].scatter(fraud_by_amount['Avg_Amount'], fraud_by_amount['Fraud_Rate'], 
                  s=fraud_by_amount['Total_Count']/10, alpha=0.6, color='coral')
axes[1,1].set_title('Fraud Rate vs Average Amount\n(Bubble size = Transaction count)')
axes[1,1].set_xlabel('Average Amount ($)')
axes[1,1].set_ylabel('Fraud Rate')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Print amount insights
normal_avg = df[df['is_fraud']==0]['amount'].mean()
fraud_avg = df[df['is_fraud']==1]['amount'].mean()
print(f"💰 AMOUNT INSIGHTS:")
print(f"Average normal transaction: ${normal_avg:.2f}")
print(f"Average fraud transaction: ${fraud_avg:.2f}")
print(f"Fraud premium: {fraud_avg/normal_avg:.1f}x higher")

# %% [markdown]
# ## 4. Transaction Patterns (Temporal Analysis)

# %%
# Convert timestamp and extract temporal features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday
df['day_name'] = df['timestamp'].dt.day_name()
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5

# Temporal analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Fraud rate by hour of day
hourly_stats = df.groupby('hour').agg({
    'is_fraud': ['count', 'sum', 'mean'],
    'amount': 'sum'
}).round(4)
hourly_stats.columns = ['Total_Txns', 'Fraud_Count', 'Fraud_Rate', 'Total_Amount']

axes[0,0].plot(hourly_stats.index, hourly_stats['Fraud_Rate'], 
               marker='o', linewidth=2, markersize=6, color='red')
axes[0,0].set_title('Fraud Rate by Hour of Day', fontweight='bold')
axes[0,0].set_xlabel('Hour of Day')
axes[0,0].set_ylabel('Fraud Rate')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_xticks(range(0, 24, 2))

# Transaction volume by hour
axes[0,1].bar(hourly_stats.index, hourly_stats['Total_Txns'], 
              alpha=0.7, color='lightblue')
axes[0,1].set_title('Transaction Volume by Hour', fontweight='bold')
axes[0,1].set_xlabel('Hour of Day')
axes[0,1].set_ylabel('Number of Transactions')
axes[0,1].set_xticks(range(0, 24, 2))

# Fraud rate by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_stats = df.groupby('day_name')['is_fraud'].agg(['count', 'mean']).reindex(day_order)
daily_stats.columns = ['Total_Txns', 'Fraud_Rate']

bars = axes[1,0].bar(range(len(day_order)), daily_stats['Fraud_Rate'], 
                     color=['lightblue' if day not in ['Saturday', 'Sunday'] else 'lightcoral' 
                            for day in day_order])
axes[1,0].set_title('Fraud Rate by Day of Week', fontweight='bold')
axes[1,0].set_xlabel('Day of Week')
axes[1,0].set_ylabel('Fraud Rate')
axes[1,0].set_xticks(range(len(day_order)))
axes[1,0].set_xticklabels([day[:3] for day in day_order])

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                   f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

# Weekend vs Weekday comparison
weekend_stats = df.groupby('is_weekend').agg({
    'is_fraud': ['count', 'sum', 'mean'],
    'amount': 'mean'
})
weekend_stats.columns = ['Total_Txns', 'Fraud_Count', 'Fraud_Rate', 'Avg_Amount']
weekend_labels = ['Weekday', 'Weekend']

x_pos = [0, 1]
bars = axes[1,1].bar(x_pos, weekend_stats['Fraud_Rate'], 
                     color=['lightblue', 'lightcoral'], width=0.6)
axes[1,1].set_title('Fraud Rate: Weekend vs Weekday', fontweight='bold')
axes[1,1].set_xlabel('Day Type')
axes[1,1].set_ylabel('Fraud Rate')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(weekend_labels)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                   f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Print temporal insights
peak_hour = hourly_stats['Fraud_Rate'].idxmax()
peak_day = daily_stats['Fraud_Rate'].idxmax()

print(f"🕐 TEMPORAL INSIGHTS:")
print(f"Peak fraud hour: {peak_hour:02d}:00 ({hourly_stats.loc[peak_hour, 'Fraud_Rate']:.2%})")
print(f"Peak fraud day: {peak_day} ({daily_stats.loc[peak_day, 'Fraud_Rate']:.2%})")
print(f"Weekend fraud rate: {weekend_stats.loc[True, 'Fraud_Rate']:.2%}")
print(f"Weekday fraud rate: {weekend_stats.loc[False, 'Fraud_Rate']:.2%}")

# %% [markdown]
# ## 5. Geographic Analysis

# %%
# Extract location data for analysis
df['lat'] = df['location'].apply(lambda x: x.get('lat', 0) if isinstance(x, dict) else 0)
df['lon'] = df['location'].apply(lambda x: x.get('lon', 0) if isinstance(x, dict) else 0)
df['city'] = df['location'].apply(lambda x: x.get('city', 'Unknown') if isinstance(x, dict) else 'Unknown')

# Geographic analysis
print("🌍 GEOGRAPHIC DISTRIBUTION:")
city_stats = df.groupby('city').agg({
    'is_fraud': ['count', 'sum', 'mean'],
    'amount': ['sum', 'mean']
}).round(4)
city_stats.columns = ['Total_Txns', 'Fraud_Count', 'Fraud_Rate', 'Total_Amount', 'Avg_Amount']
city_stats = city_stats.sort_values('Fraud_Rate', ascending=False)
print(city_stats.head(10))

# Create interactive map (if plotly is available)
try:
    # Sample data for performance (1000 points)
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    
    fig = px.scatter_mapbox(
        sample_df,
        lat="lat", 
        lon="lon",
        color="is_fraud",
        color_discrete_map={0: 'blue', 1: 'red'},
        hover_data=["amount", "merchant_category", "city"],
        mapbox_style="open-street-map",
        title="Transaction Locations (Sample of 1000)",
        height=600,
        labels={'is_fraud': 'Fraud Status'}
    )
    
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig.show()
except Exception as e:
    print(f"Could not create interactive map: {e}")
    print("Creating static geographic analysis instead...")
    
    # Static geographic analysis
    plt.figure(figsize=(12, 8))
    normal_txns = df[df['is_fraud'] == 0].sample(min(500, len(df[df['is_fraud'] == 0])))
    fraud_txns = df[df['is_fraud'] == 1].sample(min(200, len(df[df['is_fraud'] == 1])))
    
    plt.scatter(normal_txns['lon'], normal_txns['lat'], c='blue', alpha=0.6, s=20, label='Normal')
    plt.scatter(fraud_txns['lon'], fraud_txns['lat'], c='red', alpha=0.8, s=30, label='Fraud')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of Transactions (Sample)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# %% [markdown]
# ## 6. Feature Engineering and Correlation Analysis

# %%
# Feature engineering using our feature engineer
from features.feature_engineer import FeatureEngineer

print("🔧 FEATURE ENGINEERING...")
engineer = FeatureEngineer()

# Create comprehensive feature set
try:
    features_df = engineer.create_features(df)
    print(f"Created {len(features_df.columns)} features from {len(df.columns)} original columns")
    
    # Select numeric features for correlation analysis
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    print(f"Analyzing correlations for {len(numeric_features)} numeric features")
    
    # Calculate correlation with fraud
    target_correlations = features_df[numeric_features].corrwith(df['is_fraud']).abs().sort_values(ascending=False)
    
    # Display top correlations
    print("\n📊 TOP 20 FEATURES CORRELATED WITH FRAUD:")
    for i, (feature, corr) in enumerate(target_correlations.head(20).items(), 1):
        print(f"{i:2d}. {feature:30s} {corr:.4f}")
    
    # Visualize top correlations
    plt.figure(figsize=(12, 8))
    top_features = target_correlations.head(15)
    bars = plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), [f.replace('_', ' ').title() for f in top_features.index])
    plt.xlabel('Absolute Correlation with Fraud')
    plt.title('Top 15 Features Correlated with Fraud', fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Color bars by correlation strength
    for i, bar in enumerate(bars):
        if top_features.values[i] > 0.1:
            bar.set_color('red')
        elif top_features.values[i] > 0.05:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create correlation matrix for top features
    top_feature_names = target_correlations.head(10).index.tolist()
    if 'is_fraud' in features_df.columns:
        top_feature_names.append('is_fraud')
    
    corr_matrix = features_df[top_feature_names].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.3f')
    plt.title('Correlation Matrix - Top Features', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Feature engineering failed: {e}")
    print("Using original features for analysis...")
    features_df = df.select_dtypes(include=[np.number])
    numeric_features = features_df.columns

# %% [markdown]
# ## 7. Model Performance Analysis

# %%
# Train a baseline model for performance analysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("🤖 TRAINING BASELINE MODEL...")

# Prepare features
X = features_df.select_dtypes(include=[np.number]).copy()
y = df['is_fraud'].copy()

# Remove target if it exists in features
if 'is_fraud' in X.columns:
    X = X.drop('is_fraud', axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"Features: {X_train.shape[1]:,}")

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Generate predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n📈 MODEL PERFORMANCE:")
print(classification_report(y_test, y_pred))

# %%
# Create comprehensive performance visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

axes[0,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0,0].set_xlim([0.0, 1.0])
axes[0,0].set_ylim([0.0, 1.05])
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].set_title('ROC Curve', fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

axes[0,1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
axes[0,1].set_xlim([0.0, 1.0])
axes[0,1].set_ylim([0.0, 1.05])
axes[0,1].set_xlabel('Recall')
axes[0,1].set_ylabel('Precision')
axes[0,1].set_title('Precision-Recall Curve', fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# 3. Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_15_features = feature_importance.head(15)

axes[1,0].barh(range(len(top_15_features)), top_15_features.values, color='lightblue')
axes[1,0].set_yticks(range(len(top_15_features)))
axes[1,0].set_yticklabels([f.replace('_', ' ').title() for f in top_15_features.index])
axes[1,0].set_xlabel('Feature Importance')
axes[1,0].set_title('Top 15 Feature Importances', fontweight='bold')
axes[1,0].invert_yaxis()
axes[1,0].grid(axis='x', alpha=0.3)

# 4. Prediction Distribution
axes[1,1].hist(y_pred_proba[y_test==0], bins=50, alpha=0.7, label='Normal', 
               color='lightblue', density=True)
axes[1,1].hist(y_pred_proba[y_test==1], bins=50, alpha=0.7, label='Fraud', 
               color='salmon', density=True)
axes[1,1].axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
axes[1,1].set_xlabel('Predicted Probability')
axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Prediction Distribution', fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Model performance summary
precision_val = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)
f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

print(f"\n🎯 MODEL PERFORMANCE SUMMARY:")
print(f"ROC AUC:     {roc_auc:.4f}")
print(f"PR AUC:      {pr_auc:.4f}")
print(f"Precision:   {precision_val:.4f}")
print(f"Recall:      {recall_val:.4f}")
print(f"F1 Score:    {f1_val:.4f}")

# %% [markdown]
# ## 8. Business Impact Analysis

# %%
# Calculate business impact metrics
total_transactions = len(df)
total_fraud_amount = df[df['is_fraud'] == 1]['amount'].sum()
total_normal_amount = df[df['is_fraud'] == 0]['amount'].sum()
avg_fraud_amount = df[df['is_fraud'] == 1]['amount'].mean()
avg_normal_amount = df[df['is_fraud'] == 0]['amount'].mean()

# Model performance on test set
if len(y_test) > 0:
    test_df = X_test.copy()
    test_df['actual_fraud'] = y_test
    test_df['predicted_fraud'] = y_pred
    test_df['fraud_probability'] = y_pred_proba
    
    # Merge with original amounts for test set
    test_indices = y_test.index
    test_amounts = df.loc[test_indices, 'amount']
    
    # Calculate confusion matrix components
    true_positives = sum((y_test == 1) & (y_pred == 1))
    false_positives = sum((y_test == 0) & (y_pred == 1))
    true_negatives = sum((y_test == 0) & (y_pred == 0))
    false_negatives = sum((y_test == 1) & (y_pred == 0))
    
    # Business impact calculations
    detected_fraud_amount = test_amounts[(y_test == 1) & (y_pred == 1)].sum()
    missed_fraud_amount = test_amounts[(y_test == 1) & (y_pred == 0)].sum()
    false_alarm_amount = test_amounts[(y_test == 0) & (y_pred == 1)].sum()
    
    # Create business impact visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Financial Impact
    categories = ['Fraud Detected', 'Fraud Missed', 'False Alarms']
    amounts = [detected_fraud_amount, missed_fraud_amount, false_alarm_amount]
    colors = ['green', 'red', 'orange']
    
    bars = axes[0,0].bar(categories, amounts, color=colors, alpha=0.7)
    axes[0,0].set_title('Financial Impact ($)', fontweight='bold')
    axes[0,0].set_ylabel('Amount ($)')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confusion Matrix Visualization
    conf_matrix = [[true_negatives, false_positives], [false_negatives, true_positives]]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
                xticklabels=['Predicted Normal', 'Predicted Fraud'],
                yticklabels=['Actually Normal', 'Actually Fraud'])
    axes[0,1].set_title('Confusion Matrix', fontweight='bold')
    
    # 3. Threshold Analysis
    thresholds = np.arange(0.1, 1.0, 0.05)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
    
    axes[1,0].plot(thresholds, precision_scores, label='Precision', marker='o')
    axes[1,0].plot(thresholds, recall_scores, label='Recall', marker='s')
    axes[1,0].plot(thresholds, f1_scores, label='F1 Score', marker='^')
    axes[1,0].set_xlabel('Threshold')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Threshold Analysis', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # 4. Cost-Benefit Analysis
    # Assume: Cost of investigating false positive = $10
    # Benefit of catching fraud = Average fraud amount
    # Cost of missing fraud = Average fraud amount
    
    investigation_cost = 10  # Cost per false positive investigation
    fraud_benefit = avg_fraud_amount  # Benefit of catching fraud
    fraud_cost = avg_fraud_amount     # Cost of missing fraud
    
    total_benefit = true_positives * fraud_benefit
    investigation_costs = false_positives * investigation_cost
    missed_fraud_costs = false_negatives * fraud_cost
    net_benefit = total_benefit - investigation_costs - missed_fraud_costs
    
    cost_benefit_data = {
        'Fraud Prevented': total_benefit,
        'Investigation Costs': -investigation_costs,
        'Missed Fraud Costs': -missed_fraud_costs,
        'Net Benefit': net_benefit
    }
    
    colors_cb = ['green', 'orange', 'red', 'blue']
    bars = axes[1,1].bar(range(len(cost_benefit_data)), list(cost_benefit_data.values()), 
                         color=colors_cb, alpha=0.7)
    axes[1,1].set_xticks(range(len(cost_benefit_data)))
    axes[1,1].set_xticklabels(list(cost_benefit_data.keys()), rotation=45)
    axes[1,1].set_title('Cost-Benefit Analysis ($)', fontweight='bold')
    axes[1,1].set_ylabel('Amount ($)')
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., 
                       height + (abs(height)*0.02 if height > 0 else -abs(height)*0.02),
                       f'${height:,.0f}', ha='center', 
                       va='bottom' if height > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

print("💰 BUSINESS IMPACT ANALYSIS:")
print("=" * 50)
print(f"Total Dataset Metrics:")
print(f"  Total transactions: {total_transactions:,}")
print(f"  Total fraud amount: ${total_fraud_amount:,.2f}")
print(f"  Average fraud amount: ${avg_fraud_amount:.2f}")
print(f"  Average normal amount: ${avg_normal_amount:.2f}")
print(f"  Fraud premium: {avg_fraud_amount/avg_normal_amount:.1f}x higher")

if len(y_test) > 0:
    fraud_detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    
    print(f"\nModel Performance on Test Set:")
    print(f"  Fraud detection rate: {fraud_detection_rate:.1%}")
    print(f"  False positive rate: {false_positive_rate:.1%}")
    print(f"  Fraud caught: {true_positives} out of {true_positives + false_negatives}")
    print(f"  False alarms: {false_positives}")
    
    print(f"\nFinancial Impact (Test Set):")
    print(f"  Fraud prevented: ${detected_fraud_amount:,.2f}")
    print(f"  Fraud missed: ${missed_fraud_amount:,.2f}")
    print(f"  False alarm amount: ${false_alarm_amount:,.2f}")
    
    savings_rate = detected_fraud_amount / (detected_fraud_amount + missed_fraud_amount) if (detected_fraud_amount + missed_fraud_amount) > 0 else 0
    print(f"  Fraud savings rate: {savings_rate:.1%}")
    
    print(f"\nCost-Benefit Analysis:")
    print(f"  Benefit from caught fraud: ${total_benefit:,.2f}")
    print(f"  Investigation costs: ${investigation_costs:,.2f}")
    print(f"  Missed fraud costs: ${missed_fraud_costs:,.2f}")
    print(f"  Net benefit: ${net_benefit:,.2f}")
    
    if net_benefit > 0:
        print(f"  ✅ Model is profitable!")
    else:
        print(f"  ❌ Model needs optimization")

# %%
# Key Insights and Recommendations Summary
print("\n" + "="*60)
print("🎯 KEY INSIGHTS & RECOMMENDATIONS")
print("="*60)

print("\n📊 KEY FINDINGS:")
print("• Fraud Rate:", f"{df['is_fraud'].mean():.2%}")
print("• High-Risk Categories:", ", ".join(category_analysis.head(3).index.tolist()))
print("• Peak Fraud Time:", f"{peak_hour:02d}:00 hours" if 'peak_hour' in locals() else "Analysis pending")
print("• Average Fraud Amount:", f"${avg_fraud_amount:.2f} ({avg_fraud_amount/avg_normal_amount:.1f}x normal)")

if 'roc_auc' in locals():
    print(f"• Model Performance: {roc_auc:.1%} ROC AUC, {precision_val:.1%} Precision")

print("\n🚀 RECOMMENDATIONS:")
print("\n1. FEATURE ENHANCEMENT:")
print("   • Add velocity features (transactions per hour/day)")
print("   • Include network analysis (user-merchant relationships)")
print("   • Enhance location-based risk scoring")
print("   • Add device fingerprinting features")

print("\n2. MODEL IMPROVEMENTS:")
print("   • Implement ensemble methods (RF + XGBoost + LightGBM)")
print("   • Optimize threshold based on business cost-benefit")
print("   • Set up automated retraining pipeline")
print("   • Add real-time drift detection")

print("\n3. OPERATIONAL RECOMMENDATIONS:")
print("   • Deploy real-time monitoring dashboard")
print("   • Implement tiered alert system")
print("   • Create feedback loop for false positives")
print("   • Set up A/B testing framework")

print("\n4. BUSINESS RULES:")
print("   • Higher scrutiny for transactions >$1000 during night hours")
print("   • Enhanced verification for gambling/cash advance categories")
print("   • Location-based risk adjustments")
print("   • User behavior baselines and anomaly detection")

if 'net_benefit' in locals() and net_benefit > 0:
    roi_percentage = (net_benefit / max(investigation_costs + missed_fraud_costs, 1)) * 100
    print(f"\n💡 BUSINESS IMPACT:")
    print(f"   • Estimated ROI: {roi_percentage:.0f}%")
    print(f"   • Monthly savings potential: ${net_benefit * 30:,.2f}")
    print(f"   • Model is delivering positive business value!")

# %%
# Save analysis results and artifacts
print("\n💾 SAVING ANALYSIS RESULTS...")

# Create reports directory
import os
os.makedirs('../reports', exist_ok=True)

# Save analysis summary
analysis_summary = {
    'dataset_overview': {
        'total_transactions': int(total_transactions),
        'fraud_rate': float(df['is_fraud'].mean()),
        'total_fraud_amount': float(total_fraud_amount),
        'avg_fraud_amount': float(avg_fraud_amount),
        'avg_normal_amount': float(avg_normal_amount)
    },
    'temporal_patterns': {
        'peak_fraud_hour': int(peak_hour) if 'peak_hour' in locals() else None,
        'weekend_fraud_rate': float(weekend_stats.loc[True, 'Fraud_Rate']) if 'weekend_stats' in locals() else None,
        'weekday_fraud_rate': float(weekend_stats.loc[False, 'Fraud_Rate']) if 'weekend_stats' in locals() else None
    }
}

if 'roc_auc' in locals():
    analysis_summary['model_performance'] = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1_val)
    }

if 'net_benefit' in locals():
    analysis_summary['business_impact'] = {
        'fraud_prevented_amount': float(detected_fraud_amount),
        'fraud_missed_amount': float(missed_fraud_amount),
        'net_benefit': float(net_benefit),
        'investigation_costs': float(investigation_costs)
    }

# Save to JSON
import json
with open('../reports/eda_summary.json', 'w') as f:
    json.dump(analysis_summary, f, indent=2, default=str)

# Save feature importance if available
if 'feature_importance' in locals():
    feature_importance.to_csv('../reports/feature_importance.csv', header=['importance'])
    print("• Feature importance saved to reports/feature_importance.csv")

# Save merchant category analysis
category_analysis.to_csv('../reports/merchant_category_analysis.csv')
print("• Merchant analysis saved to reports/merchant_category_analysis.csv")

# Save temporal analysis
if 'hourly_stats' in locals():
    hourly_stats.to_csv('../reports/hourly_fraud_analysis.csv')
    print("• Hourly analysis saved to reports/hourly_fraud_analysis.csv")

print("• Analysis summary saved to reports/eda_summary.json")
print("\n✅ EXPLORATORY DATA ANALYSIS COMPLETED!")
print("🎉 All insights and recommendations are ready for implementation.")

# %%
# Final visualization: Executive Summary Dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])

# Title
fig.suptitle('FRAUD DETECTION SYSTEM - EXECUTIVE SUMMARY', fontsize=20, fontweight='bold', y=0.98)

# 1. Fraud Rate KPI
ax1 = fig.add_subplot(gs[0, 0])
fraud_rate_pct = df['is_fraud'].mean() * 100
ax1.text(0.5, 0.5, f'{fraud_rate_pct:.2f}%', ha='center', va='center', 
         fontsize=32, fontweight='bold', color='red')
ax1.text(0.5, 0.2, 'Fraud Rate', ha='center', va='center', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# 2. Total Loss KPI
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.5, f'${total_fraud_amount/1000:.0f}K', ha='center', va='center',
         fontsize=32, fontweight='bold', color='red')
ax2.text(0.5, 0.2, 'Total Fraud Amount', ha='center', va='center', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# 3. Model Performance KPI
ax3 = fig.add_subplot(gs[0, 2])
if 'roc_auc' in locals():
    ax3.text(0.5, 0.5, f'{roc_auc:.1%}', ha='center', va='center',
             fontsize=32, fontweight='bold', color='green')
    ax3.text(0.5, 0.2, 'Model ROC AUC', ha='center', va='center', fontsize=14, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'N/A', ha='center', va='center',
             fontsize=32, fontweight='bold', color='gray')
    ax3.text(0.5, 0.2, 'Model ROC AUC', ha='center', va='center', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# 4. Savings KPI
ax4 = fig.add_subplot(gs[0, 3])
if 'net_benefit' in locals():
    ax4.text(0.5, 0.5, f'${net_benefit/1000:.0f}K', ha='center', va='center',
             fontsize=32, fontweight='bold', color='green' if net_benefit > 0 else 'red')
    ax4.text(0.5, 0.2, 'Net Benefit', ha='center', va='center', fontsize=14, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'TBD', ha='center', va='center',
             fontsize=32, fontweight='bold', color='gray')
    ax4.text(0.5, 0.2, 'Net Benefit', ha='center', va='center', fontsize=14, fontweight='bold')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# 5. Fraud by Category (Top 5)
ax5 = fig.add_subplot(gs[1, :2])
top_5_categories = category_analysis.head(5)
bars = ax5.bar(range(len(top_5_categories)), top_5_categories['Fraud_Rate'], 
               color=['red' if x > 0.05 else 'orange' if x > 0.02 else 'lightblue' 
                      for x in top_5_categories['Fraud_Rate']])
ax5.set_xticks(range(len(top_5_categories)))
ax5.set_xticklabels([cat.title().replace('_', ' ') for cat in top_5_categories.index], rotation=45)
ax5.set_ylabel('Fraud Rate')
ax5.set_title('Fraud Rate by Top 5 Categories', fontweight='bold')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

# 6. Hourly Pattern
ax6 = fig.add_subplot(gs[1, 2:])
if 'hourly_stats' in locals():
    ax6.plot(hourly_stats.index, hourly_stats['Fraud_Rate'], marker='o', linewidth=2, markersize=4)
    ax6.set_xlabel('Hour of Day')
    ax6.set_ylabel('Fraud Rate')
    ax6.set_title('Fraud Rate by Hour', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xticks(range(0, 24, 4))

# 7. Geographic Distribution (if available)
ax7 = fig.add_subplot(gs[2, :2])
if 'city_stats' in locals() and len(city_stats) > 0:
    top_cities = city_stats.head(8)
    bars = ax7.barh(range(len(top_cities)), top_cities['Fraud_Rate'])
    ax7.set_yticks(range(len(top_cities)))
    ax7.set_yticklabels(top_cities.index)
    ax7.set_xlabel('Fraud Rate')
    ax7.set_title('Fraud Rate by City (Top 8)', fontweight='bold')
    ax7.invert_yaxis()

# 8. Model Performance Summary
ax8 = fig.add_subplot(gs[2, 2:])
if 'roc_auc' in locals():
    metrics_names = ['ROC AUC', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [roc_auc, precision_val, recall_val, f1_val]
    
    bars = ax8.bar(metrics_names, metrics_values, 
                   color=['green' if x >= 0.8 else 'orange' if x >= 0.6 else 'red' for x in metrics_values])
    ax8.set_ylim(0, 1)
    ax8.set_ylabel('Score')
    ax8.set_title('Model Performance Metrics', fontweight='bold')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
else:
    ax8.text(0.5, 0.5, 'Model Performance\nNot Available', ha='center', va='center',
             fontsize=14, fontweight='bold')
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('../reports/executive_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Executive summary dashboard saved to reports/executive_summary_dashboard.png")
print("\n🎯 ANALYSIS COMPLETE - Ready for production deployment!")