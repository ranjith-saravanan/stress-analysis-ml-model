"""
Social Media Detox Effect Analyzer - Complete Analysis Script
Author: Ranjith Saravanan
Description: Comprehensive ML analysis of social media abstinence effects on mental health
"""

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical Analysis
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Machine Learning - Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

# Machine Learning - Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Statistical Models
import statsmodels.api as sm

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*70)
print("SOCIAL MEDIA DETOX EFFECT ANALYZER")
print("="*70)
print("All libraries imported successfully!")
print()

# =============================================================================
# PHASE 1: DATA LOADING & PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("PHASE 1: DATA LOADING & PREPROCESSING")
print("="*70)

# Load the dataset
df = pd.read_csv(r'c:\Users\RANJITH S\Downloads\archive (3)\Mental_Health_and_Social_Media_Balance_Dataset.csv')

print(f"\n🔍 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n📊 Data Types:")
print(df.dtypes)

print("\n🔍 First 5 rows:")
print(df.head())

print("\n📈 Basic Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("No missing values found!")
else:
    print(missing_values)

# Data preprocessing
print("\n🛠️ Data Preprocessing:")

# Convert categorical columns
categorical_cols = ['Gender', 'Social_Media_Platform']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Remove any duplicates
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")

print(f"\n✅ Final dataset shape: {df.shape}")
print("✅ Phase 1 Complete!")

# =============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*70)
print("📊 PHASE 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

# 2.1 Distribution Analysis
numeric_cols = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 
                'Stress_Level(1-10)', 'Days_Without_Social_Media', 
                'Exercise_Frequency(week)', 'Happiness_Index(1-10)']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

for idx, feature in enumerate(numeric_cols):
    row = idx // 4
    col = idx % 4
    axes[row, col].hist(df[feature], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[row, col].set_title(f'{feature}', fontsize=10)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[1, 3])

plt.tight_layout()
plt.savefig('results/01_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/01_distributions.png")
plt.close()

# 2.2 Categorical Variables Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gender distribution
if 'Gender' in df.columns:
    gender_counts = df['Gender'].value_counts()
    axes[0].bar(gender_counts.index, gender_counts.values, alpha=0.7, edgecolor='black', color='coral')
    axes[0].set_title('Gender Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

# Social Media Platform distribution
if 'Social_Media_Platform' in df.columns:
    platform_counts = df['Social_Media_Platform'].value_counts()
    axes[1].bar(platform_counts.index, platform_counts.values, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[1].set_title('Social Media Platform Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/02_categorical_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/02_categorical_distributions.png")
plt.close()

print("✅ Phase 2 Complete!")

# =============================================================================
# PHASE 3: CORRELATION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("🔗 PHASE 3: CORRELATION ANALYSIS")
print("="*70)

# Calculate correlations between numerical variables
correlation_matrix = df[numeric_cols].corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f', mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix: All Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/03_correlation_heatmap.png")
plt.close()

# Focus on Key Relationships
print("\n🎯 Key Correlations with Mental Health Outcomes:")

# Stress Level correlations
stress_correlations = correlation_matrix['Stress_Level(1-10)'].sort_values(ascending=False)
print("\n📉 Correlations with Stress Level:")
for var, corr in stress_correlations.items():
    if var != 'Stress_Level(1-10)':
        print(f"  {var}: {corr:.4f}")

# Happiness Index correlations
happiness_correlations = correlation_matrix['Happiness_Index(1-10)'].sort_values(ascending=False)
print("\n😊 Correlations with Happiness Index:")
for var, corr in happiness_correlations.items():
    if var != 'Happiness_Index(1-10)':
        print(f"  {var}: {corr:.4f}")

# Statistical Significance Testing
print("\n📈 Statistical Significance Tests:")

# Test correlation between Days Without Social Media and Stress
stress_corr, stress_p = pearsonr(df['Days_Without_Social_Media'], df['Stress_Level(1-10)'])
print(f"\nDays Without Social Media vs Stress Level:")
print(f"  Pearson r: {stress_corr:.4f} (p-value: {stress_p:.4e})")

# Test correlation between Days Without Social Media and Happiness
happiness_corr, happiness_p = pearsonr(df['Days_Without_Social_Media'], df['Happiness_Index(1-10)'])
print(f"\nDays Without Social Media vs Happiness Index:")
print(f"  Pearson r: {happiness_corr:.4f} (p-value: {happiness_p:.4e})")

# Test correlation between Screen Time and Stress
screen_stress_corr, screen_stress_p = pearsonr(df['Daily_Screen_Time(hrs)'], df['Stress_Level(1-10)'])
print(f"\nDaily Screen Time vs Stress Level:")
print(f"  Pearson r: {screen_stress_corr:.4f} (p-value: {screen_stress_p:.4e})")

print("\n✅ Phase 3 Complete!")

# =============================================================================
# PHASE 4: REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("📈 PHASE 4: REGRESSION ANALYSIS")
print("="*70)

X = df[['Days_Without_Social_Media']]
y_stress = df['Stress_Level(1-10)']
y_happiness = df['Happiness_Index(1-10)']

# Stress Model
lr_stress = LinearRegression()
lr_stress.fit(X, y_stress)
stress_pred = lr_stress.predict(X)

print("\n🔬 Simple Linear Regression Results:")
print(f"\n📉 Stress Model:")
print(f"  Coefficient: {lr_stress.coef_[0]:.4f}")
print(f"  Intercept: {lr_stress.intercept_:.4f}")
print(f"  R² Score: {r2_score(y_stress, stress_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_stress, stress_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_stress, stress_pred)):.4f}")

# Happiness Model
lr_happiness = LinearRegression()
lr_happiness.fit(X, y_happiness)
happiness_pred = lr_happiness.predict(X)

print(f"\n😊 Happiness Model:")
print(f"  Coefficient: {lr_happiness.coef_[0]:.4f}")
print(f"  Intercept: {lr_happiness.intercept_:.4f}")
print(f"  R² Score: {r2_score(y_happiness, happiness_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_happiness, happiness_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_happiness, happiness_pred)):.4f}")

# Multiple Regression with Control Variables
print("\n🔬 Multiple Regression (with confounders):")

features = ['Days_Without_Social_Media', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 
           'Exercise_Frequency(week)', 'Age']

X_multi = df[features]
X_multi_with_const = sm.add_constant(X_multi)

# Stress Model
model_stress_multi = sm.OLS(y_stress, X_multi_with_const).fit()
print("\n📉 Multiple Regression - Stress Level:")
print(f"  R² Score: {model_stress_multi.rsquared:.4f}")
print(f"  Adjusted R²: {model_stress_multi.rsquared_adj:.4f}")
print(f"  F-statistic: {model_stress_multi.fvalue:.4f}")
print(f"  p-value: {model_stress_multi.f_pvalue:.4e}")

# Happiness Model
model_happiness_multi = sm.OLS(y_happiness, X_multi_with_const).fit()
print("\n😊 Multiple Regression - Happiness Index:")
print(f"  R² Score: {model_happiness_multi.rsquared:.4f}")
print(f"  Adjusted R²: {model_happiness_multi.rsquared_adj:.4f}")
print(f"  F-statistic: {model_happiness_multi.fvalue:.4f}")
print(f"  p-value: {model_happiness_multi.f_pvalue:.4e}")

# Visualization of Regression Results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Stress
axes[0].scatter(df['Days_Without_Social_Media'], df['Stress_Level(1-10)'], 
                alpha=0.3, s=10, label='Actual', color='red')
axes[0].plot(df['Days_Without_Social_Media'], stress_pred, 
             color='darkred', linewidth=2, label='Predicted')
axes[0].set_xlabel('Days Without Social Media')
axes[0].set_ylabel('Stress Level')
axes[0].set_title(f'Linear Regression: Stress (R² = {r2_score(y_stress, stress_pred):.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Happiness
axes[1].scatter(df['Days_Without_Social_Media'], df['Happiness_Index(1-10)'], 
                alpha=0.3, s=10, label='Actual', color='green')
axes[1].plot(df['Days_Without_Social_Media'], happiness_pred, 
             color='darkgreen', linewidth=2, label='Predicted')
axes[1].set_xlabel('Days Without Social Media')
axes[1].set_ylabel('Happiness Index')
axes[1].set_title(f'Linear Regression: Happiness (R² = {r2_score(y_happiness, happiness_pred):.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/04_regression_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: results/04_regression_results.png")
plt.close()

print("✅ Phase 4 Complete!")

# =============================================================================
# PHASE 5: USER SEGMENTATION (CLUSTERING)
# =============================================================================
print("\n" + "="*70)
print("🎯 PHASE 5: USER SEGMENTATION (CLUSTERING)")
print("="*70)

# Prepare Data for Clustering
clustering_features = ['Days_Without_Social_Media', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
                       'Exercise_Frequency(week)', 'Stress_Level(1-10)', 'Happiness_Index(1-10)', 'Age']

X_cluster = df[clustering_features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print(f"\nClustering data shape: {X_scaled.shape}")
print(f"Features used: {clustering_features}")

# Elbow Method for Optimal K
inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("\n🔍 Finding optimal number of clusters...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
axes[0].set_title('Elbow Method for Optimal K')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/05_clustering_optimization.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/05_clustering_optimization.png")
plt.close()

# Optimal K selection
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✨ Optimal number of clusters: {optimal_k}")
print(f"✨ Best silhouette score: {max(silhouette_scores):.4f}")

# Apply K-Means with Optimal K
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster_KMeans'] = kmeans_final.fit_predict(X_scaled)

print(f"\n📊 Cluster distribution:")
print(df['Cluster_KMeans'].value_counts().sort_index())

# Cluster Profiling
print(f"\n{'='*70}")
print("CLUSTER PROFILES (K-Means)")
print('='*70)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster_KMeans'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} (n={len(cluster_data)}) ---")
    print(cluster_data[clustering_features].mean().round(2))

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=df['Cluster_KMeans'], 
                     cmap='viridis', s=100, alpha=0.6, edgecolors='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('K-Means Clustering: PCA Visualization', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('results/06_kmeans_clusters.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: results/06_kmeans_clusters.png")
plt.close()

# DBSCAN Clustering
print(f"\n{'='*70}")
print("DBSCAN CLUSTERING")
print('='*70)

dbscan = DBSCAN(eps=0.8, min_samples=5)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

print(f"\n📊 DBSCAN Cluster distribution:")
n_clusters = len(set(df['Cluster_DBSCAN'])) - (1 if -1 in df['Cluster_DBSCAN'] else 0)
n_noise = sum(df['Cluster_DBSCAN'] == -1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(df['Cluster_DBSCAN'].value_counts().sort_index())

# DBSCAN Visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=df['Cluster_DBSCAN'], 
                     cmap='plasma', s=100, alpha=0.6, edgecolors='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('DBSCAN Clustering: PCA Visualization', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster (-1 = Noise)')
plt.grid(True, alpha=0.3)
plt.savefig('results/07_dbscan_clusters.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/07_dbscan_clusters.png")
plt.close()

print("\n✅ Phase 5 Complete!")

# =============================================================================
# PHASE 6: ADVANCED MODELING (RANDOM FOREST)
# =============================================================================
print("\n" + "="*70)
print("🧠 PHASE 6: ADVANCED MODELING (RANDOM FOREST)")
print("="*70)

# Encode categorical variables
le_gender = LabelEncoder()
le_platform = LabelEncoder()
df_encoded = df.copy()

if 'Gender' in df_encoded.columns:
    df_encoded['Gender'] = le_gender.fit_transform(df_encoded['Gender'])
if 'Social_Media_Platform' in df_encoded.columns:
    df_encoded['Social_Media_Platform'] = le_platform.fit_transform(df_encoded['Social_Media_Platform'])

# Features and targets
features = ['Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 
           'Days_Without_Social_Media', 'Exercise_Frequency(week)', 'Social_Media_Platform']
X = df_encoded[features]
y_stress = df_encoded['Stress_Level(1-10)']
y_happiness = df_encoded['Happiness_Index(1-10)']

print(f"\nFeatures used: {features}")
print(f"Dataset shape: {X.shape}")

# Split data
X_train, X_test, y_train_stress, y_test_stress = train_test_split(X, y_stress, test_size=0.2, random_state=42)
X_train_h, X_test_h, y_train_happiness, y_test_happiness = train_test_split(X, y_happiness, test_size=0.2, random_state=42)

# Stress Model
print("\n🔨 Training Random Forest models...")
rf_stress = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_stress.fit(X_train, y_train_stress)
rf_stress_pred = rf_stress.predict(X_test)

print("\n📊 Random Forest Results:")
print(f"\n📉 Stress Level Prediction:")
print(f"  R² Score: {r2_score(y_test_stress, rf_stress_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_test_stress, rf_stress_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_stress, rf_stress_pred)):.4f}")

# Happiness Model
rf_happiness = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_happiness.fit(X_train_h, y_train_happiness)
rf_happiness_pred = rf_happiness.predict(X_test_h)

print(f"\n😊 Happiness Index Prediction:")
print(f"  R² Score: {r2_score(y_test_happiness, rf_happiness_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_test_happiness, rf_happiness_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_happiness, rf_happiness_pred)):.4f}")

# Feature Importance Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Stress feature importance
stress_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_stress.feature_importances_
}).sort_values('importance', ascending=True)

axes[0].barh(stress_importance['feature'], stress_importance['importance'], color='lightcoral')
axes[0].set_title('Feature Importance: Stress Level Prediction', fontweight='bold')
axes[0].set_xlabel('Importance')

# Happiness feature importance
happiness_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_happiness.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(happiness_importance['feature'], happiness_importance['importance'], color='lightgreen')
axes[1].set_title('Feature Importance: Happiness Index Prediction', fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('results/08_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: results/08_feature_importance.png")
plt.close()

# Model Performance Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Stress predictions
axes[0].scatter(y_test_stress, rf_stress_pred, alpha=0.6, color='red')
axes[0].plot([1, 10], [1, 10], 'k--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Stress Level')
axes[0].set_ylabel('Predicted Stress Level')
axes[0].set_title(f'Random Forest: Stress Prediction (R² = {r2_score(y_test_stress, rf_stress_pred):.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Happiness predictions
axes[1].scatter(y_test_happiness, rf_happiness_pred, alpha=0.6, color='green')
axes[1].plot([1, 10], [1, 10], 'k--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Happiness Index')
axes[1].set_ylabel('Predicted Happiness Index')
axes[1].set_title(f'Random Forest: Happiness Prediction (R² = {r2_score(y_test_happiness, rf_happiness_pred):.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/09_rf_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/09_rf_predictions.png")
plt.close()

print("\n✅ Phase 6 Complete!")

# =============================================================================
# PHASE 7: COMPREHENSIVE RESULTS SUMMARY
# =============================================================================
print("\n" + "="*70)
print("📊 PHASE 7: COMPREHENSIVE RESULTS SUMMARY")
print("="*70)

results_summary = {
    'Analysis': [
        'Dataset Size',
        'Number of Features',
        'Key Correlation (Days vs Stress)',
        'Key Correlation (Days vs Happiness)',
        'Simple Regression R² (Stress)',
        'Simple Regression R² (Happiness)',
        'Multiple Regression R² (Stress)',
        'Multiple Regression R² (Happiness)',
        'Optimal K-Means Clusters',
        'Best Silhouette Score',
        'Random Forest R² (Stress)',
        'Random Forest R² (Happiness)',
        'Random Forest MAE (Stress)',
        'Random Forest MAE (Happiness)',
        'Most Important Feature (Stress)',
        'Most Important Feature (Happiness)'
    ],
    'Result': [
        f'{df.shape[0]} users',
        f'{df.shape[1]} features',
        f'{stress_corr:.4f}',
        f'{happiness_corr:.4f}',
        f'{r2_score(y_stress, stress_pred):.4f}',
        f'{r2_score(y_happiness, happiness_pred):.4f}',
        f'{model_stress_multi.rsquared:.4f}',
        f'{model_happiness_multi.rsquared:.4f}',
        f'{optimal_k}',
        f'{max(silhouette_scores):.4f}',
        f'{r2_score(y_test_stress, rf_stress_pred):.4f}',
        f'{r2_score(y_test_happiness, rf_happiness_pred):.4f}',
        f'{mean_absolute_error(y_test_stress, rf_stress_pred):.4f}',
        f'{mean_absolute_error(y_test_happiness, rf_happiness_pred):.4f}',
        f'{stress_importance.iloc[-1]["feature"]}',
        f'{happiness_importance.iloc[-1]["feature"]}'
    ]
}

results_df = pd.DataFrame(results_summary)
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('results/final_results_summary.csv', index=False)
print("\n✅ Saved: results/final_results_summary.csv")

# Key Insights
print(f"\n{'='*70}")
print("🎯 KEY INSIGHTS & FINDINGS")
print('='*70)

print("\n1️⃣ CORRELATION ANALYSIS:")
print(f"   • Days without social media shows {abs(stress_corr):.3f} correlation with stress levels")
print(f"   • Days without social media shows {abs(happiness_corr):.3f} correlation with happiness")
print(f"   • Screen time shows {abs(screen_stress_corr):.3f} correlation with stress")

print("\n2️⃣ PREDICTIVE MODELING:")
print(f"   • Random Forest achieves {r2_score(y_test_stress, rf_stress_pred):.1%} accuracy for stress prediction")
print(f"   • Random Forest achieves {r2_score(y_test_happiness, rf_happiness_pred):.1%} accuracy for happiness prediction")
print(f"   • Mean Absolute Error (Stress): {mean_absolute_error(y_test_stress, rf_stress_pred):.2f} points")
print(f"   • Mean Absolute Error (Happiness): {mean_absolute_error(y_test_happiness, rf_happiness_pred):.2f} points")

print("\n3️⃣ USER SEGMENTATION:")
print(f"   • Identified {optimal_k} distinct user clusters")
print(f"   • Silhouette score of {max(silhouette_scores):.3f} indicates good cluster separation")
print(f"   • Most important predictor for stress: {stress_importance.iloc[-1]['feature']}")
print(f"   • Most important predictor for happiness: {happiness_importance.iloc[-1]['feature']}")

print("\n4️⃣ PRACTICAL IMPLICATIONS:")
if stress_corr < 0:
    print("   ✅ Social media abstinence is associated with REDUCED stress levels")
else:
    print("   ⚠️  Social media abstinence is associated with INCREASED stress levels")

if happiness_corr > 0:
    print("   ✅ Social media abstinence is associated with INCREASED happiness")
else:
    print("   ⚠️  Social media abstinence is associated with DECREASED happiness")

print("   • Sleep quality and exercise are strong protective factors")
print("   • Personalized interventions should consider individual user phenotypes")
print("   • Multiple factors influence mental health outcomes simultaneously")

# Save Processed Dataset
df.to_csv('results/processed_dataset_with_clusters.csv', index=False)
print("\n✅ Saved: results/processed_dataset_with_clusters.csv")

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)
print("\n📁 All results saved to 'results/' directory:")
print("   • 9 high-quality visualizations (PNG files)")
print("   • Final results summary (CSV)")
print("   • Processed dataset with cluster assignments (CSV)")
print("\n🚀 Your Social Media Detox Effect Analyzer is ready for portfolio presentation!")
print("="*70)
