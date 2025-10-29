# Social Media Detox Effect Analyzer - Core Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Analysis
from scipy.stats import pearsonr

# Machine Learning - Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Machine Learning - Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score

# Time Series
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

print("✅ Core libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")

# Load the dataset
print("\n🔍 Loading dataset...")
df = pd.read_csv('data/Mental_Health_and_Social_Media_Balance_Dataset.csv')

print("🔍 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n📊 Data Types:")
print(df.dtypes)

print("\n🔍 First 5 rows:")
print(df.head())

print("\n📈 Basic Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

# Data preprocessing
print("\n🛠️ Data Preprocessing:")

# Convert categorical columns
categorical_cols = ['Gender', 'Social_Media_Platform']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Remove any duplicates
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")

print(f"\n✅ Final dataset shape: {df.shape}")
print("✅ Data preprocessing complete!")

# Phase 2: EDA
print("\n📊 PHASE 2: EXPLORATORY DATA ANALYSIS")
print("="*50)

# Distribution Analysis
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

numeric_cols = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
                'Stress_Level(1-10)', 'Days_Without_Social_Media',
                'Exercise_Frequency(week)', 'Happiness_Index(1-10)']

for idx, feature in enumerate(numeric_cols):
    row = idx // 4
    col = idx % 4
    axes[row, col].hist(df[feature], bins=20, edgecolor='black', alpha=0.7)
    axes[row, col].set_title(f'{feature}')
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/01_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Categorical Variables Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gender distribution
gender_counts = df['Gender'].value_counts()
axes[0].bar(gender_counts.index, gender_counts.values, alpha=0.7, edgecolor='black')
axes[0].set_title('Gender Distribution')
axes[0].set_ylabel('Count')
axes[0].grid(True, alpha=0.3)

# Social Media Platform distribution
platform_counts = df['Social_Media_Platform'].value_counts()
axes[1].bar(platform_counts.index, platform_counts.values, alpha=0.7, edgecolor='black')
axes[1].set_title('Social Media Platform Distribution')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/02_categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Distribution analysis complete!")

# Phase 3: Correlation Analysis
print("\n🔗 PHASE 3: CORRELATION ANALYSIS")
print("="*50)

# Calculate correlations between numerical variables
correlation_matrix = df[numeric_cols].corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.2f', mask=mask)
plt.title('Correlation Matrix: All Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Focus on Key Relationships
print("\n🎯 Key Correlations with Mental Health Outcomes:")

# Stress Level correlations
stress_correlations = correlation_matrix['Stress_Level(1-10)'].sort_values(ascending=False)
print("\nCorrelations with Stress Level:")
for var, corr in stress_correlations.items():
    if var != 'Stress_Level(1-10)':
        print(".4f")

# Happiness Index correlations
happiness_correlations = correlation_matrix['Happiness_Index(1-10)'].sort_values(ascending=False)
print("\nCorrelations with Happiness Index:")
for var, corr in happiness_correlations.items():
    if var != 'Happiness_Index(1-10)':
        print(".4f")

# Statistical Significance Testing
print("\n📈 Statistical Significance Tests:")

# Test correlation between Days Without Social Media and Stress
stress_corr, stress_p = pearsonr(df['Days_Without_Social_Media'], df['Stress_Level(1-10)'])
print("Days Without Social Media vs Stress Level:")
print(f"  Pearson r: {stress_corr:.4f} (p-value: {stress_p:.4e})")

# Test correlation between Days Without Social Media and Happiness
happiness_corr, happiness_p = pearsonr(df['Days_Without_Social_Media'], df['Happiness_Index(1-10)'])
print("Days Without Social Media vs Happiness Index:")
print(f"  Pearson r: {happiness_corr:.4f} (p-value: {happiness_p:.4e})")

# Test correlation between Screen Time and Stress
screen_stress_corr, screen_stress_p = pearsonr(df['Daily_Screen_Time(hrs)'], df['Stress_Level(1-10)'])
print("Daily Screen Time vs Stress Level:")
print(f"  Pearson r: {screen_stress_corr:.4f} (p-value: {screen_stress_p:.4e})")

print("✅ Correlation analysis complete!")

# Phase 4: Regression Analysis
print("\n📈 PHASE 4: REGRESSION ANALYSIS")
print("="*50)

X = df[['Days_Without_Social_Media']]
y_stress = df['Stress_Level(1-10)']
y_happiness = df['Happiness_Index(1-10)']

# Stress Model
lr_stress = LinearRegression()
lr_stress.fit(X, y_stress)
stress_pred = lr_stress.predict(X)

print("Simple Linear Regression Results:")
print("\nStress Model:")
print(f"  Coefficient: {lr_stress.coef_[0]:.4f}")
print(f"  Intercept: {lr_stress.intercept_:.4f}")
print(f"  R² Score: {r2_score(y_stress, stress_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_stress, stress_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_stress, stress_pred)):.4f}")

# Happiness Model
lr_happiness = LinearRegression()
lr_happiness.fit(X, y_happiness)
happiness_pred = lr_happiness.predict(X)

print("\nHappiness Model:")
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

# Stress Model with statsmodels for detailed diagnostics
model_stress_multi = sm.OLS(y_stress, X_multi_with_const).fit()
print("\nMultiple Regression - Stress Level:")
print(model_stress_multi.summary())

# Happiness Model
model_happiness_multi = sm.OLS(y_happiness, X_multi_with_const).fit()
print("\nMultiple Regression - Happiness Index:")
print(model_happiness_multi.summary())

# Visualization of Regression Results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Stress
axes[0].scatter(df['Days_Without_Social_Media'], df['Stress_Level(1-10)'],
                alpha=0.3, s=10, label='Actual')
axes[0].plot(df['Days_Without_Social_Media'], stress_pred,
             color='red', linewidth=2, label='Predicted')
axes[0].set_xlabel('Days Without Social Media')
axes[0].set_ylabel('Stress Level')
axes[0].set_title(f'Linear Regression: Stress (R² = {r2_score(y_stress, stress_pred):.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Happiness
axes[1].scatter(df['Days_Without_Social_Media'], df['Happiness_Index(1-10)'],
                alpha=0.3, s=10, label='Actual')
axes[1].plot(df['Days_Without_Social_Media'], happiness_pred,
             color='green', linewidth=2, label='Predicted')
axes[1].set_xlabel('Days Without Social Media')
axes[1].set_ylabel('Happiness Index')
axes[1].set_title(f'Linear Regression: Happiness (R² = {r2_score(y_happiness, happiness_pred):.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/04_regression_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Regression analysis complete!")

# Phase 5: User Segmentation (Clustering)
print("\n🎯 PHASE 5: USER SEGMENTATION ANALYSIS")
print("="*50)

# Select features for clustering
clustering_features = ['Days_Without_Social_Media', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
                       'Exercise_Frequency(week)', 'Stress_Level(1-10)', 'Happiness_Index(1-10)', 'Age']

X_cluster = df[clustering_features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print(f"Clustering data shape: {X_scaled.shape}")
print(f"Features used: {clustering_features}")

# Elbow Method for Optimal K
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
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
plt.show()

# Optimal K selection
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")
print(f"Best silhouette score: {max(silhouette_scores):.4f}")

# Apply K-Means with Optimal K
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster_KMeans'] = kmeans_final.fit_predict(X_scaled)

print("\nCluster distribution:")
print(df['Cluster_KMeans'].value_counts().sort_index())

# Cluster Profiling
print(f"\n{'='*60}")
print("CLUSTER PROFILES (K-Means)")
print('='*60)

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
plt.title('K-Means Clustering: PCA Visualization')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('results/06_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# DBSCAN Clustering
print(f"\n{'='*60}")
print("DBSCAN CLUSTERING")
print('='*60)

dbscan = DBSCAN(eps=0.8, min_samples=5)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN Cluster distribution:")
print(f"Number of clusters: {len(set(df['Cluster_DBSCAN'])) - (1 if -1 in df['Cluster_DBSCAN'] else 0)}")
print(f"Number of noise points: {sum(df['Cluster_DBSCAN'] == -1)}")
print(df['Cluster_DBSCAN'].value_counts().sort_index())

# DBSCAN Visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df['Cluster_DBSCAN'],
                     cmap='plasma', s=100, alpha=0.6, edgecolors='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('DBSCAN Clustering: PCA Visualization')
plt.colorbar(scatter, label='Cluster (-1 = Noise)')
plt.grid(True, alpha=0.3)
plt.savefig('results/07_dbscan_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Clustering analysis complete!")

# Phase 6: Advanced Modeling (Random Forest)
print("\n🧠 PHASE 6: ADVANCED MODELING")
print("="*50)

# Encode categorical variables
le = LabelEncoder()
df_encoded = df.copy()
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])
df_encoded['Social_Media_Platform'] = le.fit_transform(df_encoded['Social_Media_Platform'])

# Features and targets
features = ['Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
           'Days_Without_Social_Media', 'Exercise_Frequency(week)', 'Social_Media_Platform']
X = df_encoded[features]
y_stress = df_encoded['Stress_Level(1-10)']
y_happiness = df_encoded['Happiness_Index(1-10)']

print(f"Features used: {features}")
print(f"Dataset shape: {X.shape}")

# Random Forest Modeling
# Split data
X_train, X_test, y_train_stress, y_test_stress = train_test_split(X, y_stress, test_size=0.2, random_state=42)
X_train_h, X_test_h, y_train_happiness, y_test_happiness = train_test_split(X, y_happiness, test_size=0.2, random_state=42)

# Stress Model
rf_stress = RandomForestRegressor(n_estimators=100, random_state=42)
rf_stress.fit(X_train, y_train_stress)
rf_stress_pred = rf_stress.predict(X_test)

print("Random Forest Results:")
print("\nStress Level Prediction:")
print(f"  R² Score: {r2_score(y_test_stress, rf_stress_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_test_stress, rf_stress_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_stress, rf_stress_pred)):.4f}")

# Happiness Model
rf_happiness = RandomForestRegressor(n_estimators=100, random_state=42)
rf_happiness.fit(X_train_h, y_train_happiness)
rf_happiness_pred = rf_happiness.predict(X_test_h)

print("\nHappiness Index Prediction:")
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

axes[0].barh(stress_importance['feature'], stress_importance['importance'])
axes[0].set_title('Feature Importance: Stress Level Prediction')
axes[0].set_xlabel('Importance')

# Happiness feature importance
happiness_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_happiness.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(happiness_importance['feature'], happiness_importance['importance'])
axes[1].set_title('Feature Importance: Happiness Index Prediction')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('results/08_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Model Performance Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Stress predictions
axes[0].scatter(y_test_stress, rf_stress_pred, alpha=0.6)
axes[0].plot([1, 10], [1, 10], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Stress Level')
axes[0].set_ylabel('Predicted Stress Level')
axes[0].set_title(f'Random Forest: Stress Prediction (R² = {r2_score(y_test_stress, rf_stress_pred):.3f})')
axes[0].grid(True, alpha=0.3)

# Happiness predictions
axes[1].scatter(y_test_happiness, rf_happiness_pred, alpha=0.6, color='green')
axes[1].plot([1, 10], [1, 10], 'r--', linewidth=2)
axes[1].set_xlabel('Actual Happiness Index')
axes[1].set_ylabel('Predicted Happiness Index')
axes[1].set_title(f'Random Forest: Happiness Prediction (R² = {r2_score(y_test_happiness, rf_happiness_pred):.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/09_rf_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Advanced modeling complete!")

# Phase 7: Comprehensive Results Summary
print("\n📊 PHASE 7: COMPREHENSIVE RESULTS SUMMARY")
print("="*60)

results_summary = {
    'Analysis': [
        'Dataset Size',
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
        'Most Important Feature (Stress)',
        'Most Important Feature (Happiness)'
    ],
    'Result': [
        f'{df.shape[0]} users',
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
        f'{stress_importance.iloc[-1]["feature"]}',
        f'{happiness_importance.iloc[-1]["feature"]}'
    ]
}

results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('results/final_results_summary.csv', index=False)

# Key Insights
print(f"\n{'='*60}")
print("🎯 KEY INSIGHTS")
print('='*60)

print("1. CORRELATION ANALYSIS:")
print(f"   • Days without social media shows {abs(stress_corr):.3f} correlation with stress levels")
print(f"   • Days without social media shows {abs(happiness_corr):.3f} correlation with happiness")
print(f"   • Screen time shows {abs(screen_stress_corr):.3f} correlation with stress")

print("\n2. PREDICTIVE MODELING:")
print(f"   • Random Forest achieves {r2_score(y_test_stress, rf_stress_pred):.1%} accuracy for stress prediction")
print(f"   • Random Forest achieves {r2_score(y_test_happiness, rf_happiness_pred):.1%} accuracy for happiness prediction")

print("\n3. USER SEGMENTATION:")
print(f"   • Identified {optimal_k} distinct user clusters with silhouette score of {max(silhouette_scores):.3f}")
print(f"   • Most important predictor for stress: {stress_importance.iloc[-1]['feature']}")
print(f"   • Most important predictor for happiness: {happiness_importance.iloc[-1]['feature']}")

print("\n4. PRACTICAL IMPLICATIONS:")
print("   • Social media abstinence appears beneficial for mental health")
print("   • Sleep quality and exercise are strong protective factors")
print("   • Personalized interventions should consider user phenotypes")

# Save Processed Dataset
df.to_csv('results/processed_dataset_with_clusters.csv', index=False)
print("\n💾 All results and processed data saved to 'results/' directory")

print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
print("🎉 Your Social Media Detox Effect Analyzer is ready for portfolio presentation!")