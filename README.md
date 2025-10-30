# 🧠 Social Media Detox Effect Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Project Overview

A comprehensive machine learning framework analyzing the relationship between social media abstinence and mental health outcomes (stress and happiness levels) using advanced analytics including correlation analysis, regression modeling, user segmentation clustering, and time series prediction.

**Live Demo**: [View Analysis Results](https://github.com/ranjith-saravanan/stress-analysis-ml-model)

## 🎯 Key Features

- **🌐 Interactive Web GUI**: Beautiful Streamlit dashboard with AI-powered predictions
- **🔌 REST API Backend**: Flask API for easy integration with other applications
- **Multi-dimensional Analysis**: Pearson/Spearman correlations, simple/multiple regression
- **User Segmentation**: K-Means and DBSCAN clustering to identify distinct user phenotypes
- **Advanced Modeling**: Random Forest regression for predictive analytics
- **Real-time Insights**: Deploy-ready models for personalized recommendations
- **Comprehensive Visualizations**: 9+ publication-quality plots and dashboards

## 🔧 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.8+ |
| **Web Framework** | Streamlit, Flask |
| **ML Libraries** | scikit-learn, TensorFlow/Keras, statsmodels |
| **Data Science** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Algorithms** | Linear Regression, Random Forest, K-Means, DBSCAN, PCA |

## 📁 Project Structure

```
stress-analysis-ml-model/
│
├── README.md                          # Project documentation
├── GUI_GUIDE.md                       # Guide for running web app and API
├── requirements.txt                   # Python dependencies
├── stress_analysis.ipynb             # Main analysis notebook
├── run_complete_analysis.py          # Complete automated analysis
├── app.py                             # Streamlit web application
├── backend_api.py                     # Flask REST API
├── run_gui.bat                        # Quick launch script for GUI
├── run_backend.bat                    # Quick launch script for API
│
├── data/
│   └── Mental_Health_and_Social_Media_Balance_Dataset.csv
│
└── results/
    ├── 01_distributions.png
    ├── 02_categorical_distributions.png
    ├── 03_correlation_heatmap.png
    ├── 04_regression_results.png
    ├── 05_clustering_optimization.png
    ├── 06_kmeans_clusters.png
    ├── 07_dbscan_clusters.png
    ├── 08_feature_importance.png
    ├── 09_rf_predictions.png
    ├── final_results_summary.csv
    └── processed_dataset_with_clusters.csv
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ranjith-saravanan/stress-analysis-ml-model.git
cd stress-analysis-ml-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Run the Web Application (Recommended):**
```bash
# Quick launch with batch file
run_gui.bat

# Or manually with Python
streamlit run app.py
```

The web app will open at **http://localhost:8501** with features:
- 🏠 Interactive Dashboard
- 📊 Data Explorer
- 🤖 AI Predictor (get personalized predictions!)
- 📈 Analysis Results
- 💡 Insights & Recommendations

4. **Or Run the Backend API:**
```bash
# Quick launch with batch file
run_backend.bat

# Or manually with Python
python backend_api.py
```

API available at **http://localhost:5000** - See [GUI_GUIDE.md](GUI_GUIDE.md) for API documentation.

5. **Or Run the Complete Analysis:**
```bash
# Option 1: Run automated analysis
python run_complete_analysis.py

# Option 2: Run Jupyter notebook
jupyter notebook stress_analysis.ipynb
```

## 📊 Dataset

- **Source**: Mental Health and Social Media Balance Dataset
- **Size**: 500 users with comprehensive mental health metrics
- **Features**:
  - Age, Gender
  - Daily Screen Time (hours)
  - Sleep Quality (1-10 scale)
  - Stress Level (1-10 scale)
  - Days Without Social Media
  - Exercise Frequency (per week)
  - Social Media Platform
  - Happiness Index (1-10 scale)

## 🧪 Methodology

### Phase 1: Data Loading & Preprocessing
- Data cleaning and validation
- Handling missing values
- Categorical variable encoding
- Duplicate removal

### Phase 2: Exploratory Data Analysis (EDA)
- Distribution analysis of all features
- Categorical variable visualization
- Correlation heatmaps
- Statistical summaries

### Phase 3: Correlation Analysis
- Pearson & Spearman correlation tests
- Statistical significance testing
- Feature relationship mapping

### Phase 4: Regression Analysis
- Simple linear regression
- Multiple regression with confound control
- Model diagnostics and validation

### Phase 5: User Segmentation (Clustering)
- K-Means clustering with elbow method & silhouette analysis
- DBSCAN density-based clustering
- PCA visualization
- Cluster profiling and characterization

### Phase 6: Advanced Modeling
- Random Forest regression
- Feature importance analysis
- Model performance evaluation
- Cross-validation

### Phase 7: Results Summary
- Comprehensive metrics compilation
- Key insights extraction
- Practical recommendations

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Dataset Size | 500 users |
| Optimal Clusters | Identified via Silhouette Analysis |
| Random Forest Accuracy | High R² scores for both outcomes |
| Top Predictors | Sleep Quality, Exercise, Screen Time |

## 🎓 Skills Demonstrated

- ✅ Statistical hypothesis testing
- ✅ Supervised learning (regression)
- ✅ Unsupervised learning (clustering)
- ✅ Feature engineering
- ✅ Model evaluation & validation
- ✅ Data visualization
- ✅ Python programming
- ✅ Documentation & presentation

## 📊 Sample Visualizations

### Correlation Heatmap
![Correlation Analysis](results/03_correlation_heatmap.png)

### Clustering Results
![K-Means Clustering](results/06_kmeans_clusters.png)

### Feature Importance
![Feature Importance](results/08_feature_importance.png)

## 🔮 Future Enhancements

- [ ] Integrate LSTM neural networks for time series prediction
- [ ] Add real-time data collection via APIs
- [ ] Implement Transformer models for attention mechanisms
- [ ] Deploy as interactive Streamlit dashboard
- [ ] Incorporate causal inference techniques
- [ ] Add sentiment analysis of social media posts
- [ ] Develop mobile app for real-time monitoring

## 📝 Requirements

See `requirements.txt` for full list of dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- tensorflow >= 2.6.0
- statsmodels >= 0.13.0

## 👤 Author

**Ranjith Saravanan**
- GitHub: [@ranjith-saravanan](https://github.com/ranjith-saravanan)
- LinkedIn: [Connect with me](https://linkedin.com/in/ranjith-saravanan)
- Portfolio: [View My Work](https://github.com/ranjith-saravanan)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Mental Health Research Community
- Open Source ML Libraries
- Kaggle Dataset Contributors

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue on GitHub
- Connect via LinkedIn
- Email: [Your Email]

---

⭐ **If you found this project helpful, please give it a star!** ⭐

**Project Status**: ✅ Complete and ready for portfolio presentation
