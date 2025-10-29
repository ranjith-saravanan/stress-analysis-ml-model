# 🚀 Quick Start Guide - Social Media Detox Effect Analyzer

## Running the Project

### Option 1: Run the Complete Python Script (Recommended)
```bash
python run_complete_analysis.py
```
**Output**: All 9 visualizations + results in `results/` folder

### Option 2: Run Jupyter Notebook
```bash
jupyter notebook stress_analysis.ipynb
```
**Then**: Run all cells sequentially (Cell → Run All)

---

## 📂 Project Structure
```
dse/
├── README.md                                  # Main documentation
├── PROJECT_SUMMARY.md                         # Detailed results summary
├── QUICK_START.md                            # This file
├── requirements.txt                           # Dependencies
├── .gitignore                                # Git ignore rules
│
├── stress_analysis.ipynb                     # Main Jupyter notebook
├── run_complete_analysis.py                  # Standalone script
├── social_media_detox_analyzer.py           # Analyzer module
├── social_media_detox_analyzer_core.py      # Core functions
│
├── data/
│   └── Mental_Health_and_Social_Media_Balance_Dataset.csv
│
└── results/
    ├── 01_distributions.png                  ✅ Generated
    ├── 02_categorical_distributions.png      ✅ Generated
    ├── 03_correlation_heatmap.png           ✅ Generated
    ├── 04_regression_results.png            ✅ Generated
    ├── 05_clustering_optimization.png       ✅ Generated
    ├── 06_kmeans_clusters.png               ✅ Generated
    ├── 07_dbscan_clusters.png               ✅ Generated
    ├── 08_feature_importance.png            ✅ Generated
    ├── 09_rf_predictions.png                ✅ Generated
    ├── final_results_summary.csv            ✅ Generated
    └── processed_dataset_with_clusters.csv  ✅ Generated
```

---

## 🎯 What Each File Does

| File | Purpose |
|------|---------|
| `stress_analysis.ipynb` | Interactive analysis with explanations |
| `run_complete_analysis.py` | Run entire analysis in one command |
| `requirements.txt` | List of required Python packages |
| `README.md` | Professional project documentation |
| `PROJECT_SUMMARY.md` | Detailed findings and insights |

---

## 📊 Analysis Phases

### Phase 1: Data Loading & Preprocessing
- Load dataset (500 users)
- Check for missing values
- Handle categorical variables
- Remove duplicates

### Phase 2: Exploratory Data Analysis
- Distribution plots for all features
- Gender and platform visualizations
- Statistical summaries

### Phase 3: Correlation Analysis
- Correlation matrix heatmap
- Pearson correlation tests
- Statistical significance testing

### Phase 4: Regression Analysis
- Simple linear regression
- Multiple regression with controls
- Model diagnostics

### Phase 5: User Segmentation
- K-Means clustering (optimal k=2)
- DBSCAN density-based clustering
- PCA visualization
- Cluster profiling

### Phase 6: Advanced Modeling
- Random Forest regression
- Feature importance analysis
- Model performance evaluation

### Phase 7: Results Summary
- Comprehensive metrics
- Key insights
- Practical recommendations

---

## 🔑 Key Results at a Glance

### Main Findings
✅ **Screen time** is the most important factor (not abstinence duration)  
✅ Found **2 distinct user clusters**: High-stress vs Well-balanced  
✅ **Sleep quality** strongly influences mental health  
✅ Random Forest achieves **~50% prediction accuracy**  

### Correlations
- Screen Time ↔ Stress: **0.74** (strong positive) ⚠️
- Screen Time ↔ Happiness: **-0.71** (strong negative) ⚠️
- Sleep ↔ Stress: **-0.58** (moderate negative) ✅
- Sleep ↔ Happiness: **0.68** (moderate positive) ✅

---

## 🎨 Viewing Results

### View Visualizations
All 9 plots are saved as high-resolution PNG files in `results/` folder.

### View Data Results
- `final_results_summary.csv` - All metrics in table format
- `processed_dataset_with_clusters.csv` - Full dataset with cluster assignments

---

## 🐛 Troubleshooting

### If you get "Module not found" error:
```bash
pip install -r requirements.txt
```

### If Jupyter kernel crashes:
```bash
# Update typing_extensions
pip install --upgrade typing_extensions

# Or use the Python script instead
python run_complete_analysis.py
```

### If path errors occur:
Update line 61 in `run_complete_analysis.py` with your dataset path:
```python
df = pd.read_csv(r'YOUR_PATH_HERE\Mental_Health_and_Social_Media_Balance_Dataset.csv')
```

---

## 📤 Sharing on GitHub

### Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Social Media Detox Effect Analyzer"
```

### Push to GitHub
```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/ranjith-saravanan/stress-analysis-ml-model.git
git branch -M main
git push -u origin main
```

---

## 📝 Portfolio Presentation Tips

### Highlight These Points:
1. **Multi-phase ML pipeline** (7 comprehensive phases)
2. **Multiple algorithms** (Linear Regression, Random Forest, K-Means, DBSCAN, PCA)
3. **Strong visualizations** (9 publication-quality plots)
4. **Actionable insights** (Screen time > Abstinence duration)
5. **Real-world application** (Mental health interventions)

### Key Metrics to Mention:
- 500 users analyzed
- 55% variance explained in multiple regression
- 48-49% accuracy in Random Forest predictions
- 2 distinct user phenotypes identified
- Strong correlations discovered (r > 0.7)

---

## 🔮 Next Steps for Enhancement

1. **Add time series prediction** using LSTM
2. **Deploy as web app** using Streamlit
3. **Integrate real data** from wearable devices
4. **Add sentiment analysis** of social media posts
5. **Create interactive dashboard** for real-time predictions

---

## 📞 Support

**Issues?** Check:
1. Python version (3.8+ required)
2. All packages installed (`pip list`)
3. Dataset path is correct
4. `results/` folder exists

**Questions?** 
- Review `PROJECT_SUMMARY.md` for detailed findings
- Check code comments in `run_complete_analysis.py`
- Refer to `README.md` for methodology

---

## ✅ Project Checklist

- [x] Data preprocessing complete
- [x] EDA visualizations generated
- [x] Statistical analysis performed
- [x] ML models trained & evaluated
- [x] Clustering analysis complete
- [x] Results documented
- [x] Code well-commented
- [x] README created
- [x] Ready for portfolio!

---

**🎉 Congratulations! Your project is complete and portfolio-ready!**

**Author**: Ranjith Saravanan  
**Last Updated**: October 29, 2025  
**Status**: ✅ COMPLETE
