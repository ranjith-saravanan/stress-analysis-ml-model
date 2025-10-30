# 🚀 Running the GUI Applications

## Web Application (Streamlit)

### Option 1: Streamlit Frontend (Recommended)

The Streamlit app provides a beautiful, interactive dashboard with multiple pages for data exploration, AI predictions, and insights.

#### Installation

```bash
# Install required packages
pip install streamlit plotly flask flask-cors

# Or install all requirements
pip install -r requirements.txt
```

#### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

#### Features

- 🏠 **Home Dashboard**: Overview with key metrics and findings
- 📊 **Data Explorer**: Interactive filtering and visualization
- 🤖 **AI Predictor**: Get personalized stress and happiness predictions
- 📈 **Analysis Results**: View all generated visualizations
- 💡 **Insights & Recommendations**: Actionable mental health tips

---

## Backend API (Flask)

### Option 2: Flask REST API

The Flask backend provides RESTful API endpoints for integration with other applications.

#### Running the Backend API

```bash
python backend_api.py
```

The API will be available at `http://localhost:5000`

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API documentation |
| POST | `/api/predict` | Get stress/happiness predictions |
| GET | `/api/stats` | Dataset statistics |
| GET | `/api/correlations` | Correlation analysis |
| GET | `/api/clusters` | Cluster analysis |
| POST | `/api/recommendations` | Personalized recommendations |
| GET | `/api/export` | Export filtered data |

#### Example API Usage

**Predict Stress & Happiness:**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "Male",
    "screen_time": 6.5,
    "sleep_quality": 6,
    "days_without_sm": 2,
    "exercise_freq": 3,
    "platform": "Instagram"
  }'
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "stress_level": 6.85,
    "stress_category": "Moderate",
    "stress_emoji": "🟡",
    "happiness_index": 5.42,
    "happiness_category": "Moderate",
    "happiness_emoji": "😐",
    "wellness_score": 6.78
  }
}
```

**Get Statistics:**
```bash
curl http://localhost:5000/api/stats
```

**Get Correlations:**
```bash
curl http://localhost:5000/api/correlations
```

---

## Running Both (Full Stack)

You can run both the frontend and backend simultaneously:

### Terminal 1 - Backend API:
```bash
python backend_api.py
```

### Terminal 2 - Streamlit Frontend:
```bash
streamlit run app.py
```

---

## Troubleshooting

### Issue: Module not found

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Port already in use

**Streamlit:**
```bash
streamlit run app.py --server.port 8502
```

**Flask:**
```bash
# Modify backend_api.py line 311 to use different port
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: Data file not found

Make sure you have:
1. Run `python run_complete_analysis.py` first to generate processed data
2. Or ensure the original dataset is at the correct path

---

## Screenshots

### Streamlit Dashboard
- Beautiful gradient UI with interactive widgets
- Real-time predictions
- Interactive plotly charts
- Personalized recommendations

### API Response
- JSON responses for easy integration
- RESTful design
- CORS enabled for web applications

---

## Technology Stack

**Frontend:**
- Streamlit - Web framework
- Plotly - Interactive visualizations
- Custom CSS - Beautiful styling

**Backend:**
- Flask - REST API framework
- Flask-CORS - Cross-origin support
- Scikit-learn - ML models

**Machine Learning:**
- Random Forest Regressor
- K-Means Clustering
- Feature Engineering

---

## Next Steps

1. **Run the Streamlit app** for immediate interactive analysis
2. **Use the Flask API** for integration with other applications
3. **Customize the UI** by modifying `app.py`
4. **Extend the API** by adding more endpoints in `backend_api.py`

---

## Project Structure

```
dse/
├── app.py                      # Streamlit frontend
├── backend_api.py              # Flask REST API
├── run_complete_analysis.py   # Complete ML analysis
├── requirements.txt            # Python dependencies
├── results/                    # Generated visualizations
│   ├── *.png                  # Charts and graphs
│   └── *.csv                  # Processed data
└── README.md                   # Project documentation
```

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments in `app.py` and `backend_api.py`
3. Ensure all dependencies are installed

---

**Author:** Ranjith Saravanan
**Version:** 1.0
**License:** MIT
