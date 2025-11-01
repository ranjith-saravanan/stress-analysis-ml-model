"""
Social Media Detox Effect Analyzer - Flask Backend API
RESTful API for model predictions and data analysis
Author: Ranjith Saravanan
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables for models
rf_stress = None
rf_happiness = None
le_gender = None
le_platform = None
df = None

def load_and_train_models():
    """Load data and train models on startup"""
    global rf_stress, rf_happiness, le_gender, le_platform, df
    
    try:
        # Try to load processed data first
        df = pd.read_csv('results/processed_dataset_with_clusters.csv')
    except:
        # Load original data
        df = pd.read_csv(r'c:\Users\RANJITH S\Downloads\archive (3)\Mental_Health_and_Social_Media_Balance_Dataset.csv')
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_platform = LabelEncoder()
    
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
    df['Platform_Encoded'] = le_platform.fit_transform(df['Social_Media_Platform'])
    
    # Prepare features
    features = ['Age', 'Gender_Encoded', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 
               'Days_Without_Social_Media', 'Exercise_Frequency(week)', 'Platform_Encoded']
    X = df[features]
    y_stress = df['Stress_Level(1-10)']
    y_happiness = df['Happiness_Index(1-10)']
    
    # Train models
    rf_stress = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_stress.fit(X, y_stress)
    
    rf_happiness = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_happiness.fit(X, y_happiness)
    
    print("Models trained successfully!")
    return df

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Social Media Detox Effect Analyzer API',
        'version': '1.0',
        'endpoints': {
            '/api/predict': 'POST - Get stress and happiness predictions',
            '/api/stats': 'GET - Get dataset statistics',
            '/api/correlations': 'GET - Get correlation analysis',
            '/api/clusters': 'GET - Get cluster analysis',
            '/api/recommendations': 'POST - Get personalized recommendations'
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict stress and happiness levels"""
    try:
        data = request.json
        
        # Validate input
        required_fields = ['age', 'gender', 'screen_time', 'sleep_quality', 
                          'days_without_sm', 'exercise_freq', 'platform']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Encode categorical variables
        gender_encoded = le_gender.transform([data['gender']])[0]
        platform_encoded = le_platform.transform([data['platform']])[0]
        
        # Prepare input
        input_data = np.array([[
            data['age'],
            gender_encoded,
            data['screen_time'],
            data['sleep_quality'],
            data['days_without_sm'],
            data['exercise_freq'],
            platform_encoded
        ]])
        
        # Make predictions
        stress_pred = float(rf_stress.predict(input_data)[0])
        happiness_pred = float(rf_happiness.predict(input_data)[0])
        
        # Calculate wellness score
        wellness_score = (10 - stress_pred + happiness_pred) / 2
        
        # Determine stress category
        if stress_pred < 4:
            stress_category = 'Low'
            stress_indicator = 'low'
        elif stress_pred < 7:
            stress_category = 'Moderate'
            stress_indicator = 'moderate'
        else:
            stress_category = 'High'
            stress_indicator = 'high'
        
        # Determine happiness category
        if happiness_pred < 4:
            happiness_category = 'Low'
            happiness_indicator = 'low'
        elif happiness_pred < 7:
            happiness_category = 'Moderate'
            happiness_indicator = 'moderate'
        else:
            happiness_category = 'High'
            happiness_indicator = 'high'
        
        return jsonify({
            'success': True,
            'predictions': {
                'stress_level': round(stress_pred, 2),
                'stress_category': stress_category,
                'stress_indicator': stress_indicator,
                'happiness_index': round(happiness_pred, 2),
                'happiness_category': happiness_category,
                'happiness_indicator': happiness_indicator,
                'wellness_score': round(wellness_score, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe().to_dict()
        
        return jsonify({
            'success': True,
            'total_records': len(df),
            'features': df.shape[1],
            'missing_values': int(df.isnull().sum().sum()),
            'statistics': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations', methods=['GET'])
def get_correlations():
    """Get correlation analysis"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # Get key correlations with stress
        stress_corr = correlation_matrix['Stress_Level(1-10)'].sort_values(ascending=False)
        happiness_corr = correlation_matrix['Happiness_Index(1-10)'].sort_values(ascending=False)
        
        return jsonify({
            'success': True,
            'stress_correlations': stress_corr.to_dict(),
            'happiness_correlations': happiness_corr.to_dict(),
            'key_findings': {
                'screen_time_stress': float(correlation_matrix.loc['Daily_Screen_Time(hrs)', 'Stress_Level(1-10)']),
                'screen_time_happiness': float(correlation_matrix.loc['Daily_Screen_Time(hrs)', 'Happiness_Index(1-10)']),
                'sleep_stress': float(correlation_matrix.loc['Sleep_Quality(1-10)', 'Stress_Level(1-10)']),
                'sleep_happiness': float(correlation_matrix.loc['Sleep_Quality(1-10)', 'Happiness_Index(1-10)'])
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get cluster analysis"""
    try:
        if 'Cluster' in df.columns:
            cluster_analysis = df.groupby('Cluster').agg({
                'Age': 'mean',
                'Daily_Screen_Time(hrs)': 'mean',
                'Sleep_Quality(1-10)': 'mean',
                'Stress_Level(1-10)': 'mean',
                'Happiness_Index(1-10)': 'mean',
                'Exercise_Frequency(week)': 'mean'
            }).to_dict()
            
            cluster_sizes = df['Cluster'].value_counts().to_dict()
            
            return jsonify({
                'success': True,
                'cluster_analysis': cluster_analysis,
                'cluster_sizes': cluster_sizes
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Cluster data not available. Run complete analysis first.'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations"""
    try:
        data = request.json
        recommendations = []
        priority_recommendations = []
        
        # Screen time recommendations
        screen_time = data.get('screen_time', 0)
        if screen_time > 8:
            priority_recommendations.append({
                'category': 'Critical',
                'icon': 'alert',
                'title': 'Drastically Reduce Screen Time',
                'description': f'Your screen time ({screen_time}hrs) is very high. Reduce to under 5 hours daily.',
                'action': 'Set strict app time limits'
            })
        elif screen_time > 6:
            recommendations.append({
                'category': 'High Priority',
                'icon': 'warning',
                'title': 'Reduce Screen Time',
                'description': f'Your screen time ({screen_time}hrs) is above recommended levels.',
                'action': 'Gradually reduce by 1-2 hours per week'
            })
        
        # Sleep quality recommendations
        sleep_quality = data.get('sleep_quality', 0)
        if sleep_quality < 5:
            priority_recommendations.append({
                'category': 'Critical',
                'icon': 'sleep',
                'title': 'Improve Sleep Quality',
                'description': f'Your sleep quality ({sleep_quality}/10) needs immediate attention.',
                'action': 'Establish consistent sleep schedule, avoid screens 1hr before bed'
            })
        elif sleep_quality < 7:
            recommendations.append({
                'category': 'Important',
                'icon': 'moon',
                'title': 'Enhance Sleep Habits',
                'description': 'Better sleep can reduce stress significantly.',
                'action': 'Create a relaxing bedtime routine'
            })
        
        # Exercise recommendations
        exercise_freq = data.get('exercise_freq', 0)
        if exercise_freq < 2:
            priority_recommendations.append({
                'category': 'High Priority',
                'icon': 'activity',
                'title': 'Increase Physical Activity',
                'description': f'Exercise only {exercise_freq}x/week. Aim for 3-4 sessions.',
                'action': 'Start with 20-minute walks, gradually increase intensity'
            })
        elif exercise_freq < 3:
            recommendations.append({
                'category': 'Moderate',
                'icon': 'fitness',
                'title': 'Boost Exercise Frequency',
                'description': 'Regular exercise improves mental health.',
                'action': 'Add one more workout session per week'
            })
        
        # Social media detox recommendations
        days_without_sm = data.get('days_without_sm', 0)
        if days_without_sm == 0:
            recommendations.append({
                'category': 'Moderate',
                'icon': 'phone',
                'title': 'Take Social Media Breaks',
                'description': 'No breaks from social media detected.',
                'action': 'Try a weekly 24-hour digital detox'
            })
        
        # If everything is good
        if not priority_recommendations and not recommendations:
            recommendations.append({
                'category': 'Excellent',
                'icon': 'success',
                'title': 'Maintaining Healthy Habits',
                'description': 'Great job! Your lifestyle habits are well-balanced.',
                'action': 'Keep up the good work and stay consistent'
            })
        
        return jsonify({
            'success': True,
            'priority_recommendations': priority_recommendations,
            'general_recommendations': recommendations,
            'total_recommendations': len(priority_recommendations) + len(recommendations)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['GET'])
def export_data():
    """Export filtered data"""
    try:
        # Get query parameters
        min_age = request.args.get('min_age', type=int)
        max_age = request.args.get('max_age', type=int)
        
        filtered_df = df.copy()
        if min_age:
            filtered_df = filtered_df[filtered_df['Age'] >= min_age]
        if max_age:
            filtered_df = filtered_df[filtered_df['Age'] <= max_age]
        
        return jsonify({
            'success': True,
            'data': filtered_df.to_dict(orient='records'),
            'count': len(filtered_df)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load data and train models on startup
    print("Loading data and training models...")
    df = load_and_train_models()
    
    # Run Flask app
    print("\nBackend API is ready!")
    print("API running at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("   - GET  /")
    print("   - POST /api/predict")
    print("   - GET  /api/stats")
    print("   - GET  /api/correlations")
    print("   - GET  /api/clusters")
    print("   - POST /api/recommendations")
    print("   - GET  /api/export")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
