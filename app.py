"""
Social Media Detox Effect Analyzer - Interactive Web Application
Beautiful GUI with Streamlit
Author: Ranjith Saravanan
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Social Media Detox Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, vibrant styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #00d2ff 0%, #3a47d5 50%, #ff0080 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 900;
        margin-bottom: 1rem;
        animation: gradient 3s ease infinite;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #00d2ff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.3);
    }
    
    /* Modern gradient cards */
    .metric-card {
        background: linear-gradient(135deg, #00d2ff 0%, #3a47d5 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 210, 255, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 210, 255, 0.5);
    }
    
    /* Neon insight boxes */
    .insight-box {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.1) 0%, rgba(255, 0, 128, 0.1) 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00d2ff;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2);
    }
    
    /* Interactive button with glow effect */
    .stButton>button {
        background: linear-gradient(135deg, #00d2ff 0%, #ff0080 100%);
        color: white;
        font-weight: bold;
        border-radius: 25px;
        padding: 15px 40px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(255, 0, 128, 0.6);
    }
    
    /* Real-time indicator */
    .realtime-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00ff87 0%, #00d2ff 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Input field styling */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Success/warning badges */
    .status-good {
        color: #00ff87;
        font-weight: bold;
    }
    .status-warning {
        color: #ffd700;
        font-weight: bold;
    }
    .status-critical {
        color: #ff0080;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('results/processed_dataset_with_clusters.csv')
        return df
    except:
        # If processed data doesn't exist, load original
        df = pd.read_csv(r'c:\Users\RANJITH S\Downloads\archive (3)\Mental_Health_and_Social_Media_Balance_Dataset.csv')
        return df

# Train models
@st.cache_resource
def train_models(df):
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_platform = LabelEncoder()
    
    df_encoded = df.copy()
    if 'Gender' in df_encoded.columns:
        df_encoded['Gender'] = le_gender.fit_transform(df_encoded['Gender'])
    if 'Social_Media_Platform' in df_encoded.columns:
        df_encoded['Social_Media_Platform'] = le_platform.fit_transform(df_encoded['Social_Media_Platform'])
    
    # Features
    features = ['Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 
               'Days_Without_Social_Media', 'Exercise_Frequency(week)', 'Social_Media_Platform']
    X = df_encoded[features]
    y_stress = df_encoded['Stress_Level(1-10)']
    y_happiness = df_encoded['Happiness_Index(1-10)']
    
    # Train Random Forest models
    rf_stress = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_stress.fit(X, y_stress)
    
    rf_happiness = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_happiness.fit(X, y_happiness)
    
    return rf_stress, rf_happiness, le_gender, le_platform

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header"> Social Media Detox Effect Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Mental Health Insights | Easy-to-Use Interface</p>', unsafe_allow_html=True)
    
    # Quick tip banner
    st.info(" **Tip:** New here? Click ' AI Predictor (Start Here!)' in the sidebar for instant results!")
    
    # Sidebar Navigation with helpful descriptions
    st.sidebar.title(" Navigation")
    st.sidebar.markdown('<span class="realtime-badge"> LIVE</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Add help section
    with st.sidebar.expander(" Need Help? Click Here!"):
        st.markdown("""
        **Getting Started:**
        1.  Start with Home Dashboard
        2.  Try AI Predictor for instant results
        3. ⏱ Track daily with Real-Time Tracker
        4.  Plan with Future Predictions
        
        **Tips:**
        - Hover over  icons for hints
        - Use sliders for easy input
        - All data saves automatically
        - Download your progress anytime!
        """)
    
    page = st.sidebar.radio("Choose a page:", [
        " Home Dashboard",
        " AI Predictor (Start Here!)",
        "⏱ Real-Time Tracker",
        " Future Predictions",
        " Data Explorer",
        " Analysis Results",
        " Insights & Recommendations"
    ], help="Select a page to explore different features")
    
    # Quick tips in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Quick Tip")
    tips = [
        " New here? Start with AI Predictor!",
        " Track daily for best insights",
        " Use Future Predictions to plan goals",
        " Download your tracking data anytime",
        " Sleep quality matters most!",
        " Reduce screen time for lower stress"
    ]
    import random
    st.sidebar.info(random.choice(tips))
    
    # Load data
    df = load_data()
    
    # Map pages
    if page == " Home Dashboard":
        show_home_dashboard(df)
    elif page == " AI Predictor (Start Here!)":
        show_ai_predictor(df)
    elif page == "⏱ Real-Time Tracker":
        show_realtime_tracker(df)
    elif page == " Future Predictions":
        show_future_predictions(df)
    elif page == " Data Explorer":
        show_data_explorer(df)
    elif page == " Analysis Results":
        show_analysis_results(df)
    elif page == " Insights & Recommendations":
        show_insights()

def show_home_dashboard(df):
    st.header(" Project Overview")
    
    # Welcome message
    st.success(" **Welcome to the Social Media Detox Effect Analyzer!** This AI-powered tool helps you understand and improve your mental health.")
    
    # Quick start guide
    with st.expander(" NEW HERE? Start Here! (Click to expand)", expanded=True):
        st.markdown("""
        ###  Quick Start Guide
        
        **For First-Time Users:**
        1.  Click "**AI Predictor (Start Here!)**" in the sidebar
        2.  Fill in your information (takes 1 minute!)
        3.  Get instant predictions about your stress & happiness
        4.  Read personalized recommendations
        
        **For Daily Tracking:**
        - ⏱ Use "**Real-Time Tracker**" to log daily progress
        -  See your improvement over time
        -  Download your tracking data
        
        **For Planning Ahead:**
        -  Try "**Future Predictions**" to test lifestyle changes
        -  See 4-week improvement timeline
        -  Get action steps to reach your goals
        
        ** Choose a page from the sidebar to get started!**
        """)
    
    st.markdown("---")
    
    # Key Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>500</h3>
            <p>Users Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>0.74</h3>
            <p>Screen Time Correlation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>55%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>2</h3>
            <p>User Clusters</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Summary
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Dataset Summary")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Features:** {df.shape[1]}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
        st.subheader(" Quick Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
    
    with col2:
        st.subheader(" Key Findings")
        st.markdown("""
        <div class="insight-box">
            <h4> Major Discovery</h4>
            <p><strong>Screen time (0.74 correlation)</strong> is <strong>90x MORE important</strong> 
            than social media abstinence (-0.008) for mental health!</p>
        </div>
        
        <div class="insight-box">
            <h4> Two User Phenotypes</h4>
            <p><strong>Cluster 0:</strong> High-stress users (6.85hrs screen time)<br>
            <strong>Cluster 1:</strong> Well-balanced users (4.29hrs screen time)</p>
        </div>
        
        <div class="insight-box">
            <h4> Sleep Quality Matters</h4>
            <p><strong>-59%</strong> correlation with stress<br>
            <strong>+68%</strong> correlation with happiness</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    st.subheader(" Distribution Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Stress_Level(1-10)', 
                          title='Stress Level Distribution',
                          color_discrete_sequence=['#ff6b6b'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Happiness_Index(1-10)', 
                          title='Happiness Index Distribution',
                          color_discrete_sequence=['#51cf66'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    st.header(" Interactive Data Explorer")
    
    # Filters
    st.sidebar.subheader(" Filters")
    
    age_range = st.sidebar.slider("Age Range", 
                                   int(df['Age'].min()), 
                                   int(df['Age'].max()), 
                                   (int(df['Age'].min()), int(df['Age'].max())))
    
    if 'Gender' in df.columns:
        gender_options = ['All'] + list(df['Gender'].unique())
        gender_filter = st.sidebar.multiselect("Gender", gender_options, default=['All'])
    
    screen_time_range = st.sidebar.slider("Screen Time (hrs)", 
                                          float(df['Daily_Screen_Time(hrs)'].min()), 
                                          float(df['Daily_Screen_Time(hrs)'].max()),
                                          (float(df['Daily_Screen_Time(hrs)'].min()), 
                                           float(df['Daily_Screen_Time(hrs)'].max())))
    
    # Filter data
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    filtered_df = filtered_df[(filtered_df['Daily_Screen_Time(hrs)'] >= screen_time_range[0]) & 
                              (filtered_df['Daily_Screen_Time(hrs)'] <= screen_time_range[1])]
    
    if 'Gender' in df.columns and 'All' not in gender_filter:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter)]
    
    st.write(f"**Filtered Records:** {len(filtered_df)} / {len(df)}")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([" Data Table", " Correlations", " Visualizations"])
    
    with tab1:
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label=" Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, 
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(filtered_df, 
                           x='Daily_Screen_Time(hrs)', 
                           y='Stress_Level(1-10)',
                           color='Happiness_Index(1-10)',
                           size='Age',
                           title='Screen Time vs Stress',
                           color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(filtered_df, 
                        x='Gender' if 'Gender' in filtered_df.columns else None,
                        y='Stress_Level(1-10)',
                        title='Stress Distribution by Gender',
                        color='Gender' if 'Gender' in filtered_df.columns else None)
            st.plotly_chart(fig, use_container_width=True)

def show_ai_predictor(df):
    st.header(" AI-Powered Mental Health Predictor")
    
    # Welcome message for new users
    st.success(" **Welcome!** This is the easiest way to get started. Just fill in your information below and get instant predictions!")
    
    # Instructions
    with st.expander(" How to Use This Tool (Click to expand)", expanded=False):
        st.markdown("""
        **Simple Steps:**
        1.  Fill in your personal information (left side)
        2.  Adjust the lifestyle sliders (right side) 
        3.  Click the big "Predict" button at bottom
        4.  Get instant stress & happiness predictions!
        5.  Read personalized recommendations
        
        **Pro Tip:** Be honest with your inputs for accurate results!
        """)
    
    st.write("Enter your information to get personalized stress and happiness predictions!")
    
    # Train models
    with st.spinner(" Training AI models..."):
        rf_stress, rf_happiness, le_gender, le_platform = train_models(df)
    
    st.success(" Models trained successfully!")
    
    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Personal Information")
        age = st.slider("Age", 16, 50, 30, help="Your current age")
        gender = st.selectbox("Gender", df['Gender'].unique() if 'Gender' in df.columns else ['Male', 'Female', 'Other'],
                             help="Select your gender")
        platform = st.selectbox("Primary Social Media Platform", 
                               df['Social_Media_Platform'].unique() if 'Social_Media_Platform' in df.columns else ['Facebook', 'Instagram'],
                               help="Which platform do you use most?")
    
    with col2:
        st.subheader(" Lifestyle Factors")
        st.caption(" Tip: Move the sliders below to match your typical day")
        screen_time = st.slider(" Daily Screen Time (hours)", 1.0, 12.0, 5.5, 0.5,
                               help="Total hours spent on screens (phone, computer, TV)")
        sleep_quality = st.slider(" Sleep Quality (1-10)", 1, 10, 7,
                                 help="How well do you sleep? 1=Very poor, 10=Excellent")
        days_without_sm = st.slider(" Days Without Social Media (per month)", 0, 30, 0,
                                   help="How many days per month do you take a complete break?")
        exercise_freq = st.slider(" Exercise Frequency (times/week)", 0, 7, 2,
                                 help="How many times per week do you exercise?")
    
    st.markdown("---")
    
    # Prediction button with more emphasis
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button(" Predict My Mental Health Status", use_container_width=True, type="primary")
    
    if predict_btn:
        # Prepare input
        gender_encoded = le_gender.transform([gender])[0]
        platform_encoded = le_platform.transform([platform])[0]
        
        input_data = np.array([[age, gender_encoded, screen_time, sleep_quality, 
                               days_without_sm, exercise_freq, platform_encoded]])
        
        # Make predictions
        stress_pred = rf_stress.predict(input_data)[0]
        happiness_pred = rf_happiness.predict(input_data)[0]
        
        # Display results with celebration
        st.balloons()
        st.markdown("---")
        st.subheader(" Your Personalized Results")
        st.caption(" Here's what our AI predicts based on your lifestyle!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stress_color = "" if stress_pred < 4 else "" if stress_pred < 7 else ""
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h2>{stress_color}</h2>
                <h3>{stress_pred:.1f}/10</h3>
                <p>Predicted Stress Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            happiness_color = "" if happiness_pred < 4 else "" if happiness_pred < 7 else ""
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h2>{happiness_color}</h2>
                <h3>{happiness_pred:.1f}/10</h3>
                <p>Predicted Happiness</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            overall_score = (10 - stress_pred + happiness_pred) / 2
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h2></h2>
                <h3>{overall_score:.1f}/10</h3>
                <p>Overall Wellness Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader(" Personalized Recommendations")
        
        recommendations = []
        if screen_time > 6:
            recommendations.append(" **Reduce screen time**: Your screen time is high. Try to limit it to under 5 hours daily.")
        if sleep_quality < 6:
            recommendations.append(" **Improve sleep quality**: Better sleep can significantly reduce stress and increase happiness.")
        if exercise_freq < 3:
            recommendations.append(" **Exercise more**: Aim for at least 3-4 sessions per week for better mental health.")
        if days_without_sm == 0:
            recommendations.append(" **Take social media breaks**: Try a weekly digital detox day.")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"<div class='insight-box'>{rec}</div>", unsafe_allow_html=True)
        else:
            st.success(" Great job! You're maintaining healthy habits!")

def show_analysis_results(df):
    st.header(" Comprehensive Analysis Results")
    
    # Load visualizations if they exist
    results_path = "results"
    
    tab1, tab2, tab3 = st.tabs([" Key Metrics", " Visualizations", " Detailed Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Correlation Analysis")
            st.metric("Screen Time ↔ Stress", "0.74", "Strong Positive")
            st.metric("Screen Time ↔ Happiness", "-0.71", "Strong Negative")
            st.metric("Sleep ↔ Stress", "-0.58", "Moderate Negative")
            st.metric("Sleep ↔ Happiness", "0.68", "Moderate Positive")
        
        with col2:
            st.subheader(" Model Performance")
            st.metric("Multiple Regression R²", "55.2%", "Stress Prediction")
            st.metric("Multiple Regression R²", "54.9%", "Happiness Prediction")
            st.metric("Random Forest R²", "48.1%", "Stress Model")
            st.metric("Random Forest R²", "49.4%", "Happiness Model")
    
    with tab2:
        st.subheader(" Generated Visualizations")
        
        viz_files = [
            "01_distributions.png",
            "02_categorical_distributions.png",
            "03_correlation_heatmap.png",
            "04_regression_results.png",
            "05_clustering_optimization.png",
            "06_kmeans_clusters.png",
            "07_dbscan_clusters.png",
            "08_feature_importance.png",
            "09_rf_predictions.png"
        ]
        
        for i in range(0, len(viz_files), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(f"{results_path}/{viz_files[i]}"):
                    st.image(f"{results_path}/{viz_files[i]}", use_container_width=True)
            
            with col2:
                if i + 1 < len(viz_files) and os.path.exists(f"{results_path}/{viz_files[i+1]}"):
                    st.image(f"{results_path}/{viz_files[i+1]}", use_container_width=True)
    
    with tab3:
        st.subheader(" Detailed Results Summary")
        
        if os.path.exists(f"{results_path}/final_results_summary.csv"):
            results_df = pd.read_csv(f"{results_path}/final_results_summary.csv")
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("Run the complete analysis to generate detailed results.")

def show_insights():
    st.header(" Key Insights & Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Major Findings")
        
        st.markdown("""
        <div class="insight-box">
            <h4>1. Screen Time is the Critical Factor </h4>
            <ul>
                <li><strong class="status-critical">0.74 correlation</strong> with stress (extremely strong)</li>
                <li><strong>90x more important</strong> than social media abstinence</li>
                <li>Recommendation: <strong>Reduce overall screen time to under 5 hours/day</strong></li>
            </ul>
        </div>
        
        <div class="insight-box">
            <h4>2. Two Distinct User Phenotypes </h4>
            <ul>
                <li><strong>Cluster 0 (High-stress):</strong> 6.85hrs screen time, poor sleep (5.22/10), stress 7.74/10</li>
                <li><strong>Cluster 1 (Well-balanced):</strong> 4.29hrs screen time, good sleep (7.32/10), stress 5.57/10</li>
                <li>Recommendation: <strong>Identify your cluster and follow targeted interventions</strong></li>
            </ul>
        </div>
        
        <div class="insight-box">
            <h4>3. Sleep Quality Matters Significantly </h4>
            <ul>
                <li><strong class="status-good">-59%</strong> correlation with stress</li>
                <li><strong class="status-good">+68%</strong> correlation with happiness</li>
                <li>Recommendation: <strong>Prioritize 7-8 hours of quality sleep</strong></li>
            </ul>
        </div>
        
        <div class="insight-box">
            <h4>4. Social Media Abstinence Has Minimal Direct Effect </h4>
            <ul>
                <li>Only <strong>-0.8%</strong> correlation with stress</li>
                <li>Only <strong>+6.4%</strong> correlation with happiness</li>
                <li>Effect is mediated by other factors (screen time, sleep, exercise)</li>
                <li>Recommendation: <strong>Focus on holistic approach, not just abstinence</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader(" Action Plan")
        
        st.markdown("""
        ### Daily Habits
        - ⏰ Limit screen time to 4-5 hours
        -  Get 7-8 hours quality sleep
        -  Exercise 3-4 times/week
        -  Practice digital detox (1 day/week)
        
        ### Mental Health Tracking
        -  Monitor stress levels
        -  Track happiness daily
        -  Review progress weekly
        
        ### Intervention Strategies
        -  Set screen time limits
        -  Establish sleep routine
        -  Increase social interaction
        -  Set realistic goals
        """)
        
        st.markdown("---")
        st.info(" **Remember**: Mental health is multifaceted. A holistic approach addressing screen time, sleep, exercise, and social connections is most effective!")

def show_realtime_tracker(df):
    st.header("⏱ Real-Time Mental Health Tracker")
    st.markdown('<span class="realtime-badge"> LIVE TRACKING</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Clear instructions
    st.info(" **Track your daily mental health and see trends over time!** Enter today's data below and watch your progress.")
    
    with st.expander(" Quick Guide - How Tracking Works"):
        st.markdown("""
        **It's Easy!**
        1.  Fill in today's information (left side)
        2.  Click "Save Entry" button
        3.  See success message and balloons!
        4.  Your charts update automatically (right side)
        5.  Download your data anytime
        
        **Why track daily?**
        -  See your improvement over time
        -  Identify what works for you
        -  Stay motivated with visual progress
        -  Make data-driven decisions
        
        **Pro Tip:** Track at the same time each day for consistency!
        """)
    
    # Initialize session state for tracking
    if 'tracking_data' not in st.session_state:
        st.session_state.tracking_data = []
    
    if 'daily_entry' not in st.session_state:
        st.session_state.daily_entry = {
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'screen_time': 5.0,
            'sleep_quality': 7,
            'stress_level': 5,
            'happiness': 6,
            'exercise': 2,
            'social_media_breaks': 0
        }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Today's Entry")
        st.caption(" Move the sliders to match how you feel right now")
        
        # Real-time input form
        with st.form("daily_tracker"):
            st.write("**How's your day going?**")
            
            screen_time = st.slider(" Screen Time Today (hours)", 0.0, 12.0, 
                                   float(st.session_state.daily_entry['screen_time']), 0.5,
                                   help="Total hours on phone, computer, TV")
            
            sleep_quality = st.slider(" Sleep Quality Last Night (1-10)", 1, 10, 
                                     st.session_state.daily_entry['sleep_quality'],
                                     help="How well did you sleep? 1=Terrible, 10=Perfect")
            
            stress_level = st.slider(" Stress Level Right Now (1-10)", 1, 10, 
                                    st.session_state.daily_entry['stress_level'],
                                    help="How stressed do you feel? 1=Calm, 10=Very stressed")
            
            happiness = st.slider(" Happiness Level Right Now (1-10)", 1, 10, 
                                 st.session_state.daily_entry['happiness'],
                                 help="How happy are you? 1=Very unhappy, 10=Very happy")
            
            exercise = st.slider(" Exercise Today (minutes)", 0, 120, 
                               st.session_state.daily_entry['exercise'] * 30, 15,
                               help="Total minutes of physical activity")
            
            social_media_breaks = st.number_input(" Social Media Breaks Taken", 
                                                 min_value=0, max_value=10, 
                                                 value=st.session_state.daily_entry['social_media_breaks'],
                                                 help="How many times did you take a break from social media?")
            
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                submit = st.form_submit_button(" Save Entry", use_container_width=True, type="primary",
                                              help="Click to save your entry!")
            with col_b:
                reset = st.form_submit_button(" Clear History", use_container_width=True,
                                             help="Delete all saved entries")
        
        if submit:
            entry = {
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'screen_time': screen_time,
                'sleep_quality': sleep_quality,
                'stress_level': stress_level,
                'happiness': happiness,
                'exercise': exercise // 30,
                'social_media_breaks': social_media_breaks
            }
            st.session_state.tracking_data.append(entry)
            st.session_state.daily_entry = entry
            st.success(" Entry saved successfully! Check the right side for your progress →")
            st.balloons()
        
        if reset:
            st.session_state.tracking_data = []
            st.info(" All tracking data has been cleared. Start fresh below!")
    
    with col2:
        st.subheader(" Your Progress Dashboard")
        
        if len(st.session_state.tracking_data) > 0:
            st.caption(" Your tracking data updates here automatically!")
            
            # Convert to DataFrame
            tracking_df = pd.DataFrame(st.session_state.tracking_data)
            
            # Current status
            latest = st.session_state.tracking_data[-1]
            
            st.markdown(f"** Total Entries:** {len(st.session_state.tracking_data)} days tracked")
            st.markdown("---")
            
            # Status cards
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #00d2ff 0%, #3a47d5 100%);">
                    <h3>{latest['stress_level']}/10</h3>
                    <p>Current Stress</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #ff0080 0%, #ff8c00 100%);">
                    <h3>{latest['happiness']}/10</h3>
                    <p>Current Happiness</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Trend analysis
            if len(tracking_df) >= 2:
                st.write("** Your Trends:**")
                
                stress_trend = tracking_df['stress_level'].iloc[-1] - tracking_df['stress_level'].iloc[0]
                happiness_trend = tracking_df['happiness'].iloc[-1] - tracking_df['happiness'].iloc[0]
                
                if stress_trend < 0:
                    st.success(f" Stress decreased by {abs(stress_trend):.1f} points")
                elif stress_trend > 0:
                    st.warning(f" Stress increased by {stress_trend:.1f} points")
                else:
                    st.info(" Stress level stable")
                
                if happiness_trend > 0:
                    st.success(f" Happiness increased by {happiness_trend:.1f} points")
                elif happiness_trend < 0:
                    st.warning(f" Happiness decreased by {abs(happiness_trend):.1f} points")
                else:
                    st.info(" Happiness level stable")
            
            # Progress chart
            if len(tracking_df) >= 2:
                st.markdown("---")
                st.write("** Progress Over Time:**")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(tracking_df))), y=tracking_df['stress_level'],
                                        mode='lines+markers', name='Stress', 
                                        line=dict(color='#ff0080', width=3)))
                fig.add_trace(go.Scatter(x=list(range(len(tracking_df))), y=tracking_df['happiness'],
                                        mode='lines+markers', name='Happiness',
                                        line=dict(color='#00d2ff', width=3)))
                
                fig.update_layout(title='Mental Health Progress',
                                xaxis_title='Entry #',
                                yaxis_title='Score (1-10)',
                                template='plotly_dark',
                                height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download tracking data
            st.markdown("---")
            csv = tracking_df.to_csv(index=False)
            st.download_button(
                label=" Download Your Tracking Data (CSV)",
                data=csv,
                file_name=f"mental_health_tracking_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download your data to keep permanent records or analyze in Excel"
            )
        else:
            st.info(" **Get Started!** Fill out the form on the left and click 'Save Entry'")
            st.markdown("""
            **What you'll see here once you start tracking:**
            -  Your current stress and happiness levels
            -  Trend charts showing improvement over time
            -  Arrows showing if you're getting better or worse
            -  Download button for your complete history
            
            **Start tracking now to see your progress!**
            """)

def show_future_predictions(df):
    st.header(" Future Mental Health Predictions")
    st.success(" **What if...?** See how lifestyle changes will impact your mental health!")
    
    # Clear instructions
    with st.expander(" How This Works - Click to Learn"):
        st.markdown("""
        **Simple 3-Step Process:**
        
        **Step 1:** Enter your CURRENT lifestyle habits (left column)
        - How you live now
        
        **Step 2:** Set your TARGET goals (right column)
        - How you WANT to live
        
        **Step 3:** Click "Predict Future Impact" button
        - See what happens if you make those changes!
        
        **You'll Get:**
        -  Before/After comparison
        -  Improvement predictions
        -  Personalized action steps
        -  4-week timeline
        -  Impact rating
        
        **Example:** "What if I reduce screen time from 8hrs to 4hrs?"
        → AI shows you the predicted stress reduction!
        """)
    
    st.markdown("---")
    
    # Train models
    with st.spinner(" Loading AI models..."):
        rf_stress, rf_happiness, le_gender, le_platform = train_models(df)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Your Current Lifestyle")
        st.caption(" Fill in how you live NOW")
        
        current_age = st.number_input("Age", 16, 50, 30, key="future_age",
                                     help="Your current age")
        current_gender = st.selectbox("Gender", df['Gender'].unique() if 'Gender' in df.columns else ['Male', 'Female', 'Other'], 
                                     key="future_gender",
                                     help="Select your gender")
        current_platform = st.selectbox("Primary Social Media", 
                                       df['Social_Media_Platform'].unique() if 'Social_Media_Platform' in df.columns else ['Instagram', 'Facebook'],
                                       key="future_platform",
                                       help="Which platform do you use most?")
        
        st.markdown("**Current Habits:**")
        current_screen = st.slider(" Current Screen Time (hrs)", 1.0, 12.0, 6.0, 0.5, key="future_current_screen",
                                  help="How many hours per day do you spend on screens NOW?")
        current_sleep = st.slider(" Current Sleep Quality", 1, 10, 6, key="future_current_sleep",
                                 help="How well do you sleep NOW? (1=Poor, 10=Excellent)")
        current_exercise = st.slider(" Current Exercise (times/week)", 0, 7, 2, key="future_current_exercise",
                                    help="How many times per week do you exercise NOW?")
        current_breaks = st.slider(" Current SM Breaks (days/month)", 0, 30, 0, key="future_current_breaks",
                                  help="How many social media break days NOW?")
    
    with col2:
        st.subheader(" Your Target Goals")
        st.caption(" Set your FUTURE goals - what you want to achieve!")
        st.markdown("---")
        st.markdown("**What if you made these changes?**")
        
        future_screen = st.slider(" Target Screen Time (hrs)", 1.0, 12.0, 4.0, 0.5, key="future_target_screen",
                                 help="Reduce screen time to this many hours")
        future_sleep = st.slider(" Target Sleep Quality", 1, 10, 8, key="future_target_sleep",
                                help="Improve sleep quality to this level")
        future_exercise = st.slider(" Target Exercise (times/week)", 0, 7, 4, key="future_target_exercise",
                                   help="Increase exercise to this many times per week")
        future_breaks = st.slider(" Target SM Breaks (days/month)", 0, 30, 7, key="future_target_breaks",
                                 help="Take this many social media break days")
        
        # Show the changes
        st.markdown("---")
        st.markdown("** Your Proposed Changes:**")
        if future_screen < current_screen:
            st.success(f" Screen time: -{current_screen - future_screen:.1f} hrs/day")
        if future_sleep > current_sleep:
            st.success(f" Sleep quality: +{future_sleep - current_sleep} points")
        if future_exercise > current_exercise:
            st.success(f" Exercise: +{future_exercise - current_exercise}x per week")
        if future_breaks > current_breaks:
            st.success(f" SM breaks: +{future_breaks - current_breaks} days/month")
    
    st.markdown("---")
    
    # Centered predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button(" Predict Future Impact", use_container_width=True, type="primary",
                                  help="Click to see how these changes will affect your mental health!")
    
    if predict_button:
        # Current predictions
        gender_encoded = le_gender.transform([current_gender])[0]
        platform_encoded = le_platform.transform([current_platform])[0]
        
        current_input = np.array([[current_age, gender_encoded, current_screen, current_sleep, 
                                  current_breaks, current_exercise, platform_encoded]])
        future_input = np.array([[current_age, gender_encoded, future_screen, future_sleep, 
                                 future_breaks, future_exercise, platform_encoded]])
        
        current_stress = rf_stress.predict(current_input)[0]
        current_happiness = rf_happiness.predict(current_input)[0]
        
        future_stress = rf_stress.predict(future_input)[0]
        future_happiness = rf_happiness.predict(future_input)[0]
        
        # Calculate improvements
        stress_change = current_stress - future_stress
        happiness_change = future_happiness - current_happiness
        
        st.markdown("---")
        st.subheader(" Prediction Results")
        
        # Comparison cards
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("### Current State")
            st.metric("Stress Level", f"{current_stress:.1f}/10", 
                     delta=None, delta_color="off")
            st.metric("Happiness", f"{current_happiness:.1f}/10", 
                     delta=None, delta_color="off")
        
        with col_b:
            st.markdown("### Future State")
            st.metric("Stress Level", f"{future_stress:.1f}/10", 
                     delta=f"{-stress_change:.1f}", delta_color="inverse")
            st.metric("Happiness", f"{future_happiness:.1f}/10", 
                     delta=f"{happiness_change:.1f}", delta_color="normal")
        
        with col_c:
            st.markdown("### Improvement")
            improvement_score = (stress_change + happiness_change) / 2
            
            if improvement_score > 2:
                st.success(" **Excellent Changes!**")
                st.write(f"Overall improvement: **{improvement_score:.1f} points**")
            elif improvement_score > 1:
                st.info(" **Good Changes!**")
                st.write(f"Overall improvement: **{improvement_score:.1f} points**")
            elif improvement_score > 0:
                st.warning(" **Modest Changes**")
                st.write(f"Overall improvement: **{improvement_score:.1f} points**")
            else:
                st.error(" **Changes may not help**")
                st.write("Consider different adjustments")
        
        # Detailed analysis
        st.markdown("---")
        st.subheader(" Impact Analysis")
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            # Stress comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Current', 'Future'], 
                                y=[current_stress, future_stress],
                                marker_color=['#ff0080', '#00d2ff'],
                                text=[f'{current_stress:.1f}', f'{future_stress:.1f}'],
                                textposition='auto'))
            fig.update_layout(title='Stress Level Comparison',
                            yaxis_title='Stress (1-10)',
                            template='plotly_dark',
                            height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_y:
            # Happiness comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Current', 'Future'], 
                                y=[current_happiness, future_happiness],
                                marker_color=['#ff8c00', '#00ff87'],
                                text=[f'{current_happiness:.1f}', f'{future_happiness:.1f}'],
                                textposition='auto'))
            fig.update_layout(title='Happiness Index Comparison',
                            yaxis_title='Happiness (1-10)',
                            template='plotly_dark',
                            height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader(" Personalized Action Steps")
        
        recommendations = []
        
        if future_screen < current_screen:
            time_saved = (current_screen - future_screen) * 30  # days in month
            recommendations.append(f" **Reduce screen time**: You'll save {time_saved:.0f} hours/month!")
        
        if future_sleep > current_sleep:
            recommendations.append(f" **Improve sleep**: Your stress could decrease by {stress_change:.1f} points")
        
        if future_exercise > current_exercise:
            extra_sessions = (future_exercise - current_exercise) * 4  # weeks
            recommendations.append(f" **Exercise more**: {extra_sessions} extra sessions per month")
        
        if future_breaks > current_breaks:
            recommendations.append(f" **Take more breaks**: {future_breaks} detox days per month")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"<div class='insight-box'>{i}. {rec}</div>", unsafe_allow_html=True)
        
        if stress_change > 2 or happiness_change > 2:
            st.success(" **High Impact Changes!** These modifications could significantly improve your mental health!")
        
        # Timeline
        st.markdown("---")
        st.subheader(" Expected Timeline")
        
        timeline_data = {
            'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            'Stress_Reduction': [
                current_stress - (stress_change * 0.2),
                current_stress - (stress_change * 0.4),
                current_stress - (stress_change * 0.7),
                future_stress
            ],
            'Happiness_Increase': [
                current_happiness + (happiness_change * 0.2),
                current_happiness + (happiness_change * 0.4),
                current_happiness + (happiness_change * 0.7),
                future_happiness
            ]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline_data['Week'], y=timeline_data['Stress_Reduction'],
                                mode='lines+markers', name='Projected Stress',
                                line=dict(color='#00d2ff', width=3)))
        fig.add_trace(go.Scatter(x=timeline_data['Week'], y=timeline_data['Happiness_Increase'],
                                mode='lines+markers', name='Projected Happiness',
                                line=dict(color='#00ff87', width=3)))
        
        fig.update_layout(title='4-Week Projection',
                        yaxis_title='Score (1-10)',
                        template='plotly_dark',
                        height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(" **Note**: These are AI-generated predictions based on statistical models. Individual results may vary. Consistency is key!")

def show_insights():
    st.header(" Key Insights & Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Major Findings")
        
        st.markdown("""
        <div class="insight-box">
            <h4>1. Screen Time is the Critical Factor </h4>
            <ul>
                <li><strong>0.74 correlation</strong> with stress (extremely strong)</li>
                <li><strong>90x more important</strong> than social media abstinence</li>
                <li>Recommendation: <strong>Reduce overall screen time to under 5 hours/day</strong></li>
            </ul>
        </div>
        
        <div class="insight-box">
            <h4>2. Two Distinct User Phenotypes </h4>
            <ul>
                <li><strong>Cluster 0 (High-stress):</strong> 6.85hrs screen time, poor sleep (5.22/10), stress 7.74/10</li>
                <li><strong>Cluster 1 (Well-balanced):</strong> 4.29hrs screen time, good sleep (7.32/10), stress 5.57/10</li>
                <li>Recommendation: <strong>Identify your cluster and follow targeted interventions</strong></li>
            </ul>
        </div>
        
        <div class="insight-box">
            <h4>3. Sleep Quality Matters Significantly </h4>
            <ul>
                <li><strong>-59%</strong> correlation with stress</li>
                <li><strong>+68%</strong> correlation with happiness</li>
                <li>Recommendation: <strong>Prioritize 7-8 hours of quality sleep</strong></li>
            </ul>
        </div>
        
        <div class="insight-box">
            <h4>4. Social Media Abstinence Has Minimal Direct Effect </h4>
            <ul>
                <li>Only <strong>-0.8%</strong> correlation with stress</li>
                <li>Only <strong>+6.4%</strong> correlation with happiness</li>
                <li>Effect is mediated by other factors (screen time, sleep, exercise)</li>
                <li>Recommendation: <strong>Focus on holistic approach, not just abstinence</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader(" Action Plan")
        
        st.markdown("""
        ### Daily Habits
        - ⏰ Limit screen time to 4-5 hours
        -  Get 7-8 hours quality sleep
        -  Exercise 3-4 times/week
        -  Practice digital detox (1 day/week)
        
        ### Mental Health Tracking
        -  Monitor stress levels
        -  Track happiness daily
        -  Review progress weekly
        
        ### Intervention Strategies
        -  Set screen time limits
        -  Establish sleep routine
        -  Increase social interaction
        -  Set realistic goals
        """)
        
        st.markdown("---")
        st.info(" **Remember**: Mental health is multifaceted. A holistic approach addressing screen time, sleep, exercise, and social connections is most effective!")

if __name__ == "__main__":
    main()
