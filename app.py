"""
Fitness Training Optimizer - Streamlit Application
MULTI-TARGET VERSION
Uses 6 trained ML models for comprehensive recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Fitness Training Optimizer",
    page_icon="💪",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model_package = joblib.load('final_fitness_models_multi.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please upload 'final_fitness_models_multi.pkl'")
        st.info("Run the complete ML pipeline in Google Colab first.")
        return None

model_package = load_models()

# Title
st.markdown('<h1 class="main-header">💪 AI Fitness Training Optimizer</h1>', unsafe_allow_html=True)
st.markdown("### Get comprehensive training recommendations powered by 6 specialized ML models")

if model_package is None:
    st.stop()

# Extract components
models = model_package['models']
scaler = model_package['scaler']
le_split = model_package['label_encoder_split']
feature_names = model_package['feature_names']
target_names = model_package['target_names']

# Display model info
with st.expander("ℹ️ About This AI System"):
    st.markdown("### 6 Specialized Models Working Together:")
    
    cols = st.columns(2)
    for idx, (target_name, model_info) in enumerate(models.items()):
        col = cols[idx % 2]
        with col:
            st.markdown(f"**{target_name.replace('_', ' ').title()}**")
            st.caption(f"Model: {model_info['model_name']}")
            perf = model_info['performance']
            if perf['test_f1']:
                st.caption(f"F1 Score: {perf['test_f1']:.3f}")
            else:
                st.caption(f"R² Score: {perf['test_r2']:.3f}")
    
    st.caption(f"🗓️ Trained: {model_package['training_date']}")
    st.caption(f"📊 Dataset: {model_package['dataset_size']:,} samples")

# Sidebar for user inputs
st.sidebar.header("📊 Your Current Stats")

# Basic Information
st.sidebar.subheader("Basic Information")
age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=25, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
height = st.sidebar.number_input("Height (cm)", min_value=140, max_value=220, value=175, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=40, max_value=200, value=75, step=1)
body_fat = st.sidebar.slider("Body Fat %", min_value=5, max_value=45, value=15, step=1)

st.sidebar.subheader("Current Strength Levels")
st.sidebar.caption("Enter your 1-rep max or estimated max (kg)")
bench_press = st.sidebar.number_input("Bench Press (kg)", min_value=20, max_value=300, value=80, step=5)
squat = st.sidebar.number_input("Squat (kg)", min_value=30, max_value=400, value=100, step=5)
deadlift = st.sidebar.number_input("Deadlift (kg)", min_value=40, max_value=450, value=120, step=5)

st.sidebar.subheader("Activity & Experience")
daily_steps = st.sidebar.number_input("Average Daily Steps", min_value=1000, max_value=20000, value=8000, step=500)
training_exp = st.sidebar.number_input("Training Experience (years)", min_value=0, max_value=30, value=2, step=1)

st.sidebar.subheader("🎯 Your Goals")
st.sidebar.caption("Select one or multiple")
goal_muscle_gain = st.sidebar.checkbox("Muscle Gain 🏋️")
goal_fat_loss = st.sidebar.checkbox("Fat Loss 🔥")
goal_strength = st.sidebar.checkbox("Strength 💪", value=True)
goal_general_fitness = st.sidebar.checkbox("General Fitness 🏃")

if not any([goal_muscle_gain, goal_fat_loss, goal_strength, goal_general_fitness]):
    st.sidebar.warning("⚠️ Please select at least one goal!")
    st.stop()

# Calculate derived metrics
bmi = weight / ((height / 100) ** 2)
total_lifted = bench_press + squat + deadlift
strength_to_weight = total_lifted / weight

# Main metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("BMI", f"{bmi:.1f}")
with col2:
    st.metric("Total Lifted", f"{total_lifted:.0f} kg")
with col3:
    st.metric("Strength/Weight", f"{strength_to_weight:.2f}")
with col4:
    st.metric("Experience", f"{training_exp} years")

# Prediction button
if st.sidebar.button("🚀 Generate Complete Training Plan", type="primary", use_container_width=True):
    
    # Prepare features
    user_features = pd.DataFrame([{
        'age': age,
        'height': height,
        'weight': weight,
        'body_fat': body_fat,
        'bench_press': bench_press,
        'squat': squat,
        'deadlift': deadlift,
        'daily_steps': daily_steps,
        'training_exp': training_exp,
        'bmi': bmi,
        'total_lifted': total_lifted,
        'strength_to_weight': strength_to_weight,
        'goal_muscle_gain': 1 if goal_muscle_gain else 0,
        'goal_fat_loss': 1 if goal_fat_loss else 0,
        'goal_strength': 1 if goal_strength else 0,
        'goal_general_fitness': 1 if goal_general_fitness else 0,
        'gender_encoded': 1 if gender == "Male" else 0
    }])

    user_features = user_features[feature_names]
    user_features_scaled = scaler.transform(user_features)
    
    # Make predictions for all 6 targets
    predictions = {}
    
    # Training Split (Classification)
    split_model = models['training_split']['model']
    split_pred_encoded = split_model.predict(user_features_scaled)[0]
    split_proba = split_model.predict_proba(user_features_scaled)[0]
    predicted_split = le_split.inverse_transform([split_pred_encoded])[0]
    split_confidence = split_proba[split_pred_encoded] * 100
    
    predictions['training_split'] = {
        'value': predicted_split,
        'confidence': split_confidence,
        'probabilities': dict(zip(le_split.classes_, split_proba * 100))
    }
