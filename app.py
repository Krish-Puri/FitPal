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

