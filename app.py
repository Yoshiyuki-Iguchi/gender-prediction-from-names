"""
Gender Prediction Web App
Streamlit interface for the gender prediction model
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import zipfile
import requests
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_clean_data
from src.features import extract_features
from src.predict import predict_name_gender

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/yyy998181/genderbyname/resolve/main/models.zip"

# Page configuration
st.set_page_config(
    page_title="Gender Predictor",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { text-align: center; padding: 20px 0; }
    .main-header h1 { font-size: 2.5em; margin-bottom: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header p { color: #666; font-size: 1.1em; }
    .stTextInput > div > div > input { font-size: 18px; padding: 12px 16px; border-radius: 10px; border: 2px solid #e0e0e0; }
    .stTextInput > div > div > input:focus { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2); }
    .stButton > button { border-radius: 10px; font-weight: 600; font-size: 16px; padding: 12px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; width: 100%; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
    .footer { text-align: center; color: #999; font-size: 0.85em; padding: 30px 0 10px 0; border-top: 1px solid #eee; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>👤 Gender Predictor</h1>
    <p>Predict gender from a first name with confidence score</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown("""
    This tool predicts gender from first names using machine learning.

    **How it works:**
    1. **Dataset lookup** - If the name exists in our dataset (99,683 names), we use the actual statistics
    2. **Character model** - If the name is unknown, we use a Random Forest model with 65 character-level features

    **Model Performance:**
    - **Overall Accuracy:** 86.1%
    - **AUC-ROC:** 0.9295
    - **Calibration Slope:** 0.9484 (near ideal 1.0)
    """)
    st.divider()
    
    st.subheader("📊 Accuracy by Category")
    st.markdown("""
    | Category | Accuracy |
    |----------|----------|
    | Strong Female | **91.1%** ✅ |
    | Strong Male | **82.0%** ✅ |
    | Unisex | **58.8%** ⚠️ |
    
    **Why Unisex accuracy is lower:**
    Names with P(male) between 0.04 and 0.85 are truly ambiguous. 
    The model is **honest about uncertainty** — it flags these with low confidence and a warning.
    This is a fundamental data limitation, not a model failure.
    """)
    st.divider()
    st.caption("**Dataset:** US Social Security, UK ONS, British Columbia, Australian Government")

# Load models with caching
@st.cache_resource
def load_models():
    """
    Download models from Hugging Face and load them into memory.
    First run: downloads the 132MB zip file (takes 1-3 minutes).
    Subsequent runs: loads from cache (1-2 seconds).
    """
    
    # Check if models already exist locally
    if not os.path.exists('models/rfc_model.pkl'):
        with st.spinner('📥 Downloading model from Hugging Face... (first time only, may take a few minutes)'):
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Download the zip file
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                # Show download progress
                progress_bar = st.progress(0)
                with open('models.zip', 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(min(downloaded / total_size, 1.0))
                
                # Extract the zip file
                with zipfile.ZipFile('models.zip', 'r') as zip_ref:
                    zip_ref.extractall('.')
                os.remove('models.zip')
                st.success('✅ Model download complete!')
                
            except Exception as e:
                st.error(f"❌ Download error: {str(e)}")
                st.info("💡 Please check your internet connection and try again.")
                st.stop()
    
    # Load models from disk
    try:
        rfc_model = joblib.load('models/rfc_model.pkl')
        rfr_model = joblib.load('models/rfr_model.pkl')
        low_thresh, high_thresh = joblib.load('models/thresholds.pkl')
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.stop()
    
    # Load the dataset
    df = load_and_clean_data('data/name_gender_dataset.csv')
    
    return df, low_thresh, high_thresh, rfc_model, rfr_model

# Load models with spinner
with st.spinner("🔄 Loading models..."):
    df, low_thresh, high_thresh, rfc_model, rfr_model = load_models()
st.success("✅ Models loaded successfully!")

# Main input section
st.divider()
st.subheader("🔮 Predict Gender")

# Input row with predict button
col1, col2 = st.columns([3, 1])
with col1:
    name = st.text_input(
        "Enter a first name:",
        placeholder="e.g., Casey, Jordan, James",
        key="name_input",
        label_visibility="collapsed"
    )
with col2:
    st.write("")
    st.write("")
    predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

# Example name buttons
st.caption("💡 Try these examples:")
sample_cols = st.columns(6)
sample_names = ["James", "Mary", "Casey", "Jordan", "Riley", "Quinn"]
for idx, (col, sample) in enumerate(zip(sample_cols, sample_names)):
    if col.button(sample, key=f"sample_{idx}", use_container_width=True):
        name = sample
        predict_clicked = True

# Prediction logic
if name and predict_clicked:
    try:
        result = predict_name_gender(
            name=name, df=df, rfc_model=rfc_model, rfr_model=rfr_model,
            low_thresh=low_thresh, high_thresh=high_thresh
        )
        st.divider()
        
        # Display results in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Gender
        with col1:
            gender_emoji = "♂️" if result['predicted_gender'] == "Male" else "♀️"
            gender_color = "#2196F3" if result['predicted_gender'] == "Male" else "#E91E63"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {gender_color}15; border-radius: 12px; border: 2px solid {gender_color};">
                <div style="font-size: 2.5em;">{gender_emoji}</div>
                <div style="font-size: 0.85em; color: #666;">Gender</div>
                <div style="font-size: 1.5em; font-weight: bold; color: {gender_color};">{result['predicted_gender']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence
        with col2:
            conf_value = float(result['confidence'].replace('%', ''))
            if conf_value >= 70:
                conf_color = "#4CAF50"
                conf_icon = "✅"
            elif conf_value >= 50:
                conf_color = "#FF9800"
                conf_icon = "⚠️"
            else:
                conf_color = "#f44336"
                conf_icon = "❌"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f5f5f5; border-radius: 12px;">
                <div style="font-size: 2em;">{conf_icon}</div>
                <div style="font-size: 0.85em; color: #666;">Confidence</div>
                <div style="font-size: 1.5em; font-weight: bold; color: {conf_color};">{result['confidence']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # P(male) estimate
        with col3:
            pmale_value = float(result['p_male_estimate'])
            if pmale_value > 0.5:
                pmale_color = "#2196F3"
            elif pmale_value < 0.4:
                pmale_color = "#E91E63"
            else:
                pmale_color = "#FF9800"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f5f5f5; border-radius: 12px;">
                <div style="font-size: 2em;">📈</div>
                <div style="font-size: 0.85em; color: #666;">P(Male)</div>
                <div style="font-size: 1.5em; font-weight: bold; color: {pmale_color};">{result['p_male_estimate']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Category
        with col4:
            if result['category'] == "Strong Male":
                cat_color = "#2196F3"
                cat_icon = "🟢"
            elif result['category'] == "Strong Female":
                cat_color = "#E91E63"
                cat_icon = "🟣"
            else:
                cat_color = "#FF9800"
                cat_icon = "🟠"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f5f5f5; border-radius: 12px;">
                <div style="font-size: 2em;">{cat_icon}</div>
                <div style="font-size: 0.85em; color: #666;">Category</div>
                <div style="font-size: 1.1em; font-weight: bold; color: {cat_color};">{result['category']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display warnings
        if result['warning']:
            if "⚠️" in result['warning']:
                st.warning(result['warning'])
            else:
                st.info(result['warning'])
        
        # Detailed information (collapsible)
        with st.expander("📋 Detailed Information"):
            st.markdown(f"**Name:** {result['name']}")
            st.markdown(f"**Source:** {result['source']}")
            st.markdown(f"**P(male) estimate:** {result['p_male_estimate']}")
            st.markdown(f"**GMM Thresholds:** {low_thresh:.3f} / {high_thresh:.3f}")
            
            # Show dataset statistics if name exists in dataset
            known = df[df['Name'] == result['name'].title()]
            if len(known) == 1:
                row = known.iloc[0]
                st.markdown(f"**Total occurrences:** {int(row['total_count']):,}")
                st.markdown(f"**Male count:** {int(row['count_m']):,}")
                st.markdown(f"**Female count:** {int(row['count_f']):,}")
        
    except Exception as e:
        st.error(f"❌ Error predicting '{name}': {str(e)}")

# Footer
st.divider()
st.markdown("""
<div class="footer">
    Master's Programme — Introduction to Machine Learning Methods and Data Mining · May 2026
</div>
""", unsafe_allow_html=True)
