
---

### 2. `app.py`

```python
"""
Gender Prediction Web App
Streamlit interface for the gender prediction model
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_clean_data, prepare_splits
from src.gmm_threshold import get_thresholds_and_categories
from src.features import build_feature_matrices
from src.model import train_models
from src.predict import predict_name_gender

# Page Configuration
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
    """)
    st.divider()
    st.caption("**Dataset:** US Social Security, UK ONS, British Columbia, Australian Government")

# Load Models (Cached)
@st.cache_resource
def load_models():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading data...")
    progress_bar.progress(20)
    df = load_and_clean_data('data/name_gender_dataset.csv')
    
    status_text.text("Training GMM for unisex detection...")
    progress_bar.progress(40)
    low_thresh, high_thresh, df = get_thresholds_and_categories(df)
    
    status_text.text("Creating data splits...")
    progress_bar.progress(60)
    df_train, df_val, df_test = prepare_splits(df)
    
    status_text.text("Building feature matrices...")
    progress_bar.progress(70)
    features = build_feature_matrices(df_train, df_val, df_test)
    
    status_text.text("Training models (this may take 30-60 seconds)...")
    progress_bar.progress(80)
    models = train_models(features)
    
    progress_bar.progress(100)
    status_text.text("✅ All models ready!")
    
    rfc_model = models['random_forest_classifier']['model']
    rfr_model = models['random_forest_regressor']['model']
    model_results = {
        'rfc_accuracy': models['random_forest_classifier']['val_accuracy'],
        'rfr_mae': models['random_forest_regressor']['val_mae'],
        'rfr_r2': models['random_forest_regressor']['val_r2'],
    }
    return df, low_thresh, high_thresh, rfc_model, rfr_model, model_results

# Load models
try:
    with st.spinner("🔄 Loading data and training models... (30-60 seconds)"):
        df, low_thresh, high_thresh, rfc_model, rfr_model, model_results = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# Main Input
st.divider()
st.subheader("🔮 Predict Gender")

col1, col2 = st.columns([3, 1])
with col1:
    name = st.text_input("Enter a first name:", placeholder="e.g., Casey, Jordan, James", key="name_input", label_visibility="collapsed")
with col2:
    st.write("")
    st.write("")
    predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

st.caption("💡 Try these examples:")
sample_cols = st.columns(6)
sample_names = ["James", "Mary", "Casey", "Jordan", "Riley", "Quinn"]
for idx, (col, sample) in enumerate(zip(sample_cols, sample_names)):
    if col.button(sample, key=f"sample_{idx}", use_container_width=True):
        name = sample
        predict_clicked = True

# Prediction
if name and predict_clicked:
    try:
        result = predict_name_gender(
            name=name, df=df, rfc_model=rfc_model, rfr_model=rfr_model,
            low_thresh=low_thresh, high_thresh=high_thresh
        )
        st.divider()
        
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
            conf_color = "#4CAF50" if conf_value >= 70 else "#FF9800" if conf_value >= 50 else "#f44336"
            conf_icon = "✅" if conf_value >= 70 else "⚠️" if conf_value >= 50 else "❌"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f5f5f5; border-radius: 12px;">
                <div style="font-size: 2em;">{conf_icon}</div>
                <div style="font-size: 0.85em; color: #666;">Confidence</div>
                <div style="font-size: 1.5em; font-weight: bold; color: {conf_color};">{result['confidence']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # P(male)
        with col3:
            pmale_value = float(result['p_male_estimate'])
            pmale_color = "#2196F3" if pmale_value > 0.5 else "#E91E63" if pmale_value < 0.4 else "#FF9800"
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
                cat_color, cat_icon = "#2196F3", "🟢"
            elif result['category'] == "Strong Female":
                cat_color, cat_icon = "#E91E63", "🟣"
            else:
                cat_color, cat_icon = "#FF9800", "🟠"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f5f5f5; border-radius: 12px;">
                <div style="font-size: 2em;">{cat_icon}</div>
                <div style="font-size: 0.85em; color: #666;">Category</div>
                <div style="font-size: 1.1em; font-weight: bold; color: {cat_color};">{result['category']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if result['warning']:
            if "⚠️" in result['warning']:
                st.warning(result['warning'])
            else:
                st.info(result['warning'])
        
        with st.expander("📋 Detailed Information"):
            st.markdown(f"**Name:** {result['name']}")
            st.markdown(f"**Source:** {result['source']}")
            st.markdown(f"**P(male) estimate:** {result['p_male_estimate']}")
            st.markdown(f"**GMM Thresholds:** {low_thresh:.3f} / {high_thresh:.3f}")
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
    Built with ❤️ using scikit-learn, pandas, and Streamlit<br>
    Master's Programme — Introduction to Machine Learning Methods · May 2026
</div>
""", unsafe_allow_html=True)