# Gender Prediction from First Names

> A probabilistic approach to gender prediction using GMM-based unisex detection and Random Forest models.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/yourusername/gender-prediction-from-names/blob/main/notebooks/predictor_demo.ipynb)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Try it now! (No installation required)

| Platform | Link | Description |
|----------|------|-------------|
| **\CID{632} Web App** | [Click here](https://your-app.streamlit.app) | **Easiest!** Just enter a name |
| **Google Colab** | [Open In Colab](https://colab.research.google.com/github/yourusername/gender-prediction-from-names/blob/main/notebooks/predictor_demo.ipynb) | Run in browser - no setup |
| **Jupyter Notebook** | [View on GitHub](notebooks/gender_prediction_updated_V2.ipynb) | Full analysis with all code |

---

## Overview

This project builds a machine learning system that predicts gender from a first name using character-level features.

**Key innovations:**
- **Data-driven unisex detection** using 2-component Gaussian Mixture Model
- **Calibrated confidence scores** via Random Forest Regressor (slope 0.948)
- **Two-path prediction** (dataset lookup + character model)
- **Interactive Web App** with real-time predictions

---

## Key Results

| Category | Accuracy | N (Test) | Interpretation |
|----------|----------|----------|----------------|
| **Strong Male** | 82.0% | 5,181 | Model works reliably \UTF{2705} |
| **Strong Female** | 91.1% | 8,923 | Model works reliably \UTF{2705} |
| **Unisex** | 58.8% | 849 | Near-random - fundamental limit \CID{220} |

**Overall:** 86.1% Accuracy \UTF{00B7} 0.9295 AUC-ROC \UTF{00B7} 0.9484 Calibration Slope

> \CID{220} The accuracy collapse on unisex names is **NOT a model failure** - it's a fundamental property of the data. Names that are genuinely ambiguous cannot be resolved with character features alone.

---

## Methodology

### 1. Data Cleaning
- 147,269 raw rows → 99,683 unique names
- Removed: missing values, duplicates, dirty tokens, rare names

### 2. GMM Thresholding
- Fit 2-component GMM **only on mixed-gender names**
- Thresholds: **0.0407** (Female/Unisex) and **0.8500** (Unisex/Male)
- Result: 5,657 unisex names (5.7% of corpus)

### 3. Feature Engineering (65 features)
- First/last letter one-hot (26 + 26)
- Suffix flags (-son, -lyn, -ette, -ie, -ine)
- Vowel ratio, name length, log frequency
- Deterministic ord-based suffix encoding

### 4. Models

| Model | Task | Best Params | Accuracy |
|-------|------|-------------|----------|
| Logistic Regression | Binary | C=10 | 80.0% |
| **RF Classifier** | Binary | n=100, depth=None | **86.1%** |
| RF Regressor | P(male) | n=100, depth=None | MAE=0.1775 |

---

## Quick Start

### Option 1: Web App (Easiest)
Just visit: [https://your-app.streamlit.app](https://your-app.streamlit.app)

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gender-prediction-from-names.git
cd gender-prediction-from-names

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Or run the Jupyter notebook
jupyter notebook notebooks/gender_prediction_updated_V2.ipynb