import pandas as pd
import numpy as np
from .features import extract_features


def predict_name_gender(name: str, df: pd.DataFrame, rfc_model, rfr_model, low_thresh: float, high_thresh: float) -> dict:
    """
    Predict gender for a given first name.
    
    Two-path strategy:
        1. If name exists in dataset: use actual P(male) and total_count
        2. If name is unknown: fall back to character-level RF models
    
    Parameters
    ----------
    name : str
        First name to predict
    df : pd.DataFrame
        Cleaned dataset with Name, count_m, count_f, total_count, p_male columns
    rfc_model : RandomForestClassifier
        Trained classifier for binary prediction
    rfr_model : RandomForestRegressor
        Trained regressor for P(male) estimation
    low_thresh : float
        GMM threshold for Female/Unisex boundary
    high_thresh : float
        GMM threshold for Unisex/Male boundary
    
    Returns
    -------
    dict
        Prediction results with name, gender, confidence, category, source, warning
    """
    name_clean = name.strip()
    name_lookup = name_clean.title()
    
    # ---- 1. Known-name fast path ----
    known = df[df['Name'] == name_lookup]
    if len(known) == 1:
        row = known.iloc[0]
        p_male_est = float(row['p_male'])
        pred_label = 1 if p_male_est > 0.5 else 0
        pred_proba = p_male_est
        source_label = f"dataset lookup (n={int(row['total_count']):,})"
    
    # ---- 2. Unknown-name fallback ----
    else:
        median_count = float(df['total_count'].median())
        name_series = pd.Series([name_clean])
        count_series = pd.Series([median_count])
        X_input = extract_features(name_series, count_series)
        
        pred_label = rfc_model.predict(X_input)[0]
        pred_proba = rfc_model.predict_proba(X_input)[0, 1]
        p_male_est = float(np.clip(rfr_model.predict(X_input)[0], 0, 1))
        source_label = "character-level model (name not in dataset)"
    
    # ---- 3. Category using GMM thresholds ----
    if p_male_est < low_thresh:
        category = 'Strong Female'
    elif p_male_est > high_thresh:
        category = 'Strong Male'
    else:
        category = 'Unisex'
    
    # ---- 4. Gender and confidence ----
    gender = 'Male' if pred_label == 1 else 'Female'
    
    # Confidence: probability of the predicted class (0-1 range)
    if pred_label == 1:
        confidence = pred_proba
    else:
        confidence = 1 - pred_proba
    
    # Clip to valid range (0-1) for safety
    confidence = max(0.0, min(1.0, confidence))
    
    # ---- 5. Warning messages ----
    warning = ''
    if confidence < 0.60:
        warning = '⚠️ Low confidence — this name is ambiguous.'
    elif category == 'Unisex':
        warning = 'ℹ️ Name is in the Unisex range — prediction may be unreliable.'
    
    # ---- 6. Return result ----
    return {
        'name': name_clean,
        'predicted_gender': gender,
        'confidence': f'{confidence:.1%}',
        'p_male_estimate': f'{p_male_est:.3f}',
        'category': category,
        'source': source_label,
        'warning': warning,
    }


def print_prediction(result: dict):
    """
    Pretty-print prediction results to console.
    
    Parameters
    ----------
    result : dict
        Prediction result from predict_name_gender()
    """
    print("─" * 40)
    print(f"  Name              : {result['name']}")
    print(f"  Predicted Gender  : {result['predicted_gender']}")
    print(f"  Confidence        : {result['confidence']}")
    print(f"  P(male) estimate  : {result['p_male_estimate']}")
    print(f"  Category          : {result['category']}")
    print(f"  Source            : {result['source']}")
    if result['warning']:
        print(f"  {result['warning']}")
    print("─" * 40)
