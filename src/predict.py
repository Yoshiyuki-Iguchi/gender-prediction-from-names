import pandas as pd
import numpy as np
from .features import extract_features


def predict_name_gender(name: str, df: pd.DataFrame, rfc_model, rfr_model, low_thresh: float, high_thresh: float) -> dict:
    """Predict gender for a given first name."""
    name_clean = name.strip()
    name_lookup = name_clean.title()
    known = df[df['Name'] == name_lookup]
    if len(known) == 1:
        row = known.iloc[0]
        p_male_est = float(row['p_male'])
        pred_label = 1 if p_male_est > 0.5 else 0
        pred_proba = p_male_est
        source = f"dataset lookup (n={int(row['total_count']):,})"
    else:
        median_count = float(df['total_count'].median())
        X_input = extract_features(pd.Series([name_clean]), pd.Series([median_count]))
        pred_label = rfc_model.predict(X_input)[0]
        pred_proba = rfc_model.predict_proba(X_input)[0, 1]
        p_male_est = float(np.clip(rfr_model.predict(X_input)[0], 0, 1))
        source = "character-level model (name not in dataset)"
    if p_male_est < low_thresh:
        category = 'Strong Female'
    elif p_male_est > high_thresh:
        category = 'Strong Male'
    else:
        category = 'Unisex'
    gender = 'Male' if pred_label == 1 else 'Female'
    confidence = pred_proba if pred_label == 1 else 1 - pred_proba
    warning = ''
    if confidence < 0.60:
        warning = '⚠️ Low confidence — this name is ambiguous.'
    elif category == 'Unisex':
        warning = 'ℹ️ Name is in the Unisex range — prediction may be unreliable.'
    return {
        'name': name_clean, 'predicted_gender': gender, 'confidence': f'{confidence:.1%}',
        'p_male_estimate': f'{p_male_est:.3f}', 'category': category, 'source': source, 'warning': warning,
    }


def print_prediction(result: dict):
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