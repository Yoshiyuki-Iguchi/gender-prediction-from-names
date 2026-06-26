import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def train_models(features: dict, random_state: int = 42) -> dict:
    """Train and tune all three models."""
    X_train_bin, X_val_bin = features['X_train_bin'], features['X_val_bin']
    X_train_all, X_val_all = features['X_train_all'], features['X_val_all']
    y_train_bin, y_val_bin = features['y_train_bin'], features['y_val_bin']
    y_train_reg, y_val_reg = features['y_train_reg'], features['y_val_reg']
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    gs = GridSearchCV(lr, {'C': [0.01, 0.1, 1, 10, 100]}, cv=3, scoring='f1')
    gs.fit(X_val_bin, y_val_bin)
    best_lr = gs.best_estimator_.fit(X_train_bin, y_train_bin)
    
    # Random Forest Classifier
    rfc = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    gs = GridSearchCV(rfc, {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}, cv=3, scoring='f1')
    gs.fit(X_val_bin, y_val_bin)
    best_rfc = gs.best_estimator_.fit(X_train_bin, y_train_bin)
    
    # Random Forest Regressor
    rfr = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    gs = GridSearchCV(rfr, {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}, cv=3, scoring='neg_mean_absolute_error')
    gs.fit(X_val_all, y_val_reg)
    best_rfr = gs.best_estimator_.fit(X_train_all, y_train_reg)
    
    return {
        'logistic_regression': {'model': best_lr, 'val_accuracy': accuracy_score(y_val_bin, best_lr.predict(X_val_bin))},
        'random_forest_classifier': {'model': best_rfc, 'val_accuracy': accuracy_score(y_val_bin, best_rfc.predict(X_val_bin))},
        'random_forest_regressor