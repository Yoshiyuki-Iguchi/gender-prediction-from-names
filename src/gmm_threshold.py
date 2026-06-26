import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
from typing import Tuple


def get_thresholds_and_categories(df: pd.DataFrame, random_state: int = 42) -> Tuple[float, float, pd.DataFrame]:
    """Fit GMM on mixed-gender names and assign categories."""
    p_mixed = df[(df['count_m'] > 0) & (df['count_f'] > 0)]['p_male'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_state, max_iter=500)
    gmm.fit(p_mixed)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]
    print(f"GMM: μ0={means[0]:.4f}, μ1={means[1]:.4f}")
    midpoint = (means[0] + means[1]) / 2
    def density_diff(x):
        return weights[0] * norm.pdf(x, means[0], stds[0]) - weights[1] * norm.pdf(x, means[1], stds[1])
    low_thresh = brentq(density_diff, 0.001, midpoint)
    high_thresh = brentq(density_diff, midpoint, 0.999)
    print(f"Thresholds: {low_thresh:.4f} / {high_thresh:.4f}")
    df = df.copy()
    df['Name_Type'] = df['p_male'].apply(lambda p: 'Strong Female' if p < low_thresh else 'Strong Male' if p > high_thresh else 'Unisex')
    return low_thresh, high_thresh, df