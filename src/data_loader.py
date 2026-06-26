import re
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_and_clean_data(filepath: str, min_count: int = 5) -> pd.DataFrame:
    """Complete data loading and cleaning pipeline."""
    df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}")
    
    # Clean
    df = df.dropna()
    df = df.groupby(['Name', 'Gender'], as_index=False).agg({'Count': 'sum', 'Probability': 'mean'})
    df = df[df['Gender'].isin(['M', 'F'])]
    df = df[df['Count'] > 0]
    
    # Pivot
    male = df[df['Gender'] == 'M'][['Name', 'Count']].rename(columns={'Count': 'count_m'})
    female = df[df['Gender'] == 'F'][['Name', 'Count']].rename(columns={'Count': 'count_f'})
    df = pd.merge(male, female, on='Name', how='outer').fillna(0)
    df['total_count'] = df['count_m'] + df['count_f']
    df['p_male'] = df['count_m'] / df['total_count']
    
    # Clean names
    pattern = r"^[A-Za-z][A-Za-z' \-]{0,49}$"
    mask = (df['Name'].str.match(pattern, na=False) & 
            ~df['Name'].str.contains(r'[^\x00-\x7F]', regex=True, na=False) & 
            (df['Name'].str.len() >= 2))
    df = df[mask].copy()
    
    # Remove rare
    df = df[df['total_count'] >= min_count].copy()
    print(f"Cleaned shape: {df.shape}")
    return df


def prepare_splits(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified 70/15/15 split."""
    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Name_Type'])
    val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(df_trainval, test_size=val_ratio, random_state=random_state, stratify=df_trainval['Name_Type'])
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    return df_train, df_val, df_test