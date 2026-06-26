import pandas as pd
import numpy as np

VOWELS = set('aeiouAEIOU')
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
FEATURE_NAMES = ['name_len'] + [f'first_{c}' for c in ALPHABET] + [f'last_{c}' for c in ALPHABET] + ['last_2_ord', 'last_3_ord', 'vowel_ratio', 'ends_with_a', 'ends_with_e', 'ends_with_y', 'ends_with_son', 'ends_with_lyn', 'ends_with_ette', 'ends_with_ie', 'ends_with_ine', 'log_count']


def extract_features(name_series: pd.Series, total_count_series: pd.Series) -> pd.DataFrame:
    """Extract 65 features from names."""
    names = name_series.str.lower().str.strip()
    feats = pd.DataFrame(index=name_series.index)
    feats['name_len'] = names.str.len()
    for ch in ALPHABET:
        feats[f'first_{ch}'] = (names.str[0] == ch).astype(int)
        feats[f'last_{ch}'] = (names.str[-1] == ch).astype(int)
    feats['last_2_ord'] = names.str[-2:].apply(lambda s: sum(ord(c) * (26 ** i) for i, c in enumerate(s)) % 10000)
    feats['last_3_ord'] = names.str[-3:].apply(lambda s: sum(ord(c) * (26 ** i) for i, c in enumerate(s)) % 10000)
    feats['vowel_ratio'] = names.apply(lambda n: sum(c in VOWELS for c in n) / max(len(n), 1))
    for suffix in ['a', 'e', 'y', 'son', 'lyn', 'ette', 'ie', 'ine']:
        feats[f'ends_with_{suffix}'] = names.str.endswith(suffix).astype(int)
    feats['log_count'] = np.log10(total_count_series + 1)
    return feats


def build_feature_matrices(df_train, df_val, df_test):
    """Build feature matrices for all splits."""
    X_train_all = extract_features(df_train['Name'], df_train['total_count'])
    X_val_all = extract_features(df_val['Name'], df_val['total_count'])
    X_test_all = extract_features(df_test['Name'], df_test['total_count'])
    binary_train = df_train['Name_Type'].isin(['Strong Male', 'Strong Female'])
    binary_val = df_val['Name_Type'].isin(['Strong Male', 'Strong Female'])
    binary_test = df_test['Name_Type'].isin(['Strong Male', 'Strong Female'])
    return {
        'X_train_all': X_train_all, 'X_val_all': X_val_all, 'X_test_all': X_test_all,
        'X_train_bin': X_train_all[binary_train], 'X_val_bin': X_val_all[binary_val], 'X_test_bin': X_test_all[binary_test],
        'y_train_bin': (df_train.loc[binary_train, 'p_male'] > 0.5).astype(int),
        'y_val_bin': (df_val.loc[binary_val, 'p_male'] > 0.5).astype(int),
        'y_test_bin': (df_test.loc[binary_test, 'p_male'] > 0.5).astype(int),
        'y_train_reg': df_train['p_male'], 'y_val_reg': df_val['p_male'], 'y_test_reg': df_test['p_male'],
        'feature_names': FEATURE_NAMES,
    }