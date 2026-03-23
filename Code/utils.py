import numpy as np
import pandas as pd


def remove_low_frequency_classes(df: pd.DataFrame, X: np.ndarray, target_col: str, min_count: int):
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    mask = df[target_col].isin(valid_classes)
    return X[mask.values], df[mask].reset_index(drop=True)


def keep_top_level_classes(df: pd.DataFrame, col: str, min_count: int) -> pd.DataFrame:
    valid = df[col].value_counts()
    valid = valid[valid > min_count].index
    return df.loc[df[col].isin(valid)].copy()
