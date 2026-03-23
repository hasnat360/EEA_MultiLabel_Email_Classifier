import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
from utils import remove_low_frequency_classes


class Data:
    def __init__(self, X: np.ndarray, df: pd.DataFrame, target_col: str | None = None) -> None:
        if target_col is None:
            target_col = Config.CLASS_COL
        self.target_col = target_col

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.train_df = self.test_df = None
        self.embeddings = self.y = self.classes = None

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

        df_r = df.reset_index(drop=True)
        X_r = X[: len(df_r)]
        valid_mask = df_r[target_col].notna() & (df_r[target_col].astype(str).str.strip() != "")
        X_v = X_r[valid_mask.values]
        df_v = df_r[valid_mask].reset_index(drop=True)
        X_v, df_v = remove_low_frequency_classes(df_v, X_v, target_col, Config.MIN_CLASS_COUNT)

        y = df_v[target_col].values
        classes = list(pd.Series(y).unique())
        if len(classes) < 2:
            print(f"  [SKIP] '{target_col}': fewer than 2 classes with >= {Config.MIN_CLASS_COUNT} records.")
            return

        idx = np.arange(len(df_v))
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=Config.TEST_SIZE,
            random_state=Config.SEED,
            stratify=y,
        )

        self.X_train = X_v[tr_idx]
        self.X_test = X_v[te_idx]
        self.y_train = y[tr_idx]
        self.y_test = y[te_idx]
        self.train_df = df_v.iloc[tr_idx].reset_index(drop=True)
        self.test_df = df_v.iloc[te_idx].reset_index(drop=True)
        self.embeddings = X_v
        self.y = y
        self.classes = classes

    def is_valid(self) -> bool:
        return self.X_train is not None


class FilteredData:
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if len(X_train) > 0 and len(X_test) > 0:
            self.embeddings = np.vstack([X_train, X_test])
            self.y = np.concatenate([y_train, y_test])
        elif len(X_train) > 0:
            self.embeddings = X_train
            self.y = y_train
        else:
            self.embeddings = X_test
            self.y = y_test

    def is_valid(self) -> bool:
        return self.X_train is not None and len(self.X_train) > 0
