import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    tfidf = TfidfVectorizer(
        max_features=Config.MAX_FEATURES,
        min_df=Config.MIN_DF,
        max_df=Config.MAX_DF,
    )
    combined = (
        df[Config.TICKET_SUMMARY].fillna("").astype(str)
        + " "
        + df[Config.INTERACTION_CONTENT].fillna("").astype(str)
    )
    return tfidf.fit_transform(combined).toarray()
