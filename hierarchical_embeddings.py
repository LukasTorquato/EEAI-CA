import numpy as np
import pandas as pd
from Config import *
import random
from sklearn.feature_extraction.text import TfidfVectorizer
seed =0
random.seed(seed)
np.random.seed(seed)


def get_tfidf_embd(df, min_df=1, max_df=1.0):
    tfidfconverter = TfidfVectorizer(
        max_features=2500,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )
    data = df[Config.INTERACTION_CONTENT].values.astype('U')
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

