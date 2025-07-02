import numpy as np
import pandas as pd
from Config import *
from sentence_transformers import SentenceTransformer
import random
seed =0
random.seed(seed)
np.random.seed(seed)

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentence_embeddings(df: pd.DataFrame) -> np.ndarray:
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    embeddings = model.encode(data.tolist(), show_progress_bar=True)
    return np.array(embeddings)

def combine_embd(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return np.concatenate((X1, X2), axis=1)

