import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

class MultiTargetData():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        # Get all target columns
        target_cols = ['y2', 'y3', 'y4']
        y_multi = df[target_cols].to_numpy()
        
        # Filter out rows where ALL targets are empty/nan
        valid_rows = ~pd.DataFrame(y_multi).isin(['', np.nan]).all(axis=1)
        print(f"Valid rows after filtering: {valid_rows.sum()} out of {len(y_multi)}")
        y_clean = y_multi[valid_rows]
        X_clean = X[valid_rows]
        
        if len(y_clean) < 6:  # Need at least 6 samples for train/test split
            print("Insufficient data for multi-target classification: Skipping ...")
            self.X_train = None
            return

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=0
        )
        
        self.y = y_clean
        self.embeddings = X_clean

    def get_type(self):
        return self.y
    
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    def get_type_y_train(self):
        return self.y_train
    
    def get_type_y_test(self):
        return self.y_test
    
    def get_embeddings(self):
        return self.embeddings