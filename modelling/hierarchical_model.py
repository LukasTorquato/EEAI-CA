import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *

seed = 0
np.random.seed(seed)

class DataWithIndices():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        y_series = df["y"]
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        self.embeddings = X  
        self.df_filtered = df  
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_indices = []
        self.test_indices = []
        self.y = y_series.to_numpy()
        self.classes = []

        if len(good_y_value) < 1:
            print("None of the class have more than 3 records: Skipping ...")
            return

        # Filter dataset for good labels
        mask = y_series.isin(good_y_value)
        df_filtered = df[mask].reset_index(drop=True)
        X_filtered = X[mask.values]
        y_filtered = y_series[mask].to_numpy()

        new_test_size = X.shape[0] * 0.2 / X_filtered.shape[0]

        indices = np.arange(len(X_filtered))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=new_test_size,
            random_state=0,
            stratify=y_filtered
        )

        # Store filtered subset and splits
        self.df_filtered = df_filtered
        self.X_train = X_filtered[train_idx]
        self.X_test = X_filtered[test_idx]
        self.y_train = y_filtered[train_idx]
        self.y_test = y_filtered[test_idx]
        self.train_indices = train_idx
        self.test_indices = test_idx
        self.y = y_filtered
        self.classes = good_y_value

    # Accessor methods
    def get_type(self): return 'classification'
    def get_X_train(self): return self.X_train
    def get_X_test(self): return self.X_test
    def get_type_y_train(self): return self.y_train
    def get_type_y_test(self): return self.y_test
    def get_train_indices(self): return self.train_indices
    def get_test_indices(self): return self.test_indices
    def get_embeddings(self): return self.embeddings
