import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import random

seed = 0
np.random.seed(seed)
random.seed(seed)

class ChainedMultiOutputClassifier(BaseModel):
    """
    Chained Multi-Output Architecture:
    y1 (Type 1) → y2 (Type 2) → y3 (Type 3) → y4 (Type 4)
    Each prediction uses the original features + all previous predictions
    """
    
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(ChainedMultiOutputClassifier, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y  # expected shape (n_samples, 4) for y1, y2, y3, y4
        
        # Individual models for each level
        self.model_y1 = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight='balanced')
        self.model_y2 = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight='balanced')
        self.model_y3 = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight='balanced')
        self.model_y4 = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight='balanced')
        
        # Label encoders for each target
        self.encoder_y1 = LabelEncoder()
        self.encoder_y2 = LabelEncoder()
        self.encoder_y3 = LabelEncoder()
        self.encoder_y4 = LabelEncoder()
        
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """Train models in chain: y1 → y2 → y3 → y4"""
        X_train = data.X_train
        y_train = data.y_train
        
        # Encode all targets
        y1_train_encoded = self.encoder_y1.fit_transform(y_train[:, 0])
        y2_train_encoded = self.encoder_y2.fit_transform(y_train[:, 1])
        y3_train_encoded = self.encoder_y3.fit_transform(y_train[:, 2])
        y4_train_encoded = self.encoder_y4.fit_transform(y_train[:, 3])
        
        print("Training Level 1 (y1)...")
        # Level 1: Train y1 using only original features
        self.model_y1.fit(X_train, y1_train_encoded)
        
        print("Training Level 2 (y2)...")
        # Level 2: Train y2 using original features + y1 prediction
        y1_pred_train = self.model_y1.predict(X_train)
        X_train_with_y1 = np.column_stack([X_train, y1_pred_train])
        self.model_y2.fit(X_train_with_y1, y2_train_encoded)
        
        print("Training Level 3 (y3)...")
        # Level 3: Train y3 using original features + y1 + y2 predictions
        y2_pred_train = self.model_y2.predict(X_train_with_y1)
        X_train_with_y1_y2 = np.column_stack([X_train_with_y1, y2_pred_train])
        self.model_y3.fit(X_train_with_y1_y2, y3_train_encoded)
        
        print("Training Level 4 (y4)...")
        # Level 4: Train y4 using original features + y1 + y2 + y3 predictions
        y3_pred_train = self.model_y3.predict(X_train_with_y1_y2)
        X_train_with_y1_y2_y3 = np.column_stack([X_train_with_y1_y2, y3_pred_train])
        self.model_y4.fit(X_train_with_y1_y2_y3, y4_train_encoded)

    def predict(self, X_test: np.ndarray):
        """Predict in chain: y1 → y2 → y3 → y4"""
        # Level 1: Predict y1
        y1_pred_encoded = self.model_y1.predict(X_test)
        y1_pred = self.encoder_y1.inverse_transform(y1_pred_encoded)
        
        # Level 2: Predict y2 using y1
        X_test_with_y1 = np.column_stack([X_test, y1_pred_encoded])
        y2_pred_encoded = self.model_y2.predict(X_test_with_y1)
        y2_pred = self.encoder_y2.inverse_transform(y2_pred_encoded)
        
        # Level 3: Predict y3 using y1 + y2
        X_test_with_y1_y2 = np.column_stack([X_test_with_y1, y2_pred_encoded])
        y3_pred_encoded = self.model_y3.predict(X_test_with_y1_y2)
        y3_pred = self.encoder_y3.inverse_transform(y3_pred_encoded)
        
        # Level 4: Predict y4 using y1 + y2 + y3
        X_test_with_y1_y2_y3 = np.column_stack([X_test_with_y1_y2, y3_pred_encoded])
        y4_pred_encoded = self.model_y4.predict(X_test_with_y1_y2_y3)
        y4_pred = self.encoder_y4.inverse_transform(y4_pred_encoded)
        
        # Combine all predictions
        self.predictions = np.column_stack([y1_pred, y2_pred, y3_pred, y4_pred])
        return self.predictions

    def print_results(self, data):

        if self.predictions is None:
            print("No predictions available. Please run predict() first.")
            return
            
        level_names = ['Type 1 (y1)', 'Type 2 (y2)', 'Type 3 (y3)', 'Type 4 (y4)']
        
        print(f"=== {self.model_name} - Chained Multi-Output Results ===")
        
        for i, level_name in enumerate(level_names):
            print(f"\n--- {level_name} ---")
            y_true = data.y_test[:, i]
            y_pred = self.predictions[:, i]
            
            # Filter out empty/nan values for evaluation
            valid_indices = (y_true != '') & (~pd.isna(y_true))
            if valid_indices.sum() > 0:
                y_true_clean = y_true[valid_indices]
                y_pred_clean = y_pred[valid_indices]
                
                print(f"Accuracy: {accuracy_score(y_true_clean, y_pred_clean):.4f}")
                print("Classification Report:")
                print(classification_report(y_true_clean, y_pred_clean, zero_division=0))
            else:
                print("No valid data for evaluation")

    def data_transform(self) -> None:
        pass