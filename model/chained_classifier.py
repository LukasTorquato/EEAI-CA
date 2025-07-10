import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

class ChainedMultiOutputClassifier(BaseModel):
    """
    Chained Multi-Output Architecture:
    y2 (Type 2) → y3 (Type 3) → y4 (Type 4)
    Each prediction uses the original features + all previous predictions
    """
    
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(ChainedMultiOutputClassifier, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y  # expected shape (n_samples, ) for y2, y3, y4
        
        # Individual models for each level
        rfc = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight='balanced')
        
        self.model_y2 = BaggingClassifier(estimator=rfc, n_estimators=20, random_state=seed, n_jobs=-1)
        self.model_y3 = BaggingClassifier(estimator=rfc, n_estimators=20, random_state=seed, n_jobs=-1)
        self.model_y4 = BaggingClassifier(estimator=rfc, n_estimators=20, random_state=seed, n_jobs=-1)

        # Label encoders for each target
        self.encoder_y2 = LabelEncoder()
        self.encoder_y3 = LabelEncoder()
        self.encoder_y4 = LabelEncoder()
        
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """Train models in chain: y2 → y3 → y4"""
        X_train = data.X_train
        y_train = data.y_train
        
        # Encode all targets
        y2_train_encoded = self.encoder_y2.fit_transform(y_train[:, 0])
        y3_train_encoded = self.encoder_y3.fit_transform(y_train[:, 1])
        y4_train_encoded = self.encoder_y4.fit_transform(y_train[:, 2])
 
        print("Training Level 2 (y2)...")
        # Level 2: Train y2 using original features
        self.model_y2.fit(X_train, y2_train_encoded)

        print("Training Level 3 (y3)...")
        # Level 3: Train y3 using original features + y2 predictions
        y2_pred_train = self.model_y2.predict(X_train)
        X_train_with_y2 = np.column_stack([X_train, y2_pred_train])
        self.model_y3.fit(X_train_with_y2, y3_train_encoded)
        
        print("Training Level 4 (y4)...")
        # Level 4: Train y4 using original features + y2 + y3 predictions
        y3_pred_train = self.model_y3.predict(X_train_with_y2)
        X_train_with_y2_y3 = np.column_stack([X_train_with_y2, y3_pred_train])
        self.model_y4.fit(X_train_with_y2_y3, y4_train_encoded)

    def predict(self, X_test: np.ndarray):
        """Predict in chain: y2 → y3 → y4"""
        print("Predicting with Chained Multi-Output Classifier...")
        
        # Level 2: Predict y2
        y2_pred_encoded = self.model_y2.predict(X_test)
        y2_pred = self.encoder_y2.inverse_transform(y2_pred_encoded)

        # Level 3: Predict y3 using y2
        X_test_with_y2 = np.column_stack([X_test, y2_pred_encoded])
        y3_pred_encoded = self.model_y3.predict(X_test_with_y2)
        y3_pred = self.encoder_y3.inverse_transform(y3_pred_encoded)

        # Level 4: Predict y4 using y2 + y3
        X_test_with_y2_y3 = np.column_stack([X_test_with_y2, y3_pred_encoded])
        y4_pred_encoded = self.model_y4.predict(X_test_with_y2_y3)
        y4_pred = self.encoder_y4.inverse_transform(y4_pred_encoded)
        
        # Combine all predictions
        self.predictions = np.column_stack([y2_pred, y3_pred, y4_pred])
        return self.predictions

    def print_results(self, data, by_layer = False):

        if self.predictions is None:
            print("No predictions available. Please run predict() first.")
            return
        if by_layer:
            level_names = ['Type 2 (y2)', 'Type 3 (y3)', 'Type 4 (y4)']
            
            print(f"=== {self.model_name} - Chained Multi-Output Results ===")
            
            for i, level_name in enumerate(level_names):
                print(f"\n--- {level_name} ---")
                y_true = data.y_test[:, i]
                y_pred = self.predictions[:, i]
                y_true_clean = np.array(['missing' if x is np.nan else x for x in y_true])
                y_pred_clean = np.array(['missing' if x is np.nan else x for x in y_pred])
                print(f"Accuracy: {accuracy_score(y_true_clean, y_pred_clean):.4f}")
                print("Classification Report:")
                print(classification_report(y_true_clean, y_pred_clean, zero_division=0))

        # Overall Accuracy
        else:
            self.multi_accuracy(data)

    def multi_accuracy(self, data):
        """
        Calculate multi-label accuracy.
        Returns the proportion of samples where all labels match.
        """
        y_true = data.y_test
        y_pred = self.predictions

        accuracy = []
        for yt, yp in zip(y_true, y_pred):
            acc = 0
            for i in range(len(yt)):
                if yt[i] == yp[i]:
                    acc += 1
                else:
                    break
            accuracy.append(acc)

        print("Individual Accuracies: ", np.array(accuracy) / len(y_true[0]))  # Divide by number of labels (3 in this case)
        print("Overall Accuracy: ", (np.average(np.array(accuracy)) / len(y_true[0])))  # Divide by number of labels (3 in this case)

    def data_transform(self) -> None:
        pass