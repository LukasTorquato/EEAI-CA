import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class FeedforwardNN(BaseModel):
    def __init__(self, model_name, embeddings, y):
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.model = None
        self.predictions = None
        self.data_transform()

    def data_transform(self):
        self.label_map = {label: idx for idx, label in enumerate(np.unique(self.y))}
        self.y_encoded = np.array([self.label_map[label] for label in self.y])

    def train(self, data):
        X_train = torch.tensor(data.X_train, dtype=torch.float32)
        y_train = torch.tensor([self.label_map[label] for label in data.y_train], dtype=torch.long)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        input_dim = X_train.shape[1]
        num_classes = len(self.label_map)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Compute class weights
        y_labels = np.array([self.label_map[label] for label in data.y_train])
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_labels), y=y_labels)
        weights = torch.tensor(class_weights, dtype=torch.float32)

        # Apply class weights to loss
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(30):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            preds = torch.argmax(outputs, axis=1)
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.predictions = np.array([reverse_label_map[p.item()] for p in preds])

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))
