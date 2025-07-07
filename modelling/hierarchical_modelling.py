import os
import pandas as pd
from model.neuralnetwork import FeedforwardNN
from modelling.data_model import Data
from neural_embeddings import get_sentence_embeddings
from sklearn.metrics import classification_report

def train_submodel(X, y, level_name):
    df_temp = pd.DataFrame({"y": y})
    data = Data(X, df_temp)
    if data.X_train is None:
        # print(f"Skipping {level_name} due to insufficient data.")
        print(f"{level_name}: Unable to create train/test split — likely due to single class.")
        return None

    model = FeedforwardNN(level_name, data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    print(f"\n=== Report for {level_name} ===")
    print(classification_report(data.y_test, model.predictions, zero_division=0))
    return model

def hierarchical_train(df):
    # Generate embeddings
    df["combined_text"] = df["Ticket Summary"] + " " + df["Interaction content"]
    X_all = get_sentence_embeddings(df)

    y2_values = df['y2'].unique()

    for y2_val in y2_values:
        df_y2 = df[df['y2'] == y2_val]
        X_y2 = get_sentence_embeddings(df_y2)

        print(f"\n Training y3 model for y2 = {y2_val}")
        y3_model = train_submodel(X_y2, df_y2['y3'], f"y3_model_{y2_val}")

        if y3_model:
            y3_values = df_y2['y3'].unique()
            for y3_val in y3_values:
                df_y3 = df_y2[df_y2['y3'] == y3_val]
                X_y3 = get_sentence_embeddings(df_y3)

                print(f" Training y4 model for y2 = {y2_val}, y3 = {y3_val}")
                train_submodel(X_y3, df_y3['y4'], f"y4_model_{y2_val}_{y3_val}")
