from preprocess import *
from hierarchical_embeddings import *
from modelling.modelling import *
from modelling.hierarchical_model import DataWithIndices as Data
from model.randomforest import RandomForest
from Config import Config
from sklearn.metrics import classification_report
import random
import numpy as np
import pandas as pd

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    return get_input_data()

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    return df

def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df, min_df=1, max_df=1.0)
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    for label in ['y2', 'y3', 'y4']:
        Config.CLASS_COL = label
        df_copy = df.copy()
        df_copy['y'] = df_copy[Config.CLASS_COL]
        print(f"\n Training model for {label} in {name}")
        model_predict(data, df_copy, f"{name}_{label}")

def train_filtered_hierarchical_models(df: pd.DataFrame):
    print("\n Starting Filtered Hierarchical Modelling (Model Tree)")
    all_rows = []

    print("\n Global Model for y2 (Intent)")
    Config.CLASS_COL = 'y2'
    df["y"] = df[Config.CLASS_COL]
    X_y2 = get_tfidf_embd(df, min_df=1, max_df=1.0)
    data_y2 = Data(X_y2, df)
    if data_y2.X_train is None:
        print(" Skipping y2 — not enough data.")
        return

    model_y2 = RandomForest("RF_y2_global", data_y2.get_embeddings(), data_y2.get_type())
    model_y2.train(data_y2)
    model_y2.predict(data_y2.X_test)

    df_y2_test = data_y2.df_filtered.loc[data_y2.get_test_indices()].copy()
    print(f" y2 test samples: {len(df_y2_test)}")
    df_y2_test["y2_true"] = data_y2.y_test
    df_y2_test["y2_pred"] = model_y2.predictions

    print("\n Classification Report for y2 (Intent)")
    print(classification_report(df_y2_test["y2_true"], df_y2_test["y2_pred"], zero_division=0))

    for y2_class in df_y2_test["y2_pred"].unique():
        df_y3_subset = df_y2_test[df_y2_test["y2_pred"] == y2_class].copy()
        print(f" y3 test samples for y2 = {y2_class}: {len(df_y3_subset)}")
        print(f"\n Training Model for y3 where y2_pred = '{y2_class}' (n={len(df_y3_subset)})")

        Config.CLASS_COL = 'y3'
        df_y3_subset["y"] = df_y3_subset[Config.CLASS_COL]
        X_y3 = get_tfidf_embd(df_y3_subset, min_df=1, max_df=1.0)
        data_y3 = Data(X_y3, df_y3_subset)

        if data_y3.X_train is None or len(data_y3.X_test) == 0 or pd.isnull(data_y3.y_train).any() or pd.isnull(data_y3.X_train).any():
            print(f" Skipping y3 model for y2={y2_class} due to NaNs or insufficient data")
            continue

        model_y3 = RandomForest(f"RF_y3_given_y2={y2_class}", data_y3.get_embeddings(), data_y3.get_type())
        model_y3.train(data_y3)
        model_y3.predict(data_y3.X_test)

        df_y3_test = data_y3.df_filtered.loc[data_y3.get_test_indices()].copy()
        df_y3_test["y3_true"] = data_y3.y_test
        df_y3_test["y3_pred"] = model_y3.predictions
        df_y3_test["y2_true"] = df_y3_subset["y2_true"].values[:len(df_y3_test)]
        df_y3_test["y2_pred"] = df_y3_subset["y2_pred"].values[:len(df_y3_test)]

        print(f"\n Classification Report for y3 (Tone) [y2 = {y2_class}]")
        print(classification_report(df_y3_test["y3_true"], df_y3_test["y3_pred"], zero_division=0))

        for y3_class in df_y3_test["y3_pred"].unique():
            df_y4_subset = df_y3_test[df_y3_test["y3_pred"] == y3_class].copy()
            print(f" y4 test samples for y2 = {y2_class}, y3 = {y3_class}: {len(df_y4_subset)}")
            print(f" Training Model for y4 where y2={y2_class}, y3_pred={y3_class} (n={len(df_y4_subset)})")

            Config.CLASS_COL = 'y4'
            df_y4_subset["y"] = df_y4_subset[Config.CLASS_COL]
            X_y4 = get_tfidf_embd(df_y4_subset, min_df=1, max_df=1.0)
            data_y4 = Data(X_y4, df_y4_subset)

            if data_y4.X_train is None or len(data_y4.X_test) == 0 or pd.isnull(data_y4.y_train).any() or pd.isnull(data_y4.X_train).any():
                print(f" Skipping y4 model for y2={y2_class}, y3={y3_class} due to NaNs or insufficient data")
                continue

            model_y4 = RandomForest(f"RF_y4_given_y2={y2_class}_y3={y3_class}",
                                    data_y4.get_embeddings(), data_y4.get_type())
            model_y4.train(data_y4)
            model_y4.predict(data_y4.X_test)

            df_y4_test = data_y4.df_filtered.loc[data_y4.get_test_indices()].copy()
            df_y4_test["y4_true"] = data_y4.y_test
            df_y4_test["y4_pred"] = model_y4.predictions
            df_y4_test["y2_true"] = df_y4_subset["y2_true"].values[:len(df_y4_test)]
            df_y4_test["y2_pred"] = df_y4_subset["y2_pred"].values[:len(df_y4_test)]
            df_y4_test["y3_true"] = df_y4_subset["y3_true"].values[:len(df_y4_test)]
            df_y4_test["y3_pred"] = df_y4_subset["y3_pred"].values[:len(df_y4_test)]

            print(f"\n Classification Report for y4 (Resolution Type) [y2 = {y2_class}, y3 = {y3_class}]")
            print(classification_report(df_y4_test["y4_true"], df_y4_test["y4_pred"], zero_division=0))

            all_rows.append(df_y4_test)

    if all_rows:
        final_eval_df = pd.concat(all_rows, ignore_index=True)
        evaluate_filtered_rowwise(final_eval_df)

def evaluate_filtered_rowwise(df: pd.DataFrame):
    print("\n Evaluating Row-wise Accuracy Across y2 → y3 → y4")
    total_correct = [0, 0, 0, 0]
    df = df.dropna(subset=["y2_true", "y3_true", "y4_true", "y2_pred", "y3_pred", "y4_pred"])

    for _, row in df.iterrows():
        true_vals = [row["y2_true"], row["y3_true"], row["y4_true"]]
        pred_vals = [row["y2_pred"], row["y3_pred"], row["y4_pred"]]
        correct = sum([t == p for t, p in zip(true_vals, pred_vals)])
        print(f"True: {true_vals}, Predicted: {pred_vals}")
        print(f"Individual Accuracy: {correct}\n")
        total_correct[correct] += 1

    print(" Final Distribution of Per-Row Accuracy:")
    for i in range(4):
        print(f"{i}/3 correct: {total_correct[i]} rows")

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    grouped_df = df.groupby(Config.GROUPED)
    # for name, group_df in grouped_df:
    #     print(name)
    #     X, group_df = get_embeddings(group_df)
    #     data = get_data_object(X, group_df)
    #     perform_modelling(data, group_df, name)

    train_filtered_hierarchical_models(df)

