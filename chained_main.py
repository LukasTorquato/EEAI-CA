from preprocess import *
from embeddings import *
from modelling.chained_modelling import *
from modelling.chained_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return MultiTargetData(X, df)  # Use multi-target data class

def perform_modelling(data: MultiTargetData, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    X, df = get_embeddings(df)
    data = get_data_object(X, df)
    perform_modelling(data, df, "All Data")
