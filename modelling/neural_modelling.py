from model.neuralnetwork import FeedforwardNN

def model_predict(data, df, name, model_type="nn"):
    if model_type == "nn":
        model = FeedforwardNN("NN", data.get_embeddings(), data.get_type())
    else:
        raise NotImplementedError(f"{model_type} not supported yet.")
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

def model_evaluate(model, data):
    model.print_results(data)