from model.chained_classifier import ChainedMultiOutputClassifier

def model_predict(data, df, name):
    """Compare all three approaches"""
    if data.X_train is None:
        print("Skipping due to insufficient data")
        return
    
    results = []
    
    # Chained Multi-Output approach
    print("\n" + "="*50)
    print("CHAINED MULTI-OUTPUT APPROACH")
    print("="*50)
    chained_model = ChainedMultiOutputClassifier("ChainedRF", data.get_embeddings(), data.get_type())
    chained_model.train(data)
    chained_model.predict(data.X_test)
    chained_model.print_results(data, by_layer=False)

def model_evaluate(model, data):
    model.print_results(data, by_layer=False)