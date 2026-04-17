import pickle

def load_model():
    model = pickle.load(open("models/model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    return model, vectorizer
