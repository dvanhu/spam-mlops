from fastapi import FastAPI
from api.schema import TextRequest
from api.utils import load_model

app = FastAPI()

model, vectorizer = load_model()

@app.get("/")
def home():
    return {"message": "Spam Classifier Running"}

@app.post("/predict")
def predict(request: TextRequest):
    X = vectorizer.transform([request.text])
    prediction = model.predict(X)[0]
    return {"prediction": prediction}
