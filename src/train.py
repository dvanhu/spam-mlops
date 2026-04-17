import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import joblib


# -----------------------------
# MLflow Configuration
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("spam-classifier")


# -----------------------------
# Load Data
# -----------------------------
DATA_PATH = "data/spam.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns")


# -----------------------------
# Preprocessing
# -----------------------------
X = df["text"]
y = df["label"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)


# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)


# -----------------------------
# Model Training + MLflow
# -----------------------------
model = MultinomialNB()

with mlflow.start_run():

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)

    # Log parameters
    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("vectorizer", "CountVectorizer")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model artifact
    mlflow.sklearn.log_model(model, "model")

    print(f"Training complete. Accuracy: {accuracy:.4f}")


# -----------------------------
# Save Model Locally
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved in /models/")
