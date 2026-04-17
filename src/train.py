import pandas as pd
import pickle
import mlflow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv(
    "data/raw/spam.tsv",
    sep="\t",
    names=["label", "text"]
)

X = df["text"]
y = df["label"]


# -----------------------------
# 2. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 3. Feature Engineering (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# 4. Model Training
# -----------------------------
model = LogisticRegression(class_weight="balanced", max_iter=200)
model.fit(X_train_vec, y_train)


# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)


# -----------------------------
# 6. Save Artifacts
# -----------------------------
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))


# -----------------------------
# 7. MLflow Tracking
# -----------------------------
with mlflow.start_run():
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")


# -----------------------------
# 8. Output
# -----------------------------
print(f"Accuracy: {acc}")
