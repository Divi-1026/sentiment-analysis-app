"""
Sentiment Model - Full ML Training Pipeline
===========================================
Dataset : Twitter US Airline Sentiment (14,640 tweets, Kaggle)
Task    : 3-class text classification (positive / neutral / negative)

Steps:
  1. Load data
  2. Clean text (URLs, @mentions, punctuation, lowercase)
  3. Train/test split (stratified - keeps class ratio)
  4. TF-IDF vectorization
  5. Train & compare 3 models (class_weight handles imbalance)
  6. Evaluate (accuracy, precision, recall, F1, confusion matrix)
  7. Save the best model as a .pkl

Run:  python train_model.py
"""

import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv("Tweets.csv")[["text", "airline_sentiment"]]
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    print(f"Loaded {len(df)} rows after removing duplicates/NaN")

    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]
    X, y = df["clean"], df["airline_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
    }

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro")
        results[name] = {"pipe": pipe, "acc": acc, "f1_macro": f1m, "pred": pred}
        print(f"{name:22}  accuracy={acc:.3f}  macro-F1={f1m:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best = results[best_name]
    print(f"\n>>> Best model: {best_name} (macro-F1 = {best['f1_macro']:.3f})\n")

    print("Detailed report for the best model:")
    print(classification_report(y_test, best["pred"]))

    print("Confusion matrix (rows = actual, cols = predicted):")
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, best["pred"], labels=labels)
    print("            " + "  ".join(f"{l[:4]:>6}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i]:8}  " + "  ".join(f"{v:6}" for v in row))

    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(best["pipe"], f)
    print("\nSaved best model -> sentiment_model.pkl")

    print("\nSanity check on new sentences:")
    for s in ["the flight was amazing and crew was so helpful",
              "worst airline ever, lost my luggage again",
              "flight departs at 6pm from gate 22"]:
        print(f"  {best['pipe'].predict([clean_text(s)])[0]:9} <- {s}")


if __name__ == "__main__":
    main()
