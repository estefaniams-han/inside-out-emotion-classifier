# src/train_emotion.py
from __future__ import annotations

import os
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---- Mapping: Inside Out 1 + neutral ----
# GoEmotions labels (strings)
INSIDE_OUT_MAP = {
    # Inside Out core emotions
    "anger": {"anger", "annoyance"},
    "disgust": {"disgust", "disapproval", "contempt"},
    "fear": {"fear", "nervousness"},
    "sadness": {"sadness", "grief", "disappointment", "remorse"},
    "joy": {"joy", "amusement", "love", "excitement"},
    "surprise": {"surprise", "realization"},
    # Neutral
    "neutral": {"neutral"},
}

# Priority for multilabel rows (stronger emotions first)
PRIORITY = [
    "anger",
    "disgust",
    "fear",
    "sadness",
    "joy",
    "surprise",
    "neutral",
]

# Helpful full report
LABELS_ORDER = [
        "anger",
        "disgust",
        "fear",
        "sadness",
        "joy",
        "surprise",
        "neutral",
]

def map_labels_to_inside_out(label_names: list[str]) -> str:
    """
    label_names: e.g. ["nervousness", "joy"]
    returns one of: neutral, joy, anger, disgust, sadness, fear, surprise
    """
    # Pick the first class that matches based on PRIORITY
    for target in PRIORITY:
        if any(lbl in INSIDE_OUT_MAP[target] for lbl in label_names):
            return target
    
    # Anything not mapped goes to neutral (curiosity, confusion, etc.)
    return "neutral"

def main() -> None:
    print("Loading GoEmotions dataset...")
    ds = load_dataset("google-research-datasets/go_emotions")

    id2label = ds["train"].features["labels"].feature.names

    print("Building training set...")
    texts: list[str] = []
    y: list[str] = []

    for split in ["train", "validation"]:
        for row in ds[split]:
            label_ids = row["labels"] # list of ints
            label_names = [id2label[i] for i in label_ids]
            mapped = map_labels_to_inside_out(label_names)

            texts.append(row["text"])
            y.append(mapped)

    # Train/val split (stratified) to keep class proportions
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Building pipeline (TF-IDF + Logistic Regression)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=3000)),
    ])

    # ---- Class weights (manual) ----
    # Idea: No castigar demasiado neutral para evitar que se vaya a emociones,
    # pero sí darle un empujón a fear/surpsrise/disgust etc.
    cw_light = {"neutral": 1.0, "joy": 1.2, "anger": 1.4, "sadness": 1.4, "disgust": 1.5, "surprise": 1.6, "fear": 1.8}
    cw_mid   = {"neutral": 1.0, "joy": 1.3, "anger": 1.6, "sadness": 1.6, "disgust": 1.8, "surprise": 2.0, "fear": 2.3}
    cw_strong= {"neutral": 1.0, "joy": 1.4, "anger": 1.9, "sadness": 1.9, "disgust": 2.2, "surprise": 2.4, "fear": 2.8}

    # --- Grid ----
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "tfidf__max_df": [0.9, 0.95],
        "tfidf__sublinear_tf": [True],
        "clf__C": [0.5, 1.0, 2.0, 4.0],
        "clf__class_weight": [cw_light, cw_mid, cw_strong],
        "clf__solver": ["lbfgs"],  # estable para multinomial/ovr en textos
    }

    print("Running GridSearchCV (scoring = F1 macro)...")
    grid = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        scoring = "f1_macro",
        cv = 3,
        n_jobs = -1,
        verbose = 1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best params:")
    print(grid.best_params_)
    print(f"Best CV Macro F1: {grid.best_score_:.4f}\n")

    print("Evaluating best model on holdout validation split...")
    preds = best_model.predict(X_val)

    macro_f1 = f1_score(y_val, preds, average="macro")
    print(f"\n✅ Holdout Macro F1: {macro_f1:.4f}\n")
    print(classification_report(y_val, preds, labels=LABELS_ORDER))

    # Confusion matrix
    cm = confusion_matrix(y_val, preds, labels=LABELS_ORDER)
    print("Confusion matrix (rows=true, cols=pred) order:", LABELS_ORDER)
    print(cm)

    print("Saving model...")
    os.makedirs("models", exist_ok=True)
    out_path = "models/emotion_lr_tfidf_best.joblib"
    joblib.dump(best_model, out_path)
    print(f"\nSaved -> {out_path} ✅")


if __name__ == "__main__":
    main()
