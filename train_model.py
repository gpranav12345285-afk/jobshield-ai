"""Train a TF-IDF + Logistic Regression model for fake job posting detection.

Dataset: Kaggle Real / Fake Job Posting Prediction
Dataset slug: shivamb/real-or-fake-fake-jobposting-prediction
Expected target column: fraudulent (1 = fraudulent, 0 = genuine)
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "artifacts"
DATA_PATH = Path(os.environ.get("FAKE_JOB_DATA", DATA_DIR / "fake_job_postings.csv"))
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]


def locate_csv() -> Path:
    """Find dataset CSV locally, or download the public Kaggle dataset using kagglehub."""
    if DATA_PATH.exists():
        return DATA_PATH
    for candidate in [ROOT / "fake_job_postings.csv", *DATA_DIR.glob("*.csv")]:
        if candidate.exists():
            return candidate
    try:
        import kagglehub
        downloaded_dir = Path(kagglehub.dataset_download("shivamb/real-or-fake-fake-jobposting-prediction"))
        matches = list(downloaded_dir.rglob("*.csv"))
        # Prefer the canonical file name if present.
        matches.sort(key=lambda p: (p.name.lower() != "fake_job_postings.csv", len(str(p))))
        if matches:
            print(f"Using dataset: {matches[0]}")
            return matches[0]
    except Exception as exc:
        raise RuntimeError(
            "Could not download the Kaggle dataset automatically. Download "
            "fake_job_postings.csv from the Kaggle dataset 'shivamb/real-or-fake-fake-jobposting-prediction', "
            "place it in the project's data/ folder, then rerun. If Kaggle asks for sign-in, accept its dataset "
            "terms and configure Kaggle access first. Original error: " + str(exc)
        ) from exc
    raise FileNotFoundError("No CSV found in the downloaded dataset. Put fake_job_postings.csv in data/.")


def build_text(frame: pd.DataFrame) -> pd.Series:
    available = [col for col in TEXT_COLUMNS if col in frame.columns]
    if not available:
        raise ValueError(f"Dataset must contain at least one of these text columns: {TEXT_COLUMNS}")
    clean = frame[available].fillna("").astype(str)
    # Prefix fields with their names so the model retains context.
    return clean.apply(lambda row: " ".join(f"{col}: {row[col]}" for col in available), axis=1)


def train() -> dict:
    csv_path = locate_csv()
    print(f"Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if "fraudulent" not in df.columns:
        raise ValueError("Target column 'fraudulent' was not found in the CSV.")
    df = df.dropna(subset=["fraudulent"]).copy()
    y = pd.to_numeric(df["fraudulent"], errors="coerce")
    valid = y.isin([0, 1])
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    X = build_text(df)
    # Remove completely empty records, preserving matching target indices.
    nonempty = X.str.strip().ne("")
    X, y = X.loc[nonempty], y.loc[nonempty]
    if y.nunique() != 2:
        raise ValueError("The dataset must include both fraudulent=0 and fraudulent=1 rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            max_features=120_000, min_df=2, sublinear_tf=True,
            stop_words="english", dtype="float32"
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42
        )),
    ])
    print(f"Training on {len(X_train):,} rows; testing on {len(X_test):,} rows...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "dataset_file": csv_path.name,
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraudulent_rows": int((y == 1).sum()),
        "genuine_rows": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_fraud": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        "f1_fraud": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": ["Genuine (0)", "Fraudulent (1)"],
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, pred, labels=[0, 1],
                                                       target_names=["Genuine", "Fraudulent"],
                                                       zero_division=0, output_dict=True),
        "model": "TF-IDF (unigrams + bigrams) + Logistic Regression",
        "random_state": 42,
        "test_size": 0.20,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "text_columns": [c for c in TEXT_COLUMNS if c in df.columns]}, MODEL_DIR / "fake_job_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nEVALUATION RESULTS (held-out test set)")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Fraud precision:   {metrics['precision_fraud']:.3f}")
    print(f"Fraud recall:      {metrics['recall_fraud']:.3f}")
    print(f"Fraud F1-score:    {metrics['f1_fraud']:.3f}")
    print("Confusion matrix (rows=actual, columns=predicted; order 0,1):")
    print(metrics["confusion_matrix"])
    print(f"\nSaved model: {MODEL_DIR / 'fake_job_model.joblib'}")
    print(f"Saved metrics: {MODEL_DIR / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    train()
