"""
train_model.py - Trains a Decision Tree Classifier and saves it with Joblib.
Run: python train_model.py
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from preprocessing import clean_dataframe, get_feature_columns, score_to_label
from generate_dataset import generate_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/priority_model.pkl")
DATA_PATH  = os.environ.get("DATA_PATH",  "data/tasks.csv")


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        logger.info("Loading dataset from %s", DATA_PATH)
        df = pd.read_csv(DATA_PATH)
    else:
        logger.info("No dataset found – generating synthetic data …")
        os.makedirs("data", exist_ok=True)
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
        logger.info("Synthetic dataset saved to %s", DATA_PATH)
    return df


def train():
    logger.info("=== Smart Task Priority Predictor – Model Training ===")

    # 1. Load data
    df = load_or_generate_data()
    logger.info("Dataset shape: %s", df.shape)

    # 2. Check required columns
    required = ["deadline_date", "estimated_effort", "business_impact",
                "urgency", "dependency_count", "task_type", "priority_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # 3. Clean and prepare features
    df_clean = clean_dataframe(df)
    feature_cols = get_feature_columns()

    X = df_clean[feature_cols]
    y = df_clean["priority_label"]

    logger.info("Class distribution:\n%s", y.value_counts().to_string())

    # 4. Train / test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Train size: %d  Test size: %d", len(X_train), len(X_test))

    # 5. Train Decision Tree
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    logger.info("Model trained.")

    # 6. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info("Accuracy: %.4f (%.1f%%)", acc, acc * 100)
    logger.info("Classification Report:\n%s", report)

    print("\n" + "=" * 50)
    print(f"  Accuracy : {acc * 100:.1f}%")
    print("=" * 50)
    print(report)

    if acc < 0.70:
        logger.warning("Accuracy below 70%% target – consider regenerating data.")

    # 7. Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train()
