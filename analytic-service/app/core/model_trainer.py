import os
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.core.minio_client import get_minio_client, upload_model_to_minio


def train_and_save_model(parquet_path: str, version_id: str) -> None:
    """Train a RandomForest model on the Parquet data and save to MinIO."""
    # Load data
    df = pd.read_parquet(parquet_path)

    # Assume 'income' is the target
    if 'income' not in df.columns:
        raise ValueError("Target column 'income' not found in data.")

    X = df.drop(columns=['income'])
    y = df['income']

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")

    # Save model locally
    model_path = Path("temp_rf_model.joblib")
    joblib.dump(model, model_path)

    # Upload to MinIO
    client = get_minio_client()
    upload_model_to_minio(
        client=client,
        local_path=str(model_path),
        bucket="model-zone",
        key=f"{version_id}/rf_model.joblib"
    )

    # Clean up
    os.remove(model_path)