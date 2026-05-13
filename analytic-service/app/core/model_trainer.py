import json
import os
import threading
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from app.core.minio_client import get_minio_client, upload_model_to_minio

# Lock for writing to the markdown file safely from multiple threads
md_lock = threading.Lock()
MD_REPORT_PATH = "/home/tienpv16/Desktop/Workspace/credit-score-apps/BENCHMARK_RESULTS.md"

# def init_md_report():
    # if not os.path.exists(MD_REPORT_PATH):
    #     with open(MD_REPORT_PATH, "w") as f:
    #         f.write("# 🏆 AI Model Benchmark Results\n\n")
    #         f.write("Bảng dưới đây tổng hợp kết quả của 3 mô hình (LR, RF, XGB) khi được train trên các tập dữ liệu gốc và dữ liệu đã bị ẩn danh.\n\n")
    #         f.write("| Version ID | Data Type | Model | Accuracy | Precision | Recall | F1-Score |\n")
    #         f.write("|------------|-----------|-------|----------|-----------|--------|----------|\n")

def append_to_md_report(version_id: str, data_type: str, model_name: str, metrics: dict):
    with md_lock:
        with open(MD_REPORT_PATH, "a") as f:
            f.write(f"| {version_id} | {data_type} | **{model_name}** | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |\n")

def train_and_save_model(
    parquet_path: str,
    version_id: str,
    model_key: Optional[str] = None,
) -> None:
    # init_md_report()
    
    # Identify data_type from parquet_path or model_key
    data_type = Path(parquet_path).stem
    if model_key:
        data_type = Path(model_key).stem
    if data_type == 'rf_model':
        data_type = 'clean_data'

    # Load data
    df = pd.read_parquet(parquet_path)
    if 'income' not in df.columns:
        raise ValueError("Target column 'income' not found in data.")

    X = df.drop(columns=['income'])
    y = df['income']
    
    # Determine pos_label
    pos_label = '>50K' if '>50K' in y.values else 1

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Encode target to 0/1 to support XGBoost strictly
    if pos_label == '>50K':
        y_encoded = y.apply(lambda x: 1 if x == '>50K' else 0)
        xgb_pos_label = 1
    else:
        y_encoded = y
        xgb_pos_label = pos_label

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42
    )

    models = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "XGB": XGBClassifier(random_state=42, eval_metric='logloss')
    }

    client = get_minio_client()

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            precision = precision_score(y_test, y_pred, pos_label=xgb_pos_label, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=xgb_pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=xgb_pos_label, zero_division=0)
        except Exception:
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"[{data_type}] {model_name} trained with F1: {f1:.4f}")

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "dataset_size": len(df)
        }

        # Append to Markdown report
        # append_to_md_report(version_id, data_type, model_name, metrics)

        # Generate unique keys for each model
        if model_key:
            # e.g., <version_id>/anon/adult_anon_k5.joblib -> <version_id>/anon/adult_anon_k5_{model_name}.joblib
            base_key = model_key.replace(".joblib", "")
            current_model_key = f"{base_key}_{model_name}.joblib"
        else:
            current_model_key = f"{version_id}/{model_name}_model.joblib"
            
        current_metrics_key = current_model_key.replace(".joblib", "_metrics.json")

        # Save locally
        model_path = Path(f"temp_{model_name}_model.joblib")
        metrics_path = Path(f"temp_{model_name}_metrics.json")
        
        joblib.dump(model, model_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Upload to MinIO
        upload_model_to_minio(client, str(model_path), "model-zone", current_model_key)
        upload_model_to_minio(client, str(metrics_path), "model-zone", current_metrics_key)

        # Cleanup
        os.remove(model_path)
        os.remove(metrics_path)