"""
tests/audit_model.py
Inspects saved models, scalers, and metadata to identify the exact feature schema.
"""

import os
import json
import pickle
import numpy as np

def audit_artifacts():
    print("=" * 60)
    print("MINDBLOOM MODEL & FEATURE ARTIFACT AUDIT")
    print("=" * 60)

    # 1. Inspect scaler.pkl
    scaler_path = os.path.join("models", "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print(f"[SCALER] Type: {type(scaler).__name__}")
            if hasattr(scaler, "n_features_in_"):
                print(f"[SCALER] n_features_in_: {scaler.n_features_in_}")
            if hasattr(scaler, "feature_names_in_"):
                print(f"[SCALER] feature_names_in_ ({len(scaler.feature_names_in_)}):")
                for i, name in enumerate(scaler.feature_names_in_):
                    print(f"  {i+1:2d}. {name}")
            elif hasattr(scaler, "mean_"):
                print(f"[SCALER] Feature count derived from mean_: {len(scaler.mean_)}")
        except Exception as e:
            print(f"[SCALER ERROR] {e}")
    else:
        print("[SCALER] scaler.pkl NOT FOUND")

    print("-" * 60)

    # 2. Inspect best_model.pkl
    model_path = os.path.join("models", "best_model.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"[MODEL] Type: {type(model).__name__}")
            if hasattr(model, "n_features_in_"):
                print(f"[MODEL] n_features_in_: {model.n_features_in_}")
            if hasattr(model, "feature_names_in_"):
                print(f"[MODEL] feature_names_in_ ({len(model.feature_names_in_)}):")
                for i, name in enumerate(model.feature_names_in_):
                    print(f"  {i+1:2d}. {name}")
            if hasattr(model, "classes_"):
                print(f"[MODEL] Classes: {model.classes_}")
        except Exception as e:
            print(f"[MODEL ERROR] {e}")
    else:
        print("[MODEL] best_model.pkl NOT FOUND")

    print("-" * 60)

    # 3. Inspect feature_importance.json
    fi_path = os.path.join("models", "feature_importance.json")
    if os.path.exists(fi_path):
        try:
            with open(fi_path, "r") as f:
                fi_data = json.load(f)
            print(f"[FEATURE IMPORTANCE] Entries ({len(fi_data)}):")
            for k, v in list(fi_data.items())[:15]:
                print(f"  {k:<30}: {v}")
            if len(fi_data) > 15:
                print(f"  ... and {len(fi_data) - 15} more")
        except Exception as e:
            print(f"[FEATURE IMPORTANCE ERROR] {e}")
    else:
        print("[FEATURE IMPORTANCE] feature_importance.json NOT FOUND")

    print("-" * 60)

    # 4. Inspect metrics.json
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            print(f"[METRICS]:\n{json.dumps(metrics_data, indent=2)}")
        except Exception as e:
            print(f"[METRICS ERROR] {e}")
    else:
        print("[METRICS] metrics.json NOT FOUND")

    print("=" * 60)

if __name__ == "__main__":
    audit_artifacts()