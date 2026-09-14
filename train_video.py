"""
train_video.py
─────────────────────────────────────────────────────────────────────────────
Trains the VIDEO Random Forest classifier from labeled video-derived
feature data. This is a SEPARATE script from train.py (which trains the
existing questionnaire model) and produces separate artifacts:

    models/video_model.pkl
    models/video_scaler.pkl
    models/video_metrics.json
    models/video_feature_schema.json  (also written by app.video.features)

──────────────────────────────────────────────────────────────────────────
REQUIRED INPUT DATA — DOES NOT EXIST IN THIS PROJECT YET
──────────────────────────────────────────────────────────────────────────
This script expects a CSV with exactly the 47 columns named in
app.video.features.VIDEO_FEATURES (in any order — this script reindexes to
the canonical order) plus a `target` column (0=non-ASD, 1=ASD).

None of the three CSVs currently in data/ (autism_adult_data_filled.csv,
autism_africa_baseline_1000_filled.csv, autism_africa_low_detection_2000_
filled.csv) are valid input here — they are questionnaire/demographic
datasets, not video-derived behavioral feature datasets, and this script
will refuse to run against them (see `validate_columns` below).

Usage:
    python train_video.py --data path/to/video_features_labeled.csv

If you don't have such a dataset yet, this script will simply tell you so
and exit — it will NOT fabricate one, and will NOT produce a
models/video_model.pkl. Until that file exists, the API returns
"video_model_not_available" (see app/video/model.py, app/video/service.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from app.video.config import (
    VIDEO_MODEL_PATH, VIDEO_SCALER_PATH, VIDEO_METRICS_PATH, MODELS_DIR,
)
from app.video.features import VIDEO_FEATURES, save_feature_schema

TARGET_COL = "target"


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in VIDEO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required video feature columns: "
            f"{missing}. This script only trains on data shaped like "
            "app.video.features.VIDEO_FEATURES + 'target' — the existing "
            "questionnaire CSVs in data/ are NOT valid input here."
        )
    if TARGET_COL not in df.columns:
        raise ValueError(f"Input CSV must contain a '{TARGET_COL}' column (0=non-ASD, 1=ASD).")

    unexpected_targets = set(df[TARGET_COL].unique()) - {0, 1}
    if unexpected_targets:
        raise ValueError(
            f"'{TARGET_COL}' column must be binary 0/1, found values: {unexpected_targets}"
        )


def train_and_evaluate(csv_path: str) -> dict:
    print(f"[train_video] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    validate_columns(df)

    X = df[VIDEO_FEATURES].astype(float).values
    y = df[TARGET_COL].astype(int).values
    print(f"[train_video] Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"[train_video] Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=min(5, np.bincount(y_train).min()) if np.bincount(y_train).min() >= 2 else 2,
                          shuffle=True, random_state=42)
    print("[train_video] Cross-validation ...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
    print(f"[train_video] CV F1: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")

    print("[train_video] Fitting final model ...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    classes = list(model.classes_)
    p_asd = y_proba[:, classes.index(1)] if 1 in classes else y_proba[:, -1]

    metrics = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_balance": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
    }

    # ROC-AUC requires both classes present in y_test.
    if len(set(y_test)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, p_asd)), 4)
    else:
        metrics["roc_auc"] = None
        print("[train_video] WARNING: only one class present in test split, ROC-AUC not computed.")

    print("\n── Video Model Evaluation ──")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, VIDEO_MODEL_PATH)
    joblib.dump(scaler, VIDEO_SCALER_PATH)
    save_feature_schema()
    with open(VIDEO_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[train_video] Saved: {VIDEO_MODEL_PATH}")
    print(f"[train_video] Saved: {VIDEO_SCALER_PATH}")
    print(f"[train_video] Saved: {VIDEO_METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MindBloom video Random Forest classifier.")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to labeled video-feature CSV (47 feature columns + 'target'). "
             "No default exists because no such dataset ships with this project.",
    )
    args = parser.parse_args()

    if not args.data:
        print(
            "\nNo --data path given, and no labeled video-derived training "
            "dataset exists in this project yet.\n\n"
            "This script cannot proceed without one: models/video_model.pkl "
            "will NOT be created, and the API will keep returning "
            "'video_model_not_available' for /video/analyze until you provide "
            "real labeled data of the shape:\n"
            f"    {len(VIDEO_FEATURES)} feature columns "
            "(see app/video/features.VIDEO_FEATURES) + a 'target' column.\n\n"
            "Run again with: python train_video.py --data path/to/your_video_features.csv\n"
        )
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"[train_video] File not found: {args.data}")
        sys.exit(1)

    train_and_evaluate(args.data)
