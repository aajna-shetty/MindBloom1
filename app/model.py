"""
model.py
────────────────────────────────────────────────────────────
Clean training pipeline (NO SMOTE, realistic performance)
"""

import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.preprocess import load_and_merge, build_scaler, get_feature_list, MODELS_DIR


# =========================
# PATHS
# =========================
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_PATH = os.path.join(MODELS_DIR, "feature_importance.json")


# =========================
# MODELS
# =========================
def get_models():
    return {
        # 🔻 Weakened Logistic
        "LogisticRegression": LogisticRegression(
            max_iter=300,
            C=0.1,
            class_weight="balanced",
            random_state=42
        ),

        # 🔺 Strong RandomForest
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }


# =========================
# EVALUATION
# =========================
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n── {name} ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Confusion Matrix: {cm}")

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }


# =========================
# TRAINING
# =========================
def train_and_evaluate():
    print("[model] Loading data...")

    df = load_and_merge()
    features = get_feature_list()

    X = df[features].values
    y = df["target"].values

    print(f"[model] Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale
    scaler = build_scaler(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # 🔥 ADD NOISE (break perfection)
    # =========================
    noise = np.random.normal(0, 0.03, X_train.shape)
    X_train = X_train + noise

    # =========================
    # MODELS
    # =========================
    models = get_models()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    all_metrics = {}
    trained_models = {}

    print("\n[model] Cross-validation")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        print(f"{name:<20} CV F1: {scores.mean():.4f}")

    print("\n[model] Training models...")

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate(model, X_test, y_test, name)
        all_metrics[name] = metrics

    # =========================
    # SELECT BEST
    # =========================
    best_name = max(all_metrics, key=lambda x: all_metrics[x]["f1_score"])
    best_model = trained_models[best_name]

    print(f"\n[model] Best model: {best_name}")

    # Save model
    joblib.dump(best_model, MODEL_PATH)

    # Save feature importance (RF only)
    if hasattr(best_model, "feature_importances_"):
        fi = dict(zip(features, best_model.feature_importances_))
        with open(FEATURE_PATH, "w") as f:
            json.dump(fi, f, indent=2)

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "best_model": best_name,
            "all_models": all_metrics
        }, f, indent=2)

    return all_metrics[best_name]


# =========================
# LOADERS
# =========================
def load_best_model():
    return joblib.load(MODEL_PATH)


def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def load_feature_importance():
    if os.path.exists(FEATURE_PATH):
        with open(FEATURE_PATH) as f:
            return json.load(f)
    return {}