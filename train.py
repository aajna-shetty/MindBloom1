"""
train.py
─────────────────────────────────────────────────────────────────────────────
Standalone script to train, evaluate, and save the autism risk screening model.

Usage:
    python train.py
"""

# 🔥 DEBUG: this runs immediately when file starts
print("🚀 TRAIN FILE STARTED")

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("📦 BEFORE IMPORT")
from app.model import train_and_evaluate
print("✅ AFTER IMPORT")

if __name__ == "__main__":
    print("📦 Calling train_and_evaluate()...")

    print("=" * 60)
    print("  Autism Risk Screening — Model Training")
    print("=" * 60)

    metrics = train_and_evaluate()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best model : {metrics.get('model_name', 'N/A')}")
    print(f"  Accuracy   : {metrics['accuracy']}")
    print(f"  Precision  : {metrics['precision']}")
    print(f"  Recall     : {metrics['recall']}")
    print(f"  F1 Score   : {metrics['f1_score']}")
    print(f"  CV F1      : {metrics.get('cv_f1_mean', 'N/A')} ± {metrics.get('cv_f1_std', 'N/A')}")
    print("=" * 60)

    print("\nSaved artefacts in ./models/:")
    print("  best_model.pkl")
    print("  scaler.pkl")
    print("  metrics.json")
    print("  feature_importance.json")

    print("\nNext step:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")