"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Handles everything related to loading, cleaning, merging, and encoding the
three autism datasets into a single unified feature matrix ready for training.

Important:
- Removes leaky / post-diagnosis columns
- Standardizes target to 0 / 1
- Keeps only usable screening-time features
"""

import re
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ADULT_PATH = os.path.join(DATA_DIR, "autism_adult_data_filled.csv")
BASELINE_PATH = os.path.join(DATA_DIR, "autism_africa_baseline_1000_filled.csv")
LOWDET_PATH = os.path.join(DATA_DIR, "autism_africa_low_detection_2000_filled.csv")


# IMPORTANT:
# "result" is intentionally REMOVED because it is the AQ total score and can
# act like a rule-based shortcut for diagnosis.
UNIFIED_FEATURES = [
    # AQ-10 item-level features
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",

    # shared demographics
    "age",
    "gender",

    # adult-specific medical / history
    "jaundice",
    "family_history_asd",
    "used_app_before",

    # africa-specific prenatal / perinatal risk factors
    "preeclampsia",
    "birth_asphyxia",
    "preterm_birth",
    "low_birth_weight",
    "consanguineous_marriage",

    # developmental / behavioral markers
    "social_communication_delay",
    "repetitive_behaviors_score",
    "sensory_sensitivities",
    "first_words_months",
    "two_word_phrases_months",
    "eye_contact_age_months",
    "social_smile_months",

    # co-occurring conditions
    "intellectual_disability",
    "epilepsy",
    "adhd",
    "language_disorder",
    "motor_delay",
]

TARGET_COL = "target"


def _strip_bytes(val):
    """
    Strip Python byte-string repr like b'YES' → YES
    """
    if isinstance(val, str):
        m = re.match(r"^b'(.*)'$", val)
        if m:
            return m.group(1)
    return val


def _clean_adult(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize adult AQ dataset.
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(_strip_bytes)

    df = df.rename(columns={
        "jundice": "jaundice",
        "austim": "family_history_asd_str",
        "Class/ASD": "target_raw",
    })

    aq_cols = [f"A{i}_Score" for i in range(1, 11)]
    for c in aq_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")

    df["gender"] = df["gender"].astype(str).str.lower().map({"m": 1, "f": 0}).fillna(0)
    df["jaundice"] = df["jaundice"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0)
    df["family_history_asd"] = (
        df["family_history_asd_str"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0)
    )
    df["used_app_before"] = (
        df["used_app_before"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0)
    )

    df["target"] = df["target_raw"].astype(str).str.upper().map({"YES": 1, "NO": 0})

    for c in UNIFIED_FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    return df[UNIFIED_FEATURES + [TARGET_COL]].copy()


def _clean_africa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize Africa dataset.
    """
    df = df.rename(columns={"sex": "gender_str"})
    df["age"] = pd.to_numeric(df["age_months"], errors="coerce") / 12.0

    df["gender"] = df["gender_str"].astype(str).str.strip().str.lower().map(
        lambda x: 1 if x in ("m", "male") else 0
    )

    bool_cols = [
        "preeclampsia", "birth_asphyxia", "preterm_birth", "low_birth_weight",
        "family_history_asd", "consanguineous_marriage", "sensory_sensitivities",
        "intellectual_disability", "epilepsy", "adhd",
        "language_disorder", "motor_delay",
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "social_communication_delay",
        "repetitive_behaviors_score",
        "first_words_months",
        "two_word_phrases_months",
        "eye_contact_age_months",
        "social_smile_months",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["target"] = df["asd_status"].map({"Positive": 1, "Negative": 0})

    for c in UNIFIED_FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    return df[UNIFIED_FEATURES + [TARGET_COL]].copy()


def load_and_merge() -> pd.DataFrame:
    """
    Load all three datasets, clean them, merge them, and impute missing values.
    """
    print("[preprocess] Loading datasets...")
    adult = pd.read_csv(ADULT_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    lowdet = pd.read_csv(LOWDET_PATH)

    print(f"Adult: {adult.shape} | Baseline: {baseline.shape} | LowDet: {lowdet.shape}")

    adult_c = _clean_adult(adult)
    baseline_c = _clean_africa(baseline)
    lowdet_c = _clean_africa(lowdet)

    merged = pd.concat([adult_c, baseline_c, lowdet_c], ignore_index=True)
    print(f"[preprocess] Merged shape: {merged.shape}")

    merged = merged.dropna(subset=[TARGET_COL]).copy()
    merged[TARGET_COL] = merged[TARGET_COL].astype(int)

    numerical_cols = [
        "age",
        "social_communication_delay",
        "repetitive_behaviors_score",
        "first_words_months",
        "two_word_phrases_months",
        "eye_contact_age_months",
        "social_smile_months",
    ]
    binary_cols = [c for c in UNIFIED_FEATURES if c not in numerical_cols]

    for c in numerical_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
            merged[c] = merged[c].fillna(merged[c].median())

    for c in binary_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    for c in UNIFIED_FEATURES:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(float)

    print(f"[preprocess] Final shape: {merged.shape}")
    print(f"[preprocess] Target distribution:\n{merged[TARGET_COL].value_counts()}")

    return merged


def build_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit and save a StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, path)
    print(f"[preprocess] Scaler saved → {path}")
    return scaler


def load_scaler() -> StandardScaler:
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    return joblib.load(path)


def get_feature_list() -> list:
    return UNIFIED_FEATURES