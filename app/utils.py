"""
utils.py
─────────────────────────────────────────────────────────────────────────────
Shared utility functions used by both model.py and main.py.

Includes:
  - risk_level()           → maps a probability to Low / Medium / High
  - top_features()         → returns top-N feature names with their importances
  - validate_features()    → checks the incoming JSON payload has required keys
  - build_feature_vector() → converts raw API input dict → ordered numpy array
  - combine_scores()       → combines questionnaire and video score
"""

import numpy as np
from typing import Dict, Any, List, Tuple


RISK_THRESHOLDS = {
    "Low": (0.00, 0.40),
    "Medium": (0.40, 0.70),
    "High": (0.70, 1.01),
}


def risk_level(score: float) -> str:
    """
    Convert a probability (0-1) into a human-readable risk band.
    """
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "High"


def top_features(feature_importance: Dict[str, float], n: int = 3) -> List[Dict[str, Any]]:
    """
    Return the top-N features sorted by importance score.
    """
    sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    return [
        {"feature": name, "importance": round(score, 4)}
        for name, score in sorted_fi[:n]
    ]


def validate_features(
    input_features: Dict[str, Any],
    required_features: List[str]
) -> Tuple[bool, List[str]]:
    """
    Check that every required feature is present in the input dict.
    """
    missing = [f for f in required_features if f not in input_features]
    return (len(missing) == 0, missing)


def build_feature_vector(
    input_features: Dict[str, Any],
    feature_list: List[str],
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Convert an arbitrary dict of feature values into a fixed-length numpy
    array aligned to feature_list.
    """
    vec = np.array(
        [float(input_features.get(f, fill_value)) for f in feature_list],
        dtype=np.float64
    )
    return vec


def combine_scores(
    questionnaire_score: float,
    video_score: float,
    q_weight: float = 0.6,
    v_weight: float = 0.4
) -> float:
    """
    Weighted combination of questionnaire probability and video score.

    final_score = q_weight * questionnaire_score + v_weight * video_score
    """
    score = (q_weight * questionnaire_score) + (v_weight * video_score)
    return float(np.clip(score, 0.0, 1.0))