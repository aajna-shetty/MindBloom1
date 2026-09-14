"""
app/video/model.py
─────────────────────────────────────────────────────────────────────────────
Loads and runs the VIDEO Random Forest classifier — a separate model
artifact from the existing questionnaire model (models/best_model.pkl).

This is the ONLY place in the video module where a classification decision
is made. MediaPipe/DeepFace/OpenCV never classify anything themselves (see
package docstring in app/video/__init__.py).

IMPORTANT per §15/§17 of spec: if models/video_model.pkl does not exist
(because no labeled video-derived training dataset exists yet — see
train_video.py), this module does NOT fabricate a score. Callers must check
`is_available()` and surface `video_model_not_available` rather than a fake
prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from app.video.config import VIDEO_MODEL_PATH, VIDEO_SCALER_PATH
from app.video.features import load_feature_schema, feature_dict_to_vector

logger = logging.getLogger(__name__)


@dataclass
class VideoPrediction:
    video_score: float          # Rv = P(ASD-class), per §17
    video_confidence: float     # see §18 — max class probability, NOT a calibrated confidence
    predicted_class: int        # 0 = non-ASD, 1 = ASD


class VideoModel:
    """Lazily loads model + scaler + feature schema once, reused across requests."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._feature_order = None
        self._load_attempted = False

    def _try_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            self._model = joblib.load(VIDEO_MODEL_PATH)
            logger.info("Loaded video model from %s", VIDEO_MODEL_PATH)
        except FileNotFoundError:
            logger.warning(
                "Video model not found at %s. Video predictions will be "
                "unavailable until train_video.py is run with real labeled "
                "video-derived data.", VIDEO_MODEL_PATH,
            )
            self._model = None
        except Exception as exc:
            logger.error("Failed to load video model: %s", exc)
            self._model = None

        try:
            self._scaler = joblib.load(VIDEO_SCALER_PATH)
        except FileNotFoundError:
            self._scaler = None
        except Exception as exc:
            logger.error("Failed to load video scaler: %s", exc)
            self._scaler = None

        try:
            self._feature_order = load_feature_schema()
        except Exception as exc:
            logger.error("Failed to load video feature schema: %s", exc)
            self._feature_order = None

    def is_available(self) -> bool:
        self._try_load()
        return self._model is not None and self._feature_order is not None

    def predict(self, feats: dict) -> Optional[VideoPrediction]:
        """
        Run the video Random Forest on an aggregated feature dict (see
        app/video/features.build_feature_vector). Returns None if the model
        isn't available — caller must handle that as
        `video_model_not_available`, never substitute a placeholder score.
        """
        self._try_load()
        if self._model is None or self._feature_order is None:
            return None

        X = feature_dict_to_vector(feats, self._feature_order).reshape(1, -1)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        proba = self._model.predict_proba(X)[0]
        # Random Forest classes_ ordering isn't guaranteed to be [0, 1] by
        # position unless we check — read it explicitly to avoid a silent
        # class-index bug.
        classes = list(self._model.classes_)
        p_asd = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
        p_non_asd = 1.0 - p_asd

        predicted_class = int(self._model.predict(X)[0])
        # §18: raw RF probabilities are not necessarily calibrated. We use
        # max-class-probability as a simple, documented confidence proxy;
        # see fusion.py for how this maps to the fusion weight, and the
        # module docstring below for the calibration caveat.
        confidence = float(max(p_asd, p_non_asd))

        return VideoPrediction(
            video_score=p_asd,
            video_confidence=confidence,
            predicted_class=predicted_class,
        )


# Module-level singleton, mirroring how app/main.py loads the questionnaire
# model once at startup via app_state.
_video_model_singleton: Optional[VideoModel] = None


def get_video_model() -> VideoModel:
    global _video_model_singleton
    if _video_model_singleton is None:
        _video_model_singleton = VideoModel()
    return _video_model_singleton


# =========================================================================
# CALIBRATION NOTE (documented per §18, not implemented here)
# =========================================================================
# RandomForestClassifier.predict_proba() outputs the fraction of trees
# voting for each class, which is a common but NOT inherently
# well-calibrated probability estimate (it tends to be overconfident near
# 0/1 and underconfident near 0.5 for many datasets). If calibration proves
# necessary once real labeled video data exists, the standard approach is
# sklearn.calibration.CalibratedClassifierCV wrapped around the trained
# RandomForest (Platt scaling or isotonic regression), fit on a held-out
# calibration split. This is intentionally NOT implemented here because
# calibrating against a dataset that doesn't exist yet would itself be
# fabricated confidence.
