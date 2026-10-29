"""
app/video/deepface_processor.py
─────────────────────────────────────────────────────────────────────────────
DeepFace's ONLY responsibility here: facial-expression / affective analysis
on sampled frames (not classification of autism — see model.py).

DeepFace is a heavier model than MediaPipe Face Mesh, so we do NOT run it on
every frame (that would make live webcam processing unusably slow). It runs
every `DEEPFACE_EVERY_N_FRAMES` frames (config.py), which the caller
(session.py / service.py) is responsible for gating.

Emotion categories (fixed, matches DeepFace's default `emotion` output):
    neutral, happy, sad, surprise, fear, disgust, angry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

try:
    from deepface import DeepFace
except Exception as exc:  # pragma: no cover
    DeepFace = None
    _import_error = exc

from app.video.config import (
    EMOTION_LABELS, DEEPFACE_DETECTOR_BACKEND, DEEPFACE_ACTIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class DeepFaceFrameResult:
    success: bool
    dominant_emotion: Optional[str] = None
    emotion_scores: Optional[Dict[str, float]] = None  # normalized to sum to 1
    error: Optional[str] = None


class DeepFaceProcessor:
    """
    Stateless wrapper (DeepFace manages its own model caching internally).
    Kept as a class for symmetry with MediaPipeProcessor and so a detector
    backend / actions choice can be configured once per session.
    """

    def __init__(
        self,
        detector_backend: str = DEEPFACE_DETECTOR_BACKEND,
        actions=None,
    ):
        if DeepFace is None:
            raise ImportError(
                "deepface is not installed. Add it to requirements.txt and "
                "`pip install deepface`."
            ) from _import_error
        self.detector_backend = detector_backend
        self.actions = actions or DEEPFACE_ACTIONS

    def analyze_frame(self, rgb_frame: np.ndarray) -> DeepFaceFrameResult:
        """
        Run DeepFace emotion analysis on a single RGB frame. Never raises —
        failures (no face, DeepFace internal error, unsupported backend,
        etc.) are captured in the returned result so the caller can just
        tally successes/failures per §11 of the spec.
        """
        try:
            # DeepFace expects BGR-ish/np arrays fine either way for its
            # internal detectors, but to be explicit and avoid a subtle
            # color-channel bug, we pass RGB and rely on enforce_detection
            # to fail cleanly (not crash) when no face is present.
            result = DeepFace.analyze(
                img_path=rgb_frame,
                actions=self.actions,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                silent=True,
            )
        except Exception as exc:
            return DeepFaceFrameResult(success=False, error=str(exc))

        # DeepFace.analyze can return a list (one dict per detected face) or
        # a single dict depending on version; normalize to "first face".
        if isinstance(result, list):
            if not result:
                return DeepFaceFrameResult(success=False, error="no_face_returned")
            result = result[0]

        try:
            raw_scores = dict(result.get("emotion", {}))
            dominant = result.get("dominant_emotion")
        except Exception as exc:
            return DeepFaceFrameResult(success=False, error=f"unexpected_result_shape:{exc}")

        if not raw_scores:
            return DeepFaceFrameResult(success=False, error="empty_emotion_scores")

        # DeepFace returns percentages (0-100) that should sum to ~100;
        # normalize defensively to an exact probability distribution over
        # our fixed EMOTION_LABELS ordering, filling any missing label with
        # 0.0 rather than guessing.
        total = sum(raw_scores.values())
        normalized = {
            label: (float(raw_scores.get(label, 0.0)) / total if total > 0 else 0.0)
            for label in EMOTION_LABELS
        }

        return DeepFaceFrameResult(
            success=True,
            dominant_emotion=dominant,
            emotion_scores=normalized,
        )
