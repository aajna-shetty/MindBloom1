"""
app/video/service.py
─────────────────────────────────────────────────────────────────────────────
Orchestrates the full pipeline described in the spec:

    Webcam -> OpenCV -> frames -> MediaPipe -> landmarks -> behavioral
    features
                                -> DeepFace (sampled) -> expression features
    -> temporal aggregation -> 47-dim vector -> video Random Forest
    -> Rv + confidence

This is the single entry point app/main.py (or a future /video/analyze
route) should call. It does not touch the database or FastAPI directly —
that stays in main.py, keeping this module framework-agnostic and testable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from app.video.capture import VideoCaptureSession, CameraError
from app.video.mediapipe_processor import MediaPipeProcessor
from app.video.deepface_processor import DeepFaceProcessor
from app.video.session import SessionData
from app.video.features import build_feature_vector, VIDEO_FEATURES
from app.video.model import get_video_model
from app.video.config import SESSION_DURATION_SECONDS, DEEPFACE_EVERY_N_FRAMES

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisResult:
    status: str  # "success" | "insufficient_video_data" | "video_model_not_available" | "error"
    video_score: Optional[float] = None
    video_confidence: Optional[float] = None
    predicted_class: Optional[int] = None
    session_duration_seconds: Optional[float] = None
    valid_frame_ratio: Optional[float] = None
    feature_count: Optional[int] = None
    detail: Optional[str] = None


def run_video_session(
    source: int | str = 0,
    duration_seconds: int = SESSION_DURATION_SECONDS,
    deepface_every_n_frames: int = DEEPFACE_EVERY_N_FRAMES,
) -> VideoAnalysisResult:
    """
    Runs one full capture+analysis session synchronously and returns a
    VideoAnalysisResult. Never raises for expected failure modes (no camera,
    insufficient data, missing model) — those are returned as status codes
    per §11/§15/§22 of spec. Only truly unexpected errors propagate as
    status="error" with `detail` set, rather than crashing the caller.
    """
    session = SessionData()
    start = time.monotonic()

    try:
        with VideoCaptureSession(source=source, duration_seconds=duration_seconds) as cap:
            with MediaPipeProcessor() as mp_proc:
                try:
                    df_proc = DeepFaceProcessor()
                    deepface_enabled = True
                except ImportError as exc:
                    logger.warning("DeepFace unavailable, continuing without emotion features: %s", exc)
                    df_proc = None
                    deepface_enabled = False

                for frame in cap.frames():
                    if not frame.valid or frame.rgb_frame is None:
                        session.record_capture_failure()
                        continue

                    landmark_result = mp_proc.process_frame(frame.rgb_frame)
                    session.record_landmark_result(frame.timestamp, landmark_result)

                    if deepface_enabled and frame.index % max(1, deepface_every_n_frames) == 0:
                        df_result = df_proc.analyze_frame(frame.rgb_frame)
                        session.record_deepface_result(df_result)

    except CameraError as exc:
        logger.error("Camera error: %s", exc)
        return VideoAnalysisResult(status="error", detail=str(exc))
    except ImportError as exc:
        # mediapipe not installed, etc.
        logger.error("Missing dependency for video pipeline: %s", exc)
        return VideoAnalysisResult(status="error", detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception("Unexpected error during video session")
        return VideoAnalysisResult(status="error", detail=str(exc))

    elapsed = time.monotonic() - start
    quality = session.quality_metrics()

    if not session.has_sufficient_data():
        logger.warning(
            "Insufficient video data: valid_frame_ratio=%.3f valid_face_frames=%s",
            quality["valid_frame_ratio"], quality["valid_face_frames"],
        )
        return VideoAnalysisResult(
            status="insufficient_video_data",
            session_duration_seconds=round(elapsed, 2),
            valid_frame_ratio=quality["valid_frame_ratio"],
        )

    feats = build_feature_vector(session)

    video_model = get_video_model()
    if not video_model.is_available():
        logger.warning("Video model not available; returning video_model_not_available.")
        return VideoAnalysisResult(
            status="video_model_not_available",
            session_duration_seconds=round(elapsed, 2),
            valid_frame_ratio=quality["valid_frame_ratio"],
            feature_count=len(VIDEO_FEATURES),
        )

    prediction = video_model.predict(feats)
    if prediction is None:  # pragma: no cover - defensive, is_available() already checked
        return VideoAnalysisResult(
            status="video_model_not_available",
            session_duration_seconds=round(elapsed, 2),
            valid_frame_ratio=quality["valid_frame_ratio"],
            feature_count=len(VIDEO_FEATURES),
        )

    return VideoAnalysisResult(
        status="success",
        video_score=round(prediction.video_score, 4),
        video_confidence=round(prediction.video_confidence, 4),
        predicted_class=prediction.predicted_class,
        session_duration_seconds=round(elapsed, 2),
        valid_frame_ratio=quality["valid_frame_ratio"],
        feature_count=len(VIDEO_FEATURES),
    )
