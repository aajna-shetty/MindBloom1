"""
app/video/service.py
─────────────────────────────────────────────────────────────────────────────
Orchestrates the full pipeline with participant identity verification and
multimodal feature extraction:

    Webcam / Video File
        ↓
    Participant Pre-Verification (Multi-frame consensus against reference template)
        ↓
    Single-Person Camera Enforcement:
      - 0 people -> PAUSE collection
      - >1 people -> PAUSE collection ("Multiple people detected")
      - 1 person -> Verify identity -> IF MATCH -> Collect features
        ↓
    [MediaPipe] -> 47-dim geometric behavioral vector
    [ViT Model] -> Affective expression analysis & documented interpretation
    [DeepFace]  -> Sampled emotion probabilities
    [Unverified intervals] -> Discarded & excluded from aggregation
        ↓
    Video Random Forest Classifier -> Rv + confidence
        ↓
    VideoAnalysisResult with behavioral + ViT summaries
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from app.video.capture import VideoCaptureSession, CameraError
from app.video.mediapipe_processor import MediaPipeProcessor
from app.video.deepface_processor import DeepFaceProcessor
from app.video.emotion_processor import get_emotion_processor
from app.video.session import SessionData
from app.video.features import build_feature_vector, VIDEO_FEATURES
from app.video.model import get_video_model
from app.video.config import SESSION_DURATION_SECONDS, DEEPFACE_EVERY_N_FRAMES
from app.video.verification import (
    ParticipantVerifier,
    VerificationState,
    VERIFICATION_CHECK_INTERVAL,
)

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisResult:
    status: str  # "success" | "insufficient_video_data" | "video_model_not_available" | "participant_mismatch" | "identity_uncertain" | "no_face_detected" | "multiple_faces_detected" | "missing_reference_template" | "error"
    video_score: Optional[float] = None
    video_confidence: Optional[float] = None
    predicted_class: Optional[int] = None
    session_duration_seconds: Optional[float] = None
    valid_frame_ratio: Optional[float] = None
    feature_count: Optional[int] = None
    verification_status: Optional[str] = None
    unverified_frames_excluded: Optional[int] = None
    mediapipe_summary: Optional[Dict[str, Any]] = None
    vit_dominant_emotion: Optional[str] = None
    vit_emotion_scores: Optional[Dict[str, float]] = None
    vit_interpretation: Optional[str] = None
    detail: Optional[str] = None
    recovery_guidance: Optional[str] = None
    liveness_status: Optional[str] = None


def run_video_session(
    source: int | str = 0,
    duration_seconds: int = SESSION_DURATION_SECONDS,
    deepface_every_n_frames: int = DEEPFACE_EVERY_N_FRAMES,
    reference_template: Optional[np.ndarray] = None,
    enforce_verification: bool = False,
    verification_check_every_n_frames: int = VERIFICATION_CHECK_INTERVAL,
    challenge_action: Optional[str] = None,
) -> VideoAnalysisResult:
    """
    Runs one capture+analysis session with continuous participant identity verification,
    single-person camera enforcement, and ViT visual-model integration.
    """
    session = SessionData()
    start = time.monotonic()
    verifier = ParticipantVerifier(reference_template=reference_template)
    vit_processor = get_emotion_processor()
    latest_vit_result: Optional[Dict[str, Any]] = None

    # 1. Enforce enrollment if verification is requested
    if enforce_verification and not verifier.has_reference_template():
        logger.warning("Verification requested but child has no enrolled biometric template.")
        return VideoAnalysisResult(
            status="missing_reference_template",
            verification_status="not_enrolled",
            detail="Participant has not enrolled a reference template with consent.",
            recovery_guidance="Please enroll a reference photo for the child profile before conducting video analysis, or proceed with questionnaire-only screening."
        )

    try:
        with VideoCaptureSession(source=source, duration_seconds=duration_seconds) as cap:
            with MediaPipeProcessor() as mp_proc:
                try:
                    df_proc = DeepFaceProcessor()
                    deepface_enabled = True
                except Exception as exc:
                    logger.warning("DeepFace unavailable, continuing without emotion features: %s", exc)
                    df_proc = None
                    deepface_enabled = False

                # --------------------------------------------------------
                # Pre-Assessment Verification: Multi-Frame Consensus Check
                # --------------------------------------------------------
                frame_generator = cap.frames()

                if enforce_verification:
                    pre_check_frames: List[np.ndarray] = []
                    pre_check_observations = []
                    collected = 0
                    for frame in frame_generator:
                        if not frame.valid or frame.rgb_frame is None:
                            session.record_capture_failure()
                            continue

                        lm_pre = mp_proc.process_frame(frame.rgb_frame)
                        pre_check_frames.append(frame.rgb_frame)
                        pre_check_observations.append(lm_pre)
                        if lm_pre.face_detected:
                            collected += 1

                        if collected >= 8 or len(pre_check_frames) >= 12:
                            break

                    consensus = verifier.verify_initial_consensus(pre_check_frames, observations=pre_check_observations, challenge_action=challenge_action)
                    if not consensus["verified"]:
                        logger.warning(
                            "Pre-assessment verification failed: %s (%s)",
                            consensus["status"], consensus["detail"]
                        )
                        return VideoAnalysisResult(
                            status=consensus["status"],
                            verification_status=consensus["status"],
                            detail=consensus["detail"],
                            recovery_guidance=consensus.get("recovery"),
                            liveness_status=consensus.get("liveness_status"),
                            unverified_frames_excluded=len(pre_check_frames),
                        )
                    logger.info("Pre-assessment verification PASSED.")

                # --------------------------------------------------------
                # Assessment Session Loop: Single-Person & Identity Gating
                # --------------------------------------------------------
                current_allow_collection = True

                for frame in frame_generator:
                    if not frame.valid or frame.rgb_frame is None:
                        session.record_capture_failure()
                        continue

                    # MediaPipe Landmark Detection
                    landmark_result = mp_proc.process_frame(frame.rgb_frame)

                    # Periodic Single-Person and Identity Check
                    if enforce_verification and frame.index % max(1, verification_check_every_n_frames) == 0:
                        verification_args = {
                            "face_detected": landmark_result.face_detected,
                            "num_faces": landmark_result.num_faces,
                            "face_crop": frame.rgb_frame,
                        }
                        if landmark_result.person_count is not None:
                            verification_args["person_count"] = landmark_result.person_count
                        allow, state = verifier.evaluate_session_frame(**verification_args)
                        current_allow_collection = allow
                        if not allow:
                            logger.debug("Participant monitoring paused: state=%s", state.value)

                    # Gating: If unverified or multiple people, exclude frame
                    if enforce_verification and not current_allow_collection:
                        verifier.stats.unverified_frames_excluded += 1
                        continue

                    # Verified single-participant interval: record measurements
                    session.record_landmark_result(frame.timestamp, landmark_result)

                    # Sample ViT visual-model affective analysis
                    if frame.index % max(1, deepface_every_n_frames * 2) == 0 and landmark_result.face_detected:
                        try:
                            latest_vit_result = vit_processor.analyze(frame.rgb_frame)
                        except Exception as vit_err:
                            logger.debug("ViT analysis blip: %s", vit_err)

                    if deepface_enabled and frame.index % max(1, deepface_every_n_frames) == 0:
                        df_result = df_proc.analyze_frame(frame.rgb_frame)
                        session.record_deepface_result(df_result)

    except CameraError as exc:
        logger.error("Camera error: %s", exc)
        return VideoAnalysisResult(status="error", detail=str(exc))
    except ImportError as exc:
        logger.error("Missing dependency for video pipeline: %s", exc)
        return VideoAnalysisResult(status="error", detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during video session")
        return VideoAnalysisResult(status="error", detail=str(exc))

    elapsed = time.monotonic() - start
    quality = session.quality_metrics()

    # Data sufficiency check
    if not session.has_sufficient_data():
        logger.warning(
            "Insufficient video data: valid_frame_ratio=%.3f valid_face_frames=%s unverified_excluded=%s",
            quality["valid_frame_ratio"], quality["valid_face_frames"],
            verifier.stats.unverified_frames_excluded
        )
        return VideoAnalysisResult(
            status="insufficient_video_data",
            session_duration_seconds=round(elapsed, 2),
            valid_frame_ratio=quality["valid_frame_ratio"],
            verification_status=verifier.current_state.value if enforce_verification else "not_enforced",
            unverified_frames_excluded=verifier.stats.unverified_frames_excluded,
            detail="Insufficient verified video data collected during assessment."
        )

    feats = build_feature_vector(session)

    # Build concise MediaPipe behavioral summary
    mediapipe_summary = {
        "gaze_horizontal_mean": round(float(feats.get("gaze_horizontal_mean", 0.5)), 3),
        "gaze_looking_away_proportion": round(float(feats.get("gaze_looking_away_proportion", 0.0)), 3),
        "blink_rate_per_min": round(float(feats.get("blink_rate_per_min", 0.0)), 1),
        "ear_mean": round(float(feats.get("ear_mean", 0.0)), 3),
        "head_movement_magnitude_mean": round(float(feats.get("head_movement_magnitude_mean", 0.0)), 2),
        "facial_movement_frequency": round(float(feats.get("facial_movement_frequency", 0.0)), 3),
    }

    # Extract ViT output summary
    vit_dom = latest_vit_result.get("dominant_emotion", "neutral") if latest_vit_result else "neutral"
    vit_scores = latest_vit_result.get("emotion_scores", {}) if latest_vit_result else {}
    vit_interp = latest_vit_result.get("interpretation", "Affective expressivity recorded.") if latest_vit_result else "Affective expressivity recorded."

    video_model = get_video_model()
    if not video_model.is_available():
        logger.warning("Video model not available; returning video_model_not_available.")
        return VideoAnalysisResult(
            status="video_model_not_available",
            session_duration_seconds=round(elapsed, 2),
            valid_frame_ratio=quality["valid_frame_ratio"],
            feature_count=len(VIDEO_FEATURES),
            verification_status=verifier.current_state.value if enforce_verification else "not_enforced",
            unverified_frames_excluded=verifier.stats.unverified_frames_excluded,
            mediapipe_summary=mediapipe_summary,
            vit_dominant_emotion=vit_dom,
            vit_emotion_scores=vit_scores,
            vit_interpretation=vit_interp,
            detail="Video behavioral features extracted successfully. Note: video_model.pkl is unavailable as documented."
        )

    prediction = video_model.predict(feats)
    if prediction is None:
        return VideoAnalysisResult(
            status="video_model_not_available",
            session_duration_seconds=round(elapsed, 2),
            valid_frame_ratio=quality["valid_frame_ratio"],
            feature_count=len(VIDEO_FEATURES),
            verification_status=verifier.current_state.value if enforce_verification else "not_enforced",
            unverified_frames_excluded=verifier.stats.unverified_frames_excluded,
            mediapipe_summary=mediapipe_summary,
            vit_dominant_emotion=vit_dom,
            vit_emotion_scores=vit_scores,
            vit_interpretation=vit_interp,
        )

    return VideoAnalysisResult(
        status="success",
        video_score=round(prediction.video_score, 4),
        video_confidence=round(prediction.video_confidence, 4),
        predicted_class=prediction.predicted_class,
        session_duration_seconds=round(elapsed, 2),
        valid_frame_ratio=quality["valid_frame_ratio"],
        feature_count=len(VIDEO_FEATURES),
        verification_status="verified" if enforce_verification else "not_enforced",
        unverified_frames_excluded=verifier.stats.unverified_frames_excluded,
        mediapipe_summary=mediapipe_summary,
        vit_dominant_emotion=vit_dom,
        vit_emotion_scores=vit_scores,
        vit_interpretation=vit_interp,
    )
