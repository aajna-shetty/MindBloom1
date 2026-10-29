"""
app/video/schemas.py
─────────────────────────────────────────────────────────────────────────────
Pydantic request/response models for video screening, face enrollment options,
and comprehensive screening reports.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class VideoAnalyzeRequest(BaseModel):
    """
    Video screening request parameters.
    `source` can be a webcam index (e.g. '0') or a controlled local video path.
    `enforce_verification`: When True, verifies that the participant matches the
    child's enrolled reference template and enforces single-person presence.
    """
    child_id: int
    source: str = Field(
        default="0",
        description="Webcam index (e.g. '0') or an explicitly allowed local video path.",
    )
    duration_seconds: Optional[int] = Field(
        default=None,
        description="Overrides config.SESSION_DURATION_SECONDS for this session.",
    )
    enforce_verification: bool = Field(
        default=True,
        description="Enforces biometric participant verification against the enrolled child template.",
    )
    challenge_token: Optional[str] = None


class VideoAnalyzeResponse(BaseModel):
    status: str
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


class BiometricEnrollRequest(BaseModel):
    """
    Caregiver consent-governed enrollment request (Option B: Camera capture or JSON base64).
    """
    caregiver_consent: bool = Field(
        ...,
        description="Explicit caregiver consent required for processing biometric face representation.",
    )
    caregiver_name: str = Field(
        ...,
        min_length=2,
        description="Full name of caregiver granting consent.",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image (from camera capture or file preview).",
    )


class BiometricEnrollResponse(BaseModel):
    status: str
    child_id: int
    consent_obtained: bool
    enrolled_at: str
    template_model: str
    embedding_dimension: int
    detail: str


class BiometricStatusResponse(BaseModel):
    child_id: int
    is_enrolled: bool
    consent_obtained: bool
    consent_caregiver_name: Optional[str] = None
    enrolled_at: Optional[str] = None
    template_model: Optional[str] = None


class ScreeningReportResponse(BaseModel):
    child_id: int
    child_name: str
    child_age_months: int
    child_sex: str
    screening_id: int
    created_at: str
    result_type: str  # "questionnaire_only" | "hybrid"
    questionnaire_score: float
    video_score: Optional[float] = None
    video_confidence: Optional[float] = None
    hybrid_risk_score: float
    risk_tier: str
    verification_status: str
    models_evaluated: str
    explanation: str
    top_features: List[Dict[str, Any]] = []
    mediapipe_summary: Optional[Dict[str, Any]] = None
    vit_summary: Optional[Dict[str, Any]] = None
    screening_history: List[Dict[str, Any]] = []
    clinical_guidance: str
    disclaimer: str
