"""
app/video/schemas.py
─────────────────────────────────────────────────────────────────────────────
Pydantic request/response models for the new POST /video/analyze endpoint.
Kept separate from app/main.py's existing Pydantic models so the video
module stays a self-contained addition (§20 of spec: don't unnecessarily
touch existing definitions).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class VideoAnalyzeRequest(BaseModel):
    """
    `source` lets the same endpoint be used for a live local webcam session
    on the machine running the API (source=0, or a device index), OR for a
    pre-recorded file path already accessible to the server (e.g. uploaded
    separately). This mirrors VideoCaptureSession's source parameter.

    In a typical deployment the actual webcam capture happens client-side
    (browser/app) and only the resulting video file is sent to the server;
    wiring an upload endpoint is a follow-up integration step and is out of
    scope for this module, which focuses on the extraction/scoring
    pipeline itself.
    """
    child_id: int
    source: str = Field(
        default="0",
        description="Webcam index (e.g. '0') or a server-accessible video file path.",
    )
    duration_seconds: Optional[int] = Field(
        default=None,
        description="Overrides config.SESSION_DURATION_SECONDS for this session (e.g. dev testing).",
    )


class VideoAnalyzeResponse(BaseModel):
    status: str  # "success" | "insufficient_video_data" | "video_model_not_available" | "error"
    video_score: Optional[float] = None
    video_confidence: Optional[float] = None
    predicted_class: Optional[int] = None
    session_duration_seconds: Optional[float] = None
    valid_frame_ratio: Optional[float] = None
    feature_count: Optional[int] = None
    detail: Optional[str] = None
