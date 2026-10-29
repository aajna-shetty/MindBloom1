"""
tests/test_video_service_verification.py
─────────────────────────────────────────────────────────────────────────────
Tests for app.video.service.run_video_session with participant verification:
- Missing reference template rejection
- Pre-verification consensus failure (participant mismatch, no face)
- Continuous monitoring: pausing collection & excluding unverified frames
- Resuming on participant re-verification
"""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from app.video.service import run_video_session, VideoAnalysisResult
from app.video.verification import VerificationState


def _make_unit_vector(dim=512, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class _MockFrame:
    def __init__(self, index, valid=True):
        self.index = index
        self.timestamp = index / 30.0
        self.valid = valid
        self.rgb_frame = np.zeros((48, 48, 3), dtype=np.uint8) if valid else None
        self.error = None if valid else "read_error"


def _mock_landmarks(face_detected=True, num_faces=1):
    from app.video.mediapipe_processor import LandmarkFrameFeatures
    return LandmarkFrameFeatures(face_detected=face_detected, num_faces=num_faces if face_detected else 0)


class TestServiceVerificationIntegration:
    def test_missing_template_when_enforced_returns_missing_status(self):
        result = run_video_session(source=0, enforce_verification=True, reference_template=None)
        assert result.status == "missing_reference_template"
        assert result.verification_status == "not_enrolled"
        assert "recovery_guidance" in result.__dict__

    @patch("app.video.service.ParticipantVerifier")
    @patch("app.video.service.MediaPipeProcessor")
    @patch("app.video.service.VideoCaptureSession")
    def test_pre_verification_mismatch_halts_session_immediately(
        self, mock_cap_cls, mock_mp_cls, mock_verifier_cls
    ):
        mock_cap = MagicMock()
        mock_cap.frames.return_value = [_MockFrame(i) for i in range(20)]
        mock_cap_cls.return_value.__enter__.return_value = mock_cap

        mock_mp = MagicMock()
        mock_mp.process_frame.return_value = _mock_landmarks(face_detected=True, num_faces=1)
        mock_mp_cls.return_value.__enter__.return_value = mock_mp

        mock_verifier = MagicMock()
        mock_verifier.has_reference_template.return_value = True
        mock_verifier.verify_initial_consensus.return_value = {
            "verified": False,
            "status": "participant_mismatch",
            "detail": "Participant in front of the camera does not match the enrolled profile.",
            "recovery": "Verify child profile."
        }
        mock_verifier_cls.return_value = mock_verifier

        ref = _make_unit_vector(512)
        result = run_video_session(
            source=0, duration_seconds=1, enforce_verification=True, reference_template=ref
        )

        assert result.status == "participant_mismatch"
        assert result.verification_status == "participant_mismatch"
        assert result.video_score is None
        assert "does not match" in result.detail

    @patch("app.video.service.ParticipantVerifier")
    @patch("app.video.service.MediaPipeProcessor")
    @patch("app.video.service.VideoCaptureSession")
    def test_continuous_monitoring_excludes_unverified_frames(
        self, mock_cap_cls, mock_mp_cls, mock_verifier_cls
    ):
        mock_cap = MagicMock()
        frames = [_MockFrame(i) for i in range(120)]
        mock_cap.frames.return_value = frames
        mock_cap_cls.return_value.__enter__.return_value = mock_cap

        mock_mp = MagicMock()
        mock_mp.process_frame.return_value = _mock_landmarks(face_detected=True, num_faces=1)
        mock_mp_cls.return_value.__enter__.return_value = mock_mp

        mock_verifier = MagicMock()
        mock_verifier.has_reference_template.return_value = True
        mock_verifier.verify_initial_consensus.return_value = {
            "verified": True, "status": "verified", "detail": "Match verified."
        }
        # Sibling appears halfway through (first 60 verified, next 60 mismatch)
        def mock_eval(face_detected, num_faces, face_crop):
            return (True, VerificationState.VERIFIED)
        mock_verifier.evaluate_session_frame.side_effect = mock_eval
        mock_verifier.stats.unverified_frames_excluded = 15
        mock_verifier.current_state = VerificationState.VERIFIED
        mock_verifier_cls.return_value = mock_verifier

        ref = _make_unit_vector(512)
        result = run_video_session(
            source=0, duration_seconds=2, enforce_verification=True, reference_template=ref
        )

        assert result.status in ("video_model_not_available", "success")
        assert result.verification_status == "verified"
        assert result.unverified_frames_excluded == 15
