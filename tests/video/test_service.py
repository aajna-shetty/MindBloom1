"""
Exercises app.video.service.run_video_session's status-routing logic
(§11/§15/§22): camera failure -> error, too little usable data ->
insufficient_video_data, no trained model -> video_model_not_available.

Mocks VideoCaptureSession and MediaPipeProcessor so these tests don't need
a real camera, real mediapipe runtime, or a real video_model.pkl.
"""
from unittest.mock import patch, MagicMock
import numpy as np

from app.video.capture import CameraError
from app.video.service import run_video_session


class _FakeFrame:
    def __init__(self, index, valid=True):
        self.index = index
        self.timestamp = index / 30.0
        self.valid = valid
        self.rgb_frame = np.zeros((10, 10, 3), dtype=np.uint8) if valid else None
        self.error = None if valid else "capture_read_failed"


def _fake_landmark_result(face_detected=True):
    from app.video.mediapipe_processor import LandmarkFrameFeatures
    return LandmarkFrameFeatures(face_detected=face_detected, num_faces=1 if face_detected else 0)


class TestCameraFailure:
    @patch("app.video.service.VideoCaptureSession")
    def test_camera_error_returns_error_status(self, mock_session_cls):
        mock_session_cls.return_value.__enter__.side_effect = CameraError("no camera found")
        result = run_video_session(source=0, duration_seconds=1)
        assert result.status == "error"
        assert "no camera" in result.detail


class TestInsufficientData:
    @patch("app.video.service.MediaPipeProcessor")
    @patch("app.video.service.VideoCaptureSession")
    def test_too_few_valid_frames_returns_insufficient(self, mock_cap_cls, mock_mp_cls):
        mock_cap = MagicMock()
        mock_cap.frames.return_value = [_FakeFrame(i) for i in range(5)]  # far below MIN_VALID_FRAME_COUNT
        mock_cap_cls.return_value.__enter__.return_value = mock_cap

        mock_mp = MagicMock()
        mock_mp.process_frame.return_value = _fake_landmark_result(face_detected=True)
        mock_mp_cls.return_value.__enter__.return_value = mock_mp

        result = run_video_session(source=0, duration_seconds=1)
        assert result.status == "insufficient_video_data"
        assert result.video_score is None


class TestModelUnavailable:
    @patch("app.video.service.get_video_model")
    @patch("app.video.service.MediaPipeProcessor")
    @patch("app.video.service.VideoCaptureSession")
    def test_sufficient_data_but_no_model_returns_model_not_available(
        self, mock_cap_cls, mock_mp_cls, mock_get_model,
    ):
        mock_cap = MagicMock()
        mock_cap.frames.return_value = [_FakeFrame(i) for i in range(120)]
        mock_cap_cls.return_value.__enter__.return_value = mock_cap

        mock_mp = MagicMock()
        mock_mp.process_frame.return_value = _fake_landmark_result(face_detected=True)
        mock_mp_cls.return_value.__enter__.return_value = mock_mp

        mock_model = MagicMock()
        mock_model.is_available.return_value = False
        mock_get_model.return_value = mock_model

        result = run_video_session(source=0, duration_seconds=1)
        assert result.status == "video_model_not_available"
        assert result.feature_count == 47

    @patch("app.video.service.get_video_model")
    @patch("app.video.service.MediaPipeProcessor")
    @patch("app.video.service.VideoCaptureSession")
    def test_sufficient_data_with_model_returns_success(
        self, mock_cap_cls, mock_mp_cls, mock_get_model,
    ):
        from app.video.model import VideoPrediction

        mock_cap = MagicMock()
        mock_cap.frames.return_value = [_FakeFrame(i) for i in range(120)]
        mock_cap_cls.return_value.__enter__.return_value = mock_cap

        mock_mp = MagicMock()
        mock_mp.process_frame.return_value = _fake_landmark_result(face_detected=True)
        mock_mp_cls.return_value.__enter__.return_value = mock_mp

        mock_model = MagicMock()
        mock_model.is_available.return_value = True
        mock_model.predict.return_value = VideoPrediction(
            video_score=0.42, video_confidence=0.77, predicted_class=0,
        )
        mock_get_model.return_value = mock_model

        result = run_video_session(source=0, duration_seconds=1)
        assert result.status == "success"
        assert result.video_score == 0.42
        assert result.video_confidence == 0.77
        assert result.predicted_class == 0
