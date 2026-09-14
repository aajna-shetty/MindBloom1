from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.video.capture import VideoCaptureSession, CameraError


def _fake_frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


class TestCameraOpenFailure:
    @patch("app.video.capture.cv2.VideoCapture")
    def test_raises_camera_error_when_device_cannot_open(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc.return_value = mock_cap

        session = VideoCaptureSession(source=0, duration_seconds=1)
        with pytest.raises(CameraError):
            session.open()
        mock_cap.release.assert_called_once()


class TestFrameReading:
    @patch("app.video.capture.cv2.VideoCapture")
    def test_single_bad_frame_does_not_crash_session(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 64
        # First read fails, second succeeds, then we let duration run out.
        mock_cap.read.side_effect = [
            (False, None),
            (True, _fake_frame()),
        ] + [(True, _fake_frame())] * 50
        mock_vc.return_value = mock_cap

        session = VideoCaptureSession(source=0, duration_seconds=0.05, target_fps=1000)
        session.open()
        results = list(session.frames())
        session.release()

        assert results[0].valid is False
        assert results[0].error == "capture_read_failed"
        assert any(r.valid for r in results)
        assert session.stats.invalid_frames >= 1

    @patch("app.video.capture.cv2.VideoCapture")
    def test_valid_frame_has_rgb_conversion_and_dimensions(self, mock_vc):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 64
        mock_cap.read.return_value = (True, _fake_frame())
        mock_vc.return_value = mock_cap

        session = VideoCaptureSession(source=0, duration_seconds=0.02, target_fps=1000)
        session.open()
        results = list(session.frames())
        session.release()

        assert len(results) > 0
        first = results[0]
        assert first.valid is True
        assert first.rgb_frame is not None
        assert first.rgb_frame.shape[2] == 3
        assert first.width == 64 and first.height == 48

    @patch("app.video.capture.cv2.VideoCapture")
    def test_release_is_safe_to_call_when_never_opened(self, mock_vc):
        session = VideoCaptureSession(source=0)
        session.release()  # should not raise
