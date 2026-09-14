"""
app/video/capture.py

Handles webcam/video-file capture using OpenCV.

Responsibilities:
- Open webcam or video file
- Read frames continuously
- Convert BGR -> RGB
- Track frame dimensions and statistics
- Stop after the configured duration
- Release the camera safely
- Handle bad frames without crashing the whole pipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from app.video.config import SESSION_DURATION_SECONDS, TARGET_FPS

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Raised when the capture device cannot be opened."""


@dataclass
class FrameResult:
    """Result of processing one captured frame."""

    index: int
    timestamp: float
    rgb_frame: Optional[np.ndarray]

    width: int = 0
    height: int = 0
    valid: bool = True
    error: Optional[str] = None


@dataclass
class CaptureStats:
    """Statistics collected during a capture session."""

    total_frames: int = 0
    invalid_frames: int = 0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


class VideoCaptureSession:
    """
    Thin wrapper around cv2.VideoCapture.

    Example:

        with VideoCaptureSession(source=0, duration_seconds=10) as session:
            for frame in session.frames():
                ...
    """

    def __init__(
        self,
        source: int | str = 0,
        duration_seconds: int = SESSION_DURATION_SECONDS,
        target_fps: int = TARGET_FPS,
    ):
        self.source = source

        # Main duration property used by the current implementation.
        self.duration_seconds = max(1, int(duration_seconds))

        # Compatibility alias.
        # Some older code may expect `session.duration`.
        self.duration = self.duration_seconds

        self.target_fps = max(1, int(target_fps))

        self._min_frame_interval = 1.0 / self.target_fps

        self._cap: Optional[cv2.VideoCapture] = None

        self.stats = CaptureStats()

        self.width = 0
        self.height = 0

    # ============================================================
    # CONTEXT MANAGER
    # ============================================================

    def __enter__(self) -> "VideoCaptureSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

        # Do not suppress exceptions.
        return False

    # ============================================================
    # OPEN CAMERA / VIDEO
    # ============================================================

    def open(self) -> None:
        """
        Open the configured webcam or video file.
        """

        logger.info(
            "Attempting to open video source: %s",
            self.source,
        )

        try:
            cap = cv2.VideoCapture(self.source)
        except Exception as exc:
            raise CameraError(
                f"OpenCV could not create video capture for "
                f"source '{self.source}': {exc}"
            ) from exc

        if not cap.isOpened():
            cap.release()

            raise CameraError(
                f"Could not open video source '{self.source}'. "
                f"Check that the camera is connected, camera permissions "
                f"are enabled, or the video file path is valid."
            )

        self._cap = cap

        self.width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ) or 0

        self.height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ) or 0

        self.stats = CaptureStats(
            started_at=time.monotonic()
        )

        logger.info(
            "VideoCaptureSession opened "
            "source=%s resolution=%sx%s duration=%ss target_fps=%s",
            self.source,
            self.width,
            self.height,
            self.duration_seconds,
            self.target_fps,
        )

    # ============================================================
    # RELEASE CAMERA
    # ============================================================

    def release(self) -> None:
        """
        Release the webcam/video source safely.
        """

        if self._cap is None:
            return

        try:
            self._cap.release()

        except Exception as exc:
            logger.warning(
                "Error while releasing video capture: %s",
                exc,
            )

        finally:
            self._cap = None

            self.stats.ended_at = time.monotonic()

            logger.info(
                "VideoCaptureSession released. "
                "total_frames=%s invalid_frames=%s",
                self.stats.total_frames,
                self.stats.invalid_frames,
            )

    # ============================================================
    # FRAME GENERATOR
    # ============================================================

    def frames(self) -> Iterator[FrameResult]:
        """
        Read frames until:

        1. Session duration is reached
        2. A video file reaches its end
        3. The capture source becomes unavailable

        Individual bad frames do not crash the session.
        """

        if self._cap is None:
            raise CameraError(
                "Video capture session has not been opened. "
                "Use `with VideoCaptureSession(...)` or call `.open()` first."
            )

        start = (
            self.stats.started_at
            if self.stats.started_at is not None
            else time.monotonic()
        )

        index = 0
        last_read_time = 0.0

        logger.info(
            "Starting frame capture for %s seconds...",
            self.duration_seconds,
        )

        while True:

            # ----------------------------------------------------
            # Check session duration
            # ----------------------------------------------------

            now = time.monotonic()

            elapsed = now - start

            if elapsed >= self.duration_seconds:

                logger.info(
                    "Session duration reached: %.2f seconds",
                    elapsed,
                )

                break

            # ----------------------------------------------------
            # Frame pacing
            # ----------------------------------------------------

            since_last_read = now - last_read_time

            if since_last_read < self._min_frame_interval:

                sleep_time = (
                    self._min_frame_interval
                    - since_last_read
                )

                time.sleep(max(0, sleep_time))

            # ----------------------------------------------------
            # Read frame
            # ----------------------------------------------------

            try:

                ret, frame_bgr = self._cap.read()

            except Exception as exc:

                logger.warning(
                    "OpenCV read() raised an exception: %s",
                    exc,
                )

                ret = False
                frame_bgr = None

            last_read_time = time.monotonic()

            timestamp = last_read_time - start

            # ----------------------------------------------------
            # Invalid frame
            # ----------------------------------------------------

            if not ret or frame_bgr is None:

                self.stats.total_frames += 1
                self.stats.invalid_frames += 1

                logger.warning(
                    "Frame %s could not be captured.",
                    index,
                )

                yield FrameResult(
                    index=index,
                    timestamp=timestamp,
                    rgb_frame=None,
                    valid=False,
                    error="capture_read_failed",
                )

                index += 1

                # A video file that reaches the end will continue
                # returning False. Stop instead of looping forever.
                if isinstance(self.source, str):
                    logger.info(
                        "Video file reached its end."
                    )
                    break

                # For webcams, continue trying to read.
                continue

            # ----------------------------------------------------
            # Convert BGR -> RGB
            # ----------------------------------------------------

            try:

                rgb_frame = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                height, width = frame_bgr.shape[:2]

            except Exception as exc:

                self.stats.total_frames += 1
                self.stats.invalid_frames += 1

                logger.warning(
                    "Frame conversion failed: %s",
                    exc,
                )

                yield FrameResult(
                    index=index,
                    timestamp=timestamp,
                    rgb_frame=None,
                    valid=False,
                    error="frame_conversion_failed",
                )

                index += 1

                continue

            # ----------------------------------------------------
            # Valid frame
            # ----------------------------------------------------

            self.stats.total_frames += 1

            yield FrameResult(
                index=index,
                timestamp=timestamp,
                rgb_frame=rgb_frame,
                width=width,
                height=height,
                valid=True,
            )

            index += 1

        logger.info(
            "Frame capture finished. "
            "total_frames=%s invalid_frames=%s",
            self.stats.total_frames,
            self.stats.invalid_frames,
        )