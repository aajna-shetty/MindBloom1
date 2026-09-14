"""
Tests the geometry/math methods of MediaPipeProcessor using synthetic
landmark arrays, bypassing __init__ (which requires the mediapipe package
and a live FaceMesh instance) via object.__new__. This lets us test the
EAR/gaze/head-pose/movement CALCULATIONS in isolation from the mediapipe
dependency and from any camera.
"""
import numpy as np
import pytest

from app.video.mediapipe_processor import (
    MediaPipeProcessor, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    LEFT_EYE_BOX_IDX, RIGHT_EYE_BOX_IDX, LEFT_IRIS_CENTER_IDX, RIGHT_IRIS_CENTER_IDX,
    MOUTH_IDX, LEFT_EYEBROW_IDX, RIGHT_EYEBROW_IDX,
)


def make_bare_processor(refine_landmarks=True) -> MediaPipeProcessor:
    """Construct a MediaPipeProcessor without running __init__ (which needs
    the real mediapipe package + camera-independent FaceMesh model)."""
    proc = object.__new__(MediaPipeProcessor)
    proc._refine_landmarks = refine_landmarks
    proc._prev_landmarks = None
    proc._consec_closed_frames = 0
    return proc


def synthetic_landmarks(n=478, open_eyes=True):
    """Build a plausible-shaped (N, 3) landmark array. Real coordinate
    relationships for the eye/pose/mouth index groups are set explicitly;
    everything else is filler."""
    pts = np.random.RandomState(0).uniform(0, 100, size=(n, 3))

    def set_eye(idx_list, open_):
        outer, top1, top2, inner, bot2, bot1 = idx_list
        pts[outer] = [0, 50, 0]
        pts[inner] = [20, 50, 0]
        gap = 6 if open_ else 0.5
        pts[top1] = [7, 50 - gap, 0]
        pts[top2] = [13, 50 - gap, 0]
        pts[bot1] = [7, 50 + gap, 0]
        pts[bot2] = [13, 50 + gap, 0]

    set_eye(LEFT_EYE_EAR_IDX, open_eyes)
    set_eye(RIGHT_EYE_EAR_IDX, open_eyes)

    # eye bounding boxes for gaze + iris centered
    pts[LEFT_EYE_BOX_IDX["left"]] = [0, 50, 0]
    pts[LEFT_EYE_BOX_IDX["right"]] = [20, 50, 0]
    pts[LEFT_EYE_BOX_IDX["top"]] = [10, 44, 0]
    pts[LEFT_EYE_BOX_IDX["bottom"]] = [10, 56, 0]
    pts[LEFT_IRIS_CENTER_IDX] = [10, 50, 0]  # centered -> gaze ~0.5

    pts[RIGHT_EYE_BOX_IDX["left"]] = [0, 50, 0]
    pts[RIGHT_EYE_BOX_IDX["right"]] = [20, 50, 0]
    pts[RIGHT_EYE_BOX_IDX["top"]] = [10, 44, 0]
    pts[RIGHT_EYE_BOX_IDX["bottom"]] = [10, 56, 0]
    pts[RIGHT_IRIS_CENTER_IDX] = [10, 50, 0]

    return pts


class TestEyeFeatures:
    def test_open_eyes_have_higher_ear_than_closed(self):
        proc_open = make_bare_processor()
        proc_closed = make_bare_processor()
        ear_open, closed_open = proc_open.extract_eye_features(synthetic_landmarks(open_eyes=True))
        ear_closed, closed_closed = proc_closed.extract_eye_features(synthetic_landmarks(open_eyes=False))

        assert ear_open is not None and ear_closed is not None
        assert ear_open > ear_closed
        assert closed_open is False
        assert closed_closed is True

    def test_returns_none_on_degenerate_landmarks(self):
        proc = make_bare_processor()
        pts = np.zeros((478, 3))  # all points coincide -> horizontal distance 0
        ear, closed = proc.extract_eye_features(pts)
        assert ear is None and closed is None


class TestGaze:
    def test_centered_iris_gives_midpoint_gaze(self):
        proc = make_bare_processor(refine_landmarks=True)
        gh, gv = proc.extract_gaze(synthetic_landmarks())
        assert gh is not None and gv is not None
        assert 0.4 < gh < 0.6
        assert 0.4 < gv < 0.6

    def test_disabled_without_refine_landmarks(self):
        proc = make_bare_processor(refine_landmarks=False)
        gh, gv = proc.extract_gaze(synthetic_landmarks())
        assert gh is None and gv is None


class TestFacialMovement:
    def test_no_movement_between_identical_frames(self):
        proc = make_bare_processor()
        pts = synthetic_landmarks()
        proc._prev_landmarks = pts.copy()
        overall, mouth, eyebrow = proc.extract_facial_movement(pts)
        assert overall == pytest.approx(0.0, abs=1e-9)
        assert mouth == pytest.approx(0.0, abs=1e-9)
        assert eyebrow == pytest.approx(0.0, abs=1e-9)

    def test_none_on_first_frame(self):
        proc = make_bare_processor()
        pts = synthetic_landmarks()
        overall, mouth, eyebrow = proc.extract_facial_movement(pts)
        assert overall is None and mouth is None and eyebrow is None

    def test_movement_detected_when_mouth_shifts(self):
        proc = make_bare_processor()
        pts_prev = synthetic_landmarks()
        proc._prev_landmarks = pts_prev.copy()
        pts_now = pts_prev.copy()
        for idx in MOUTH_IDX:
            pts_now[idx][0] += 5.0  # shift mouth landmarks
        overall, mouth, eyebrow = proc.extract_facial_movement(pts_now)
        assert mouth > 0
        assert eyebrow == pytest.approx(0.0, abs=1e-6)


class TestHeadPose:
    def test_returns_three_angles_or_none_gracefully(self):
        proc = make_bare_processor()
        pts = synthetic_landmarks()
        # Head pose indices weren't explicitly set to a realistic face shape
        # in synthetic_landmarks(), so we just assert it doesn't raise and
        # returns a well-formed (yaw, pitch, roll) tuple.
        yaw, pitch, roll = proc.extract_head_pose(pts, frame_w=640, frame_h=480)
        assert yaw is None or isinstance(yaw, float)
        assert pitch is None or isinstance(pitch, float)
        assert roll is None or isinstance(roll, float)
