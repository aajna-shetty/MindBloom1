from app.video.session import SessionData
from app.video.mediapipe_processor import LandmarkFrameFeatures
from app.video.deepface_processor import DeepFaceFrameResult


def test_blink_counting_filters_single_frame_jitter():
    s = SessionData()
    # Frame pattern: single-frame "closed" blip should NOT count as a blink
    # (BLINK_MIN_CONSEC_FRAMES default is 2), but a real 3-frame closure
    # should.
    pattern = [False, False, True, False, False, True, True, True, False, False]
    for t, closed in enumerate(pattern):
        s.timestamps.append(t / 30.0)
        s.eye_closed_frames.append(closed)

    stats = s.compute_blink_stats()
    assert stats["blink_count"] == 1.0  # only the 3-frame run counts


def test_no_face_frames_reduce_valid_frame_ratio():
    s = SessionData()
    s.record_landmark_result(0.0, LandmarkFrameFeatures(face_detected=False))
    s.record_landmark_result(0.1, LandmarkFrameFeatures(face_detected=True, num_faces=1))
    assert s.total_frames == 2
    assert s.valid_face_frames == 1
    assert s.valid_frame_ratio() == 0.5


def test_capture_failure_counts_toward_total_but_not_valid():
    s = SessionData()
    s.record_capture_failure()
    s.record_capture_failure()
    assert s.total_frames == 2
    assert s.invalid_frames == 2
    assert s.valid_frame_ratio() == 0.0


def test_has_sufficient_data_requires_both_ratio_and_absolute_count():
    s = SessionData()
    # High ratio but too few absolute frames -> insufficient
    for i in range(10):
        s.record_landmark_result(i * 0.03, LandmarkFrameFeatures(face_detected=True, num_faces=1))
    assert s.valid_frame_ratio() == 1.0
    assert s.has_sufficient_data() is False  # below MIN_VALID_FRAME_COUNT


def test_deepface_success_and_failure_tracked_independently():
    s = SessionData()
    s.record_deepface_result(DeepFaceFrameResult(success=True, dominant_emotion="happy",
                                                   emotion_scores={"happy": 1.0}))
    s.record_deepface_result(DeepFaceFrameResult(success=False, error="no_face"))
    q = s.quality_metrics()
    assert q["deepface_attempts"] == 2
    assert q["deepface_successes"] == 1
    assert q["deepface_failures"] == 1
    assert q["deepface_success_ratio"] == 0.5
