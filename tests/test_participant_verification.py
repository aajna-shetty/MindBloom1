"""
tests/test_participant_verification.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive tests for Participant Identity Verification:
- Template serialization, deserialization, and cosine similarity
- Correct participant verification against enrolled template
- Sibling / different participant mismatch detection
- No face and multiple faces handling
- Uncertain identity handling
- Pre-assessment multi-frame consensus
- Continuous session monitoring: pausing on participant change,
  resuming on re-verification, and excluding unverified frames
- Missing reference template graceful handling
"""

import numpy as np
import pytest

from app.video.verification import (
    ParticipantVerifier,
    VerificationState,
    cosine_similarity,
    serialize_embedding,
    deserialize_embedding,
    SIMILARITY_VERIFIED_THRESHOLD,
    SIMILARITY_UNCERTAIN_THRESHOLD,
    FaceEmbedder,
    challenge_completed,
    issue_liveness_challenge,
    read_liveness_challenge,
)


def _make_unit_vector(dim=512, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_random_challenge_is_signed_and_unpredictable():
    a, b = issue_liveness_challenge(), issue_liveness_challenge()
    assert a["action"] in {"blink", "turn_left", "turn_right", "nod"}
    assert read_liveness_challenge(a["challenge"]) == a["action"]
    assert a["challenge"] != b["challenge"]
    assert read_liveness_challenge(a["challenge"] + "x") is None


def test_liveness_challenge_requires_consecutive_single_face_motion():
    from types import SimpleNamespace
    stable = [SimpleNamespace(face_detected=True, num_faces=1, yaw=0, pitch=0, eye_closed=False) for _ in range(8)]
    assert not challenge_completed("turn_left", stable)  # still photo / static replay
    assert not challenge_completed("turn_left", [SimpleNamespace(face_detected=True, num_faces=2, yaw=-20, pitch=0, eye_closed=False) for _ in range(8)])
    moved = [SimpleNamespace(face_detected=True, num_faces=1, yaw=y, pitch=0, eye_closed=False) for y in (0, 0, -15, -18, -17, -16, -2, 0)]
    assert challenge_completed("turn_left", moved)


class TestVectorMathAndSerialization:
    def test_cosine_similarity_identical_vectors_is_one(self):
        v = _make_unit_vector(512, seed=1)
        sim = cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal_vectors_is_zero(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-5)

    def test_serialize_and_deserialize_preserves_vector(self):
        v = _make_unit_vector(512, seed=99)
        serialized = serialize_embedding(v)
        assert isinstance(serialized, str)
        restored = deserialize_embedding(serialized)
        assert restored.shape == (512,)
        assert np.allclose(v, restored, atol=1e-5)


class TestParticipantVerifierSingleFrame:
    def test_correct_participant_verified(self):
        ref_template = _make_unit_vector(512, seed=10)
        verifier = ParticipantVerifier(reference_template=ref_template)

        # Candidate is nearly identical to reference (slight camera noise)
        noisy_candidate = ref_template + np.random.RandomState(2).randn(512).astype(np.float32) * 0.05
        noisy_candidate /= np.linalg.norm(noisy_candidate)

        # Mock embedder output
        verifier.embedder.extract_embedding = lambda crop: noisy_candidate

        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        state, sim = verifier.verify_single_face(dummy_crop)

        assert state == VerificationState.VERIFIED
        assert sim >= SIMILARITY_VERIFIED_THRESHOLD

    def test_different_participant_detected_as_mismatch(self):
        ref_template = _make_unit_vector(512, seed=10)
        verifier = ParticipantVerifier(reference_template=ref_template)

        # Candidate is completely different person (orthogonal random vector)
        different_candidate = _make_unit_vector(512, seed=888)
        verifier.embedder.extract_embedding = lambda crop: different_candidate

        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        state, sim = verifier.verify_single_face(dummy_crop)

        assert state == VerificationState.MISMATCH
        assert sim < SIMILARITY_UNCERTAIN_THRESHOLD

    def test_uncertain_identity_range(self):
        ref_template = _make_unit_vector(512, seed=10)
        verifier = ParticipantVerifier(reference_template=ref_template)

        # Create vector with similarity exactly in uncertain range (~0.55)
        # v = a * ref + b * ortho
        ortho = _make_unit_vector(512, seed=999)
        ortho -= np.dot(ortho, ref_template) * ref_template
        ortho /= np.linalg.norm(ortho)

        target_sim = 0.55
        uncertain_vec = target_sim * ref_template + np.sqrt(1.0 - target_sim**2) * ortho
        verifier.embedder.extract_embedding = lambda crop: uncertain_vec

        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        state, sim = verifier.verify_single_face(dummy_crop)

        assert state == VerificationState.UNCERTAIN
        assert SIMILARITY_UNCERTAIN_THRESHOLD <= sim < SIMILARITY_VERIFIED_THRESHOLD

    def test_no_face_crop_returns_no_face(self):
        ref_template = _make_unit_vector(512, seed=10)
        verifier = ParticipantVerifier(reference_template=ref_template)

        state, sim = verifier.verify_single_face(None)
        assert state == VerificationState.NO_FACE
        assert sim == 0.0

    def test_missing_reference_template_returns_uncertain(self):
        verifier = ParticipantVerifier(reference_template=None)
        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        state, sim = verifier.verify_single_face(dummy_crop)
        assert state == VerificationState.UNCERTAIN


class TestPreAssessmentConsensus:
    def test_consensus_passes_when_majority_matches(self):
        ref = _make_unit_vector(512, seed=5)
        verifier = ParticipantVerifier(reference_template=ref)

        matching_emb = ref.copy()
        verifier.embedder.extract_embedding = lambda crop: matching_emb

        frames = [np.full((64, 64, 3), i * 10, dtype=np.uint8) for i in range(5)]
        res = verifier.verify_initial_consensus(frames, min_verified_ratio=0.6)

        assert res["verified"] is True
        assert res["status"] == "verified"
        assert res["verified_ratio"] == 1.0

    def test_consensus_fails_when_wrong_participant(self):
        ref = _make_unit_vector(512, seed=5)
        verifier = ParticipantVerifier(reference_template=ref)

        mismatch_emb = _make_unit_vector(512, seed=100)
        verifier.embedder.extract_embedding = lambda crop: mismatch_emb

        frames = [np.full((64, 64, 3), i * 10, dtype=np.uint8) for i in range(5)]
        res = verifier.verify_initial_consensus(frames, min_verified_ratio=0.6)

        assert res["verified"] is False
        assert res["status"] == "participant_mismatch"
        assert "does not match" in res["detail"]

    def test_consensus_fails_when_no_template_enrolled(self):
        verifier = ParticipantVerifier(reference_template=None)
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
        res = verifier.verify_initial_consensus(frames)

        assert res["verified"] is False
        assert res["status"] == "missing_reference_template"


class TestContinuousSessionMonitoring:
    def test_session_pauses_when_child_leaves_or_another_appears(self):
        ref = _make_unit_vector(512, seed=7)
        verifier = ParticipantVerifier(reference_template=ref)

        dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)
        verifier.embedder.extract_embedding = lambda c: ref.copy()

        # Frame 1: Child verified
        allow, state = verifier.evaluate_session_frame(face_detected=True, num_faces=1, face_crop=dummy_crop)
        assert allow is True
        assert state == VerificationState.VERIFIED

        # Frame 2: Child leaves frame (no face) -> Pause collection
        allow, state = verifier.evaluate_session_frame(face_detected=False, num_faces=0, face_crop=None)
        assert allow is False
        assert state == VerificationState.PAUSED
        assert verifier.stats.unverified_frames_excluded >= 1

        # Frame 3: Another person leans in (2 faces) -> Pause collection
        allow, state = verifier.evaluate_session_frame(face_detected=True, num_faces=2, face_crop=dummy_crop)
        assert allow is False
        assert state == VerificationState.PAUSED
        assert verifier.stats.multi_face_events >= 1

        # Frame 4: Child returns alone and is verified -> Resume collection
        allow, state = verifier.evaluate_session_frame(face_detected=True, num_faces=1, face_crop=dummy_crop)
        assert allow is True
        assert state == VerificationState.VERIFIED

    def test_session_pauses_when_different_child_swaps_in(self):
        ref = _make_unit_vector(512, seed=7)
        verifier = ParticipantVerifier(reference_template=ref)

        dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)

        # Initial frames: Enrolled child
        verifier.embedder.extract_embedding = lambda c: ref.copy()
        allow, state = verifier.evaluate_session_frame(face_detected=True, num_faces=1, face_crop=dummy_crop)
        assert allow is True

        # Sibling swaps in: Different face embedding
        sibling_emb = _make_unit_vector(512, seed=777)
        verifier.embedder.extract_embedding = lambda c: sibling_emb

        allow, state = verifier.evaluate_session_frame(face_detected=True, num_faces=1, face_crop=dummy_crop)
        assert allow is False
        assert state == VerificationState.MISMATCH
        assert verifier.stats.mismatch_events >= 1
        assert verifier.stats.unverified_frames_excluded >= 1
