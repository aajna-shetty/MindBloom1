"""
tests/test_api_screening_fusion.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for MindBloom Screening, ML Pipelines, Video Analysis,
Participant Verification, and Hybrid Fusion:
- Questionnaire-only prediction
- Hybrid prediction combining questionnaire + video score via hybrid_fusion
- Verification enforcement in POST /video/analyze
- Biometric enrollment with and without caregiver consent
- Biometric erasure / deletion
- Cognitive game session logging
- Screening transaction persistence and history access
"""

from unittest.mock import patch, MagicMock
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db, User, ChildProfile, ScreeningTransaction
from app.auth import create_access_token
from app.video.verification import serialize_embedding
from app.video.service import VideoAnalysisResult
from app.video.schemas import VideoAnalyzeResponse
from app.main import app, app_state

from sqlalchemy.pool import StaticPool

# SQLite in-memory test database with StaticPool
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    # Register and login user
    client.post("/auth/register", json={"email": "screening_parent@test.com", "password": "password123"})
    login_res = client.post("/auth/login", json={"email": "screening_parent@test.com", "password": "password123"}).json()
    token = login_res["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_child(client, auth_headers):
    res = client.post("/profiles/", headers=auth_headers, json={
        "name": "Leo", "age_months": 36, "sex": "m", "family_history_asd": True, "jaundice_history": False
    })
    return res.json()["child_id"]


# ============================================================================
# BIOMETRIC ENROLLMENT & CONSENT TESTS
# ============================================================================

class TestBiometricEnrollment:
    def test_enrollment_without_reference_image_rejected(self, client, auth_headers, test_child):
        resp = client.post(f"/profiles/{test_child}/enroll-face", headers=auth_headers, json={
            "caregiver_consent": True,
            "caregiver_name": "Jane Doe"
        })
        assert resp.status_code == 400
        assert "real reference image" in resp.json()["detail"].lower()

    @patch("app.main.get_face_embedder")
    @patch("app.main.validate_and_process_enrollment_image")
    def test_enrollment_without_consent_rejected(self, mock_validate, mock_embedder, client, auth_headers, test_child):
        resp = client.post(f"/profiles/{test_child}/enroll-face", headers=auth_headers, json={
            "caregiver_consent": False,
            "caregiver_name": "Jane Doe"
        })
        assert resp.status_code == 400
        assert "consent is required" in resp.json()["detail"].lower()

    @patch("app.main.get_face_embedder")
    @patch("app.main.validate_and_process_enrollment_image", return_value=(True, np.ones((128, 128, 3), dtype=np.uint8), None))
    def test_enrollment_with_consent_succeeds(self, mock_validate, mock_embedder, client, auth_headers, test_child):
        embedder = MagicMock()
        embedder.model_name = "test-face-model"
        embedder.extract_embedding.return_value = np.ones(512, dtype=np.float32)
        mock_embedder.return_value = embedder
        resp = client.post(f"/profiles/{test_child}/enroll-face", headers=auth_headers, json={
            "caregiver_consent": True,
            "caregiver_name": "Jane Doe",
            "image_base64": "aGVsbG8="
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["consent_obtained"] is True
        assert data["child_id"] == test_child
        assert data["embedding_dimension"] > 0

        # Check status endpoint
        status_res = client.get(f"/profiles/{test_child}/biometrics", headers=auth_headers).json()
        assert status_res["is_enrolled"] is True
        assert status_res["consent_obtained"] is True
        assert status_res["consent_caregiver_name"] == "Jane Doe"

    @patch("app.main.get_face_embedder")
    @patch("app.main.validate_and_process_enrollment_image", return_value=(True, np.ones((128, 128, 3), dtype=np.uint8), None))
    def test_biometric_deletion_erases_template(self, mock_validate, mock_embedder, client, auth_headers, test_child):
        embedder = MagicMock()
        embedder.model_name = "test-face-model"
        embedder.extract_embedding.return_value = np.ones(512, dtype=np.float32)
        mock_embedder.return_value = embedder
        # Enroll first
        client.post(f"/profiles/{test_child}/enroll-face", headers=auth_headers, json={
            "caregiver_consent": True, "caregiver_name": "Jane Doe", "image_base64": "aGVsbG8="
        })
        # Delete
        del_res = client.delete(f"/profiles/{test_child}/biometrics", headers=auth_headers)
        assert del_res.status_code == 200

        # Verify erased
        status_res = client.get(f"/profiles/{test_child}/biometrics", headers=auth_headers).json()
        assert status_res["is_enrolled"] is False


# ============================================================================
# QUESTIONNAIRE ML & HYBRID PREDICTION TESTS
# ============================================================================

class TestScreeningPrediction:
    def _sample_features(self):
        return {
            "A1_Score": 1, "A2_Score": 1, "A3_Score": 0, "A4_Score": 1, "A5_Score": 1,
            "A6_Score": 0, "A7_Score": 1, "A8_Score": 1, "A9_Score": 0, "A10_Score": 1,
            "age": 3.0, "gender": 1, "jaundice": 0, "family_history_asd": 1,
            "social_communication_delay": 5.0, "repetitive_behaviors_score": 4.0
        }

    def test_questionnaire_only_prediction(self, client, auth_headers, test_child):
        # When video_score is None, returns questionnaire-only result
        req = {
            "child_id": test_child,
            "features": self._sample_features(),
            "video_score": None
        }
        resp = client.post("/predict", headers=auth_headers, json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_score"] is None
        assert data["final_score"] == pytest.approx(data["questionnaire_score"], abs=1e-3)
        assert data["models_evaluated"] == "questionnaire_only"
        assert "disclaimer" in data

    def test_hybrid_prediction_combines_scores_via_dynamic_fusion(self, client, auth_headers, test_child):
        # When video_score is provided, calls hybrid_fusion
        req = {
            "child_id": test_child,
            "features": self._sample_features(),
            "video_score": 0.40,
            "video_confidence": 0.85
        }
        resp = client.post("/predict", headers=auth_headers, json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_score"] == 0.40
        assert data["video_confidence"] == 0.85
        assert data["models_evaluated"] == "questionnaire+video_hybrid"
        # Final score should lie between questionnaire_score and video_score
        q_score = data["questionnaire_score"]
        final_score = data["final_score"]
        assert min(q_score, 0.40) <= final_score <= max(q_score, 0.40)

    def test_screening_transaction_persists_in_history(self, client, auth_headers, test_child):
        req = {
            "child_id": test_child,
            "features": self._sample_features(),
            "video_score": None
        }
        client.post("/predict", headers=auth_headers, json=req)

        # Check screening history endpoint
        hist_res = client.get(f"/profiles/{test_child}/screenings", headers=auth_headers)
        assert hist_res.status_code == 200
        records = hist_res.json()
        assert len(records) >= 1
        assert records[0]["models_evaluated"] == "questionnaire_only"


# ============================================================================
# VIDEO ANALYSIS WITH PARTICIPANT VERIFICATION TESTS
# ============================================================================

class TestVideoAnalyzeEndpoint:
    @patch("app.main.run_video_session")
    def test_video_analyze_with_enforced_verification_missing_template(self, mock_run, client, auth_headers, test_child):
        # Child has no biometric enrollment yet
        resp = client.post("/video/analyze", headers=auth_headers, json={
            "child_id": test_child,
            "source": "0",
            "enforce_verification": True
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "missing_reference_template"
        assert data["verification_status"] == "not_enrolled"
        assert "recovery_guidance" in data
        mock_run.assert_not_called()

    @patch("app.main.run_video_session")
    @patch("app.main.get_face_embedder")
    @patch("app.main.validate_and_process_enrollment_image", return_value=(True, np.ones((128, 128, 3), dtype=np.uint8), None))
    def test_video_analyze_runs_when_template_enrolled(self, mock_validate, mock_embedder, mock_run, client, auth_headers, test_child):
        # Enroll child template
        embedder = MagicMock()
        embedder.model_name = "test-face-model"
        embedder.extract_embedding.return_value = np.ones(512, dtype=np.float32)
        mock_embedder.return_value = embedder
        client.post(f"/profiles/{test_child}/enroll-face", headers=auth_headers, json={
            "caregiver_consent": True, "caregiver_name": "Jane Doe", "image_base64": "aGVsbG8="
        })

        mock_run.return_value = VideoAnalysisResult(
            status="video_model_not_available",
            session_duration_seconds=15.0,
            valid_frame_ratio=0.92,
            feature_count=47,
            verification_status="verified",
            unverified_frames_excluded=2,
            detail="Video feature extraction succeeded, video_model.pkl is unavailable as documented."
        )

        challenge = client.post("/video/challenge", headers=auth_headers).json()["challenge"]

        resp = client.post("/video/analyze", headers=auth_headers, json={
            "child_id": test_child,
            "source": "0",
            "enforce_verification": True,
            "challenge_token": challenge,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "video_model_not_available"
        assert data["verification_status"] == "verified"
        assert data["unverified_frames_excluded"] == 2
        mock_run.assert_called_once()

    def test_uploaded_video_requires_authentication(self, client):
        resp = client.post(
            "/video/analyze/upload",
            files={"file": ("sample.mp4", b"video-data", "video/mp4")},
            data={"child_id": "1"},
        )
        assert resp.status_code == 401

    def test_uploaded_video_rejects_unsupported_extension(self, client, auth_headers, test_child):
        resp = client.post(
            "/video/analyze/upload",
            headers=auth_headers,
            files={"file": ("sample.txt", b"not-video", "text/plain")},
            data={"child_id": str(test_child)},
        )
        assert resp.status_code == 400
        assert "unsupported video type" in resp.json()["detail"].lower()

    @patch("app.main.analyze_video")
    def test_uploaded_video_is_removed_after_analysis(
        self, mock_analyze, client, auth_headers, test_child
    ):
        captured_path = {}

        def analyze(request, current_user, db):
            captured_path["value"] = request.source
            assert os.path.exists(request.source)
            return VideoAnalyzeResponse(status="success")

        mock_analyze.side_effect = analyze
        resp = client.post(
            "/video/analyze/upload",
            headers=auth_headers,
            files={"file": ("sample.mp4", b"video-data", "video/mp4")},
            data={"child_id": str(test_child), "enforce_verification": "false"},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert captured_path["value"]
        assert not os.path.exists(captured_path["value"])
