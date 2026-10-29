"""
tests/test_auth.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive tests for MindBloom authentication and resource-level authorization:
- Password hashing & verification
- Token creation, expiration, malformed signatures, and blacklist/revocation
- Registration & duplicate detection
- Login & invalid credentials
- Logout & subsequent token rejection
- Unauthenticated access rejection (401)
- Cross-user data isolation (IDOR prevention)
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db, User, ChildProfile
import app.auth as auth_module
from app.auth import (
    hash_password, verify_password, create_access_token, decode_access_token,
    revoke_token,
)
from app.main import app

from sqlalchemy.pool import StaticPool

# Test SQLite in-memory database with StaticPool
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


# ============================================================================
# UNIT TESTS: PASSWORD HASHING
# ============================================================================

class TestPasswordHashing:
    def test_hash_creates_valid_pbkdf2_format(self):
        h = hash_password("secret123")
        assert h.startswith("pbkdf2_sha256$")
        parts = h.split("$")
        assert len(parts) == 4
        assert int(parts[1]) > 0

    def test_verify_correct_password_returns_true(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_wrong_password_returns_false(self):
        h = hash_password("mypassword")
        assert verify_password("wrongpassword", h) is False

    def test_verify_empty_or_malformed_hash_returns_false(self):
        assert verify_password("mypassword", "") is False
        assert verify_password("mypassword", "malformed$hash") is False

    def test_password_length_validation(self):
        with pytest.raises(ValueError, match="at least 6 characters"):
            hash_password("123")


# ============================================================================
# UNIT TESTS: TOKEN MANAGEMENT & REVOCATION
# ============================================================================

class TestTokenManagement:
    def test_create_and_decode_token(self):
        token = create_access_token(user_id=42, email="user@example.com", expires_in_seconds=3600)
        payload = decode_access_token(token)
        assert payload["sub"] == "42"
        assert payload["email"] == "user@example.com"
        assert payload["exp"] > payload["iat"]

    def test_expired_token_rejected(self):
        # Create token that expired 10 seconds ago
        token = create_access_token(user_id=1, email="test@test.com", expires_in_seconds=-10)
        with pytest.raises(Exception) as excinfo:
            decode_access_token(token)
        assert "expired" in str(excinfo.value.detail).lower()

    def test_tampered_token_signature_rejected(self):
        token = create_access_token(user_id=1, email="test@test.com")
        hdr, payload, sig = token.split(".")
        tampered_token = f"{hdr}.{payload}.tamperedSig123"
        with pytest.raises(Exception) as excinfo:
            decode_access_token(tampered_token)
        assert "signature" in str(excinfo.value.detail).lower()

    def test_revoke_token_puts_in_blacklist(self):
        token = create_access_token(user_id=5, email="revoked@test.com")
        # Valid initially
        assert decode_access_token(token)["sub"] == "5"
        # Revoke
        revoke_token(token)
        # Should be rejected
        with pytest.raises(Exception) as excinfo:
            decode_access_token(token)
        assert "revoked" in str(excinfo.value.detail).lower()

    def test_database_revocation_survives_memory_blacklist_clear(self):
        token = create_access_token(user_id=6, email="persisted-revocation@test.com")
        db = TestingSessionLocal()
        try:
            revoke_token(token, db=db)
            auth_module._TOKEN_BLACKLIST.clear()
            with pytest.raises(Exception) as excinfo:
                decode_access_token(token, db=db)
            assert "revoked" in str(excinfo.value.detail).lower()
        finally:
            db.close()


# ============================================================================
# INTEGRATION TESTS: REGISTRATION, LOGIN, LOGOUT
# ============================================================================

class TestAuthEndpoints:
    def test_register_success(self, client):
        resp = client.post("/auth/register", json={"email": "caregiver@example.com", "password": "securepassword123"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "caregiver@example.com"
        assert "user_id" in data

    def test_register_duplicate_email_rejected(self, client):
        client.post("/auth/register", json={"email": "dup@example.com", "password": "password123"})
        resp = client.post("/auth/register", json={"email": "dup@example.com", "password": "anotherpassword"})
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_login_success(self, client):
        client.post("/auth/register", json={"email": "login@example.com", "password": "mypassword123"})
        resp = client.post("/auth/login", json={"email": "login@example.com", "password": "mypassword123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password_rejected(self, client):
        client.post("/auth/register", json={"email": "login2@example.com", "password": "correctpassword"})
        resp = client.post("/auth/login", json={"email": "login2@example.com", "password": "wrongpassword"})
        assert resp.status_code == 401
        assert "Incorrect" in resp.json()["detail"]

    def test_logout_invalidates_token(self, client):
        client.post("/auth/register", json={"email": "logout@example.com", "password": "password123"})
        login_res = client.post("/auth/login", json={"email": "logout@example.com", "password": "password123"}).json()
        token = login_res["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Verify access to /auth/me before logout
        me_res = client.get("/auth/me", headers=headers)
        assert me_res.status_code == 200

        # Logout
        logout_res = client.post("/auth/logout", headers=headers)
        assert logout_res.status_code == 200

        # Subsequent access must be rejected (401)
        me_res2 = client.get("/auth/me", headers=headers)
        assert me_res2.status_code == 401


# ============================================================================
# CROSS-USER RESOURCE ISOLATION TESTS (IDOR PREVENTION)
# ============================================================================

class TestResourceOwnershipIsolation:
    def _create_user_and_token(self, client, email, password="password123"):
        client.post("/auth/register", json={"email": email, "password": password})
        res = client.post("/auth/login", json={"email": email, "password": password}).json()
        return res["access_token"]

    def test_unauthenticated_requests_rejected(self, client):
        # Sensitive endpoints must return 401 when no token is provided
        assert client.get("/questions").status_code == 401
        assert client.post("/profiles/", json={"name": "Alex", "age_months": 36, "sex": "m"}).status_code == 401
        assert client.get("/profiles/").status_code == 401
        assert client.get("/profiles/1").status_code == 401
        assert client.post("/predict", json={"child_id": 1, "features": {}}).status_code == 401
        assert client.post("/video/analyze", json={"child_id": 1}).status_code == 401

    def test_authenticated_user_can_access_questionnaire(self, client):
        token = self._create_user_and_token(client, "questionnaire@test.com", "password123")
        response = client.get(
            "/questions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["total"] > 0

    def test_user_can_access_own_child_profile(self, client):
        token_a = self._create_user_and_token(client, "alice@test.com")
        headers_a = {"Authorization": f"Bearer {token_a}"}

        # User A creates child profile
        res = client.post("/profiles/", headers=headers_a, json={
            "name": "Tommy", "age_months": 30, "sex": "m", "family_history_asd": False, "jaundice_history": False
        })
        assert res.status_code == 201
        child_id = res.json()["child_id"]

        # User A can get child profile
        get_res = client.get(f"/profiles/{child_id}", headers=headers_a)
        assert get_res.status_code == 200
        assert get_res.json()["name"] == "Tommy"

    def test_user_cannot_access_other_users_child_profile(self, client):
        token_a = self._create_user_and_token(client, "alice2@test.com")
        token_b = self._create_user_and_token(client, "bob2@test.com")

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # User A creates child profile
        res_a = client.post("/profiles/", headers=headers_a, json={
            "name": "AliceChild", "age_months": 24, "sex": "f"
        })
        child_a_id = res_a.json()["child_id"]

        # User B attempts to access User A's child profile via IDOR
        res_b = client.get(f"/profiles/{child_a_id}", headers=headers_b)
        assert res_b.status_code == 403
        assert "Not authorized" in res_b.json()["detail"]

        # User B attempts to delete User A's child profile
        del_b = client.delete(f"/profiles/{child_a_id}", headers=headers_b)
        assert del_b.status_code == 403

        # User B attempts to run video analysis on User A's child
        vid_b = client.post("/video/analyze", headers=headers_b, json={"child_id": child_a_id})
        assert vid_b.status_code == 403

        # User B attempts to post predictions for User A's child
        pred_b = client.post("/predict", headers=headers_b, json={"child_id": child_a_id, "features": {}})
        assert pred_b.status_code == 403
