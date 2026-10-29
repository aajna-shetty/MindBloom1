import base64
import json
import os
import sys
import logging
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core Machine Learning Pipelines
from app.model import load_best_model, load_metrics, load_feature_importance
from app.preprocess import load_scaler, get_feature_list
from app.utils import risk_level, top_features
from app.questionnaire import build_feature_vector_from_answers, QUESTIONS

# Relational Storage Engine & Models
from app.database import (
    engine, Base, get_db,
    User, ChildProfile, ParticipantBiometrics, ScreeningTransaction, CognitiveGameLog,
)

# Authentication & Resource Authorization
from app.auth import (
    hash_password, verify_password, create_access_token, revoke_token,
    get_current_user, verify_child_ownership,
)

# Video Processing & Identity Verification
from app.video.service import run_video_session
from app.video.schemas import (
    VideoAnalyzeRequest, VideoAnalyzeResponse,
    BiometricEnrollRequest, BiometricEnrollResponse, BiometricStatusResponse,
    ScreeningReportResponse,
)
from app.video.fusion import hybrid_fusion
from app.video.config import ALLOW_SERVER_FILE_PATHS, ALLOWED_SOURCE_DIR
from app.video.verification import (
    get_face_embedder, serialize_embedding, deserialize_embedding,
    validate_and_process_enrollment_image,
    issue_liveness_challenge, read_liveness_challenge,
)

# Initialize database schema for local development. Production deployments
# must apply the versioned Alembic migrations before starting the API.
if os.environ.get("MINDBLOOM_ENV", "development").lower() != "production":
    try:
        Base.metadata.create_all(bind=engine)
        print("\n[Database] Success: Database schema synchronized.\n")
    except Exception as db_err:
        print(f"\n[Database] Critical connection error: {str(db_err)}\n")
else:
    logger = logging.getLogger(__name__)
    logger.info("Production mode: expecting database schema from Alembic migrations.")

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artefacts ...")
    try:
        app_state["model"] = load_best_model()
        app_state["scaler"] = load_scaler()
        app_state["feature_list"] = get_feature_list()
        app_state["metrics"] = load_metrics()
        app_state["feature_import"] = load_feature_importance()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed loading artifacts: {str(e)}")
        app_state["model"] = None
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="MindBloom Autism Risk Screening API",
    description="Multimodal autism screening with authentication, participant verification, and hybrid fusion",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("MINDBLOOM_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ] or ["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================
# PYDANTIC REQ / RESP COMMUNICATORS
# ========================================================

class RegisterInput(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Valid email address")
    password: str = Field(..., min_length=6, description="Minimum 6 characters")

class LoginInput(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Valid email address")
    password: str

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str

class ProfileCreateInput(BaseModel):
    name: str
    age_months: int
    sex: str
    family_history_asd: bool = False
    jaundice_history: bool = False

class ProfileResponse(BaseModel):
    id: int
    name: str
    age_months: int
    sex: str
    family_history_asd: bool
    jaundice_history: bool
    created_at: str

class PredictRequest(BaseModel):
    child_id: int
    features: Dict[str, Any]
    video_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    video_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mediapipe_summary: Optional[Dict[str, Any]] = None
    vit_summary: Optional[Dict[str, Any]] = None

class FeatureItem(BaseModel):
    feature: str
    importance: float

class PredictResponse(BaseModel):
    screening_id: Optional[int] = None
    questionnaire_score: float
    video_score: Optional[float] = None
    video_confidence: Optional[float] = None
    final_score: float
    risk_level: str
    top_features: List[FeatureItem]
    explanation: str
    models_evaluated: str
    disclaimer: str = "MindBloom is an experimental screening aid and not a diagnostic tool."

class GameLogInput(BaseModel):
    game_name: str
    accuracy: float
    completion_time_sec: float
    reaction_time_ms: float

# ========================================================
# AUTHENTICATION ENDPOINTS (NO RBAC - OWNERSHIP ONLY)
# ========================================================

@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(user_data: RegisterInput, db: Session = Depends(get_db)):
    """Register a new user account."""
    existing = db.query(User).filter(User.email == user_data.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists."
        )

    try:
        hashed = hash_password(user_data.password)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    new_user = User(
        email=user_data.email.lower(),
        hashed_password=hashed
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info("Registered user %s (ID: %s)", new_user.email, new_user.id)
    return {
        "status": "Success",
        "user_id": new_user.id,
        "email": new_user.email,
        "message": "User registered successfully."
    }

@app.post("/auth/login", response_model=AuthTokenResponse)
def login(credentials: LoginInput, db: Session = Depends(get_db)):
    """Authenticate with email and password, returning a signed bearer token."""
    user = db.query(User).filter(User.email == credentials.email.lower()).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user_id=user.id, email=user.email)
    logger.info("User logged in: %s (ID: %s)", user.email, user.id)
    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user.id,
        email=user.email
    )

@app.post("/auth/logout")
def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Invalidate current bearer token."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        revoke_token(token, db=db)
    logger.info("User logged out: ID %s", current_user.id)
    return {"status": "Success", "message": "Logged out successfully."}

@app.get("/auth/me")
def get_me(current_user: User = Depends(get_current_user)):
    """Retrieve profile of authenticated user."""
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }

# ========================================================
# HEALTH (PUBLIC) & QUESTIONNAIRE (AUTHENTICATED)
# ========================================================

@app.get("/health")
def health():
    return {
        "status": "ok" if app_state.get("model") else "model_not_loaded",
        "model_ready": app_state.get("model") is not None,
    }

@app.get("/webcam-test", response_class=FileResponse, include_in_schema=False)
def webcam_test():
    """Serve the local browser webcam verification test page."""
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "webcam_test.html"),
        media_type="text/html",
    )

@app.post("/video/challenge")
def create_video_challenge(current_user: User = Depends(get_current_user)):
    """Create an unpredictable, short-lived prompt for webcam liveness."""
    return issue_liveness_challenge()

@app.get("/questions")
def get_questions(current_user: User = Depends(get_current_user)):
    return {"total": len(QUESTIONS), "questions": QUESTIONS}

# ========================================================
# CHILD PROFILE ENDPOINTS (RESOURCE-LEVEL AUTHORIZATION)
# ========================================================

@app.post("/profiles/", status_code=status.HTTP_201_CREATED)
def create_child_profile(
    profile: ProfileCreateInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new child profile owned by the authenticated user."""
    db_profile = ChildProfile(
        user_id=current_user.id,
        **profile.model_dump()
    )
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    logger.info("Created child profile %s for user %s", db_profile.id, current_user.id)
    return {"status": "Success", "child_id": db_profile.id, "name": db_profile.name}

@app.get("/profiles/", response_model=List[ProfileResponse])
def list_child_profiles(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve all child profiles owned by the authenticated user."""
    profiles = db.query(ChildProfile).filter(ChildProfile.user_id == current_user.id).all()
    return [
        ProfileResponse(
            id=p.id,
            name=p.name,
            age_months=p.age_months,
            sex=p.sex,
            family_history_asd=p.family_history_asd,
            jaundice_history=p.jaundice_history,
            created_at=p.created_at.isoformat() if p.created_at else ""
        )
        for p in profiles
    ]

@app.get("/profiles/{child_id}")
def get_child_profile(
    child_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve details for a specific child profile owned by the user."""
    child = verify_child_ownership(child_id, current_user, db)
    return {
        "id": child.id,
        "name": child.name,
        "age_months": child.age_months,
        "sex": child.sex,
        "family_history_asd": child.family_history_asd,
        "jaundice_history": child.jaundice_history,
        "has_biometrics": child.biometrics is not None,
        "created_at": child.created_at.isoformat() if child.created_at else None
    }

@app.delete("/profiles/{child_id}")
def delete_child_profile(
    child_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a child profile and all associated data."""
    child = verify_child_ownership(child_id, current_user, db)
    db.delete(child)
    db.commit()
    logger.info("Deleted child profile %s by user %s", child_id, current_user.id)
    return {"status": "Success", "message": f"Child profile {child_id} deleted successfully."}

# ========================================================
# PARTICIPANT BIOMETRIC ENROLLMENT & PRIVACY
# ========================================================

def save_biometric_template(
    child_id: int,
    caregiver_name: str,
    embedding: np.ndarray,
    model_name: str,
    db: Session
) -> BiometricEnrollResponse:
    """Helper to persist protected 512-dim embedding and consent record."""
    serialized_template = serialize_embedding(embedding)
    existing_biometric = db.query(ParticipantBiometrics).filter(ParticipantBiometrics.child_id == child_id).first()
    if existing_biometric:
        existing_biometric.consent_obtained = True
        existing_biometric.consent_caregiver_name = caregiver_name
        existing_biometric.consent_timestamp = datetime.utcnow()
        existing_biometric.face_template = serialized_template
        existing_biometric.template_model = model_name
        existing_biometric.embedding_dimension = len(embedding)
        existing_biometric.updated_at = datetime.utcnow()
    else:
        new_bio = ParticipantBiometrics(
            child_id=child_id,
            consent_obtained=True,
            consent_caregiver_name=caregiver_name,
            consent_timestamp=datetime.utcnow(),
            face_template=serialized_template,
            template_model=model_name,
            embedding_dimension=len(embedding),
        )
        db.add(new_bio)

    db.commit()
    logger.info("Enrolled reference face template for child %s with caregiver consent (%s)", child_id, model_name)

    return BiometricEnrollResponse(
        status="Success",
        child_id=child_id,
        consent_obtained=True,
        enrolled_at=datetime.utcnow().isoformat(),
        template_model=model_name,
        embedding_dimension=len(embedding),
        detail="Biometric reference template securely generated and saved. Raw image was not retained."
    )


@app.post("/profiles/{child_id}/enroll-face/upload", response_model=BiometricEnrollResponse)
async def enroll_child_face_upload(
    child_id: int,
    file: UploadFile = File(...),
    caregiver_consent: bool = Form(...),
    caregiver_name: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Option A: Upload a photo of the child.
    Validates file format, size, dimensions, face suitability, and sharpness.
    Extracts protected normalized 512-dim embedding without retaining raw image bytes.
    """
    child = verify_child_ownership(child_id, current_user, db)

    if not caregiver_consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Caregiver consent is required before enrolling biometric facial data."
        )

    try:
        image_bytes = await file.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded image file: {read_err}"
        )

    is_valid, face_rgb, error_msg = validate_and_process_enrollment_image(
        image_bytes, filename=file.filename
    )
    if not is_valid or face_rgb is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg or "Invalid image file for enrollment."
        )

    embedder = get_face_embedder()
    embedding = embedder.extract_embedding(face_rgb)
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to generate a valid face template from the uploaded photo."
        )

    return save_biometric_template(
        child_id=child.id,
        caregiver_name=caregiver_name,
        embedding=embedding,
        model_name=embedder.model_name,
        db=db
    )


@app.post("/profiles/{child_id}/enroll-face/capture", response_model=BiometricEnrollResponse)
def enroll_child_face_capture(
    child_id: int,
    enroll_req: BiometricEnrollRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Option B: Camera-captured photo of the child.
    Validates face presence, sharpness, and single participant.
    Extracts protected normalized 512-dim embedding without retaining raw image bytes.
    """
    child = verify_child_ownership(child_id, current_user, db)

    if not enroll_req.caregiver_consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Caregiver consent is required before enrolling biometric facial data."
        )

    if not enroll_req.image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera capture enrollment requires base64-encoded image data."
        )

    raw_b64 = enroll_req.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image data: {exc}"
        )

    is_valid, face_rgb, error_msg = validate_and_process_enrollment_image(
        image_bytes, filename="camera_capture.jpg"
    )
    if not is_valid or face_rgb is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg or "Invalid captured image for enrollment."
        )

    embedder = get_face_embedder()
    embedding = embedder.extract_embedding(face_rgb)
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to generate a valid face template from the captured photo."
        )

    return save_biometric_template(
        child_id=child.id,
        caregiver_name=enroll_req.caregiver_name,
        embedding=embedding,
        model_name=embedder.model_name,
        db=db
    )


@app.post("/profiles/{child_id}/enroll-face", response_model=BiometricEnrollResponse)
def enroll_child_face(
    child_id: int,
    enroll_req: BiometricEnrollRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Consent-based reference face template enrollment for a child profile.
    Generates a protected, normalized 512-dim embedding.
    Never stores raw facial pixels.
    """
    child = verify_child_ownership(child_id, current_user, db)

    # 1. Enforce explicit caregiver consent
    if not enroll_req.caregiver_consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Caregiver consent is required before enrolling biometric facial data."
        )

    # 2. Extract or decode reference face image
    face_rgb = None
    if enroll_req.image_base64:
        raw_b64 = enroll_req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 image data: {exc}"
            )
        is_valid, face_rgb, error_msg = validate_and_process_enrollment_image(
            image_bytes, filename="enrollment.jpg"
        )
        if not is_valid or face_rgb is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg or "Invalid image provided for enrollment."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A real reference image is required. Use the upload or capture enrollment endpoint.",
        )

    embedder = get_face_embedder()
    embedding = embedder.extract_embedding(face_rgb)
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to generate a valid face template from the provided image."
        )

    return save_biometric_template(
        child_id=child.id,
        caregiver_name=enroll_req.caregiver_name,
        embedding=embedding,
        model_name=embedder.model_name,
        db=db
    )

@app.get("/profiles/{child_id}/biometrics", response_model=BiometricStatusResponse)
def get_biometric_status(
    child_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Check participant biometric enrollment status."""
    child = verify_child_ownership(child_id, current_user, db)
    bio = child.biometrics
    if not bio:
        return BiometricStatusResponse(
            child_id=child_id,
            is_enrolled=False,
            consent_obtained=False
        )

    return BiometricStatusResponse(
        child_id=child_id,
        is_enrolled=True,
        consent_obtained=bio.consent_obtained,
        consent_caregiver_name=bio.consent_caregiver_name,
        enrolled_at=bio.enrolled_at.isoformat() if bio.enrolled_at else None,
        template_model=bio.template_model
    )

@app.delete("/profiles/{child_id}/biometrics")
def delete_biometric_data(
    child_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Permanently delete biometric templates for child profile (Right to Erasure)."""
    child = verify_child_ownership(child_id, current_user, db)
    if child.biometrics:
        db.delete(child.biometrics)
        db.commit()
        logger.info("Deleted biometric template for child %s upon user request", child_id)

    return {
        "status": "Success",
        "message": f"Biometric reference data for child {child_id} has been permanently deleted."
    }

# ========================================================
# QUESTIONNAIRE & HYBRID PREDICTION ENDPOINTS
# ========================================================

@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Calculate questionnaire risk score and optional hybrid fusion with video score.
    Does NOT fabricate video predictions when video is absent.
    """
    if app_state.get("model") is None:
        raise HTTPException(status_code=503, detail="Questionnaire model not loaded.")

    child = verify_child_ownership(request.child_id, current_user, db)

    model = app_state["model"]
    scaler = app_state["scaler"]
    feature_import = app_state.get("feature_import", {})

    try:
        X_raw = build_feature_vector_from_answers(request.features)
        X_scaled = scaler.transform([X_raw])
        q_score = float(model.predict_proba(X_scaled)[0][1])

        # Integrate Hybrid Fusion
        if request.video_score is not None:
            fusion_result = hybrid_fusion(
                questionnaire_score=q_score,
                video_score=request.video_score,
                video_confidence=request.video_confidence
            )
            final_score = fusion_result["hybrid_risk_score"]
            models_evaluated = "questionnaire+video_hybrid"
            explanation = (
                f"Hybrid risk score is {final_score:.2f} combining questionnaire ({q_score:.2f}, "
                f"weight {fusion_result['alpha']:.2f}) and video ({request.video_score:.2f}, "
                f"weight {1.0 - fusion_result['alpha']:.2f})."
            )
        else:
            final_score = q_score
            models_evaluated = "questionnaire_only"
            explanation = (
                f"Questionnaire-only risk score is {final_score:.2f}. "
                "No video assessment score was factored into this result."
            )

        risk = risk_level(final_score)

        top_fi = []
        if feature_import:
            for f in top_features(feature_import, n=3):
                top_fi.append(FeatureItem(**f))

        # Persist screening record associated with the child
        screening_record = ScreeningTransaction(
            child_id=child.id,
            questionnaire_score=round(q_score, 4),
            video_score=round(request.video_score, 4) if request.video_score is not None else None,
            video_confidence=round(request.video_confidence, 4) if request.video_confidence is not None else None,
            hybrid_risk_score=round(final_score, 4),
            risk_tier=risk,
            verification_status="verified" if request.video_score is not None else "not_applicable",
            models_evaluated=models_evaluated
        )
        db.add(screening_record)
        db.commit()

        logger.info("Screening saved for child %s: final_score=%.2f risk=%s", child.id, final_score, risk)

        return PredictResponse(
            questionnaire_score=round(q_score, 4),
            video_score=round(request.video_score, 4) if request.video_score is not None else None,
            video_confidence=round(request.video_confidence, 4) if request.video_confidence is not None else None,
            final_score=round(final_score, 4),
            risk_level=risk,
            top_features=top_fi,
            explanation=explanation,
            models_evaluated=models_evaluated,
            disclaimer="MindBloom is an experimental screening aid and not a diagnostic tool."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /predict endpoint")
        raise HTTPException(status_code=400, detail=str(e))

# ========================================================
# VIDEO ANALYSIS WITH PARTICIPANT VERIFICATION
# ========================================================

@app.post("/video/analyze", response_model=VideoAnalyzeResponse)
def analyze_video(
    request: VideoAnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Run video screening with participant identity verification.
    Ensures that only the enrolled participant's behavioral data is collected.
    """
    child = verify_child_ownership(request.child_id, current_user, db)

    challenge_action = read_liveness_challenge(request.challenge_token) if request.enforce_verification else None
    if request.enforce_verification and challenge_action is None:
        return VideoAnalyzeResponse(status="liveness_failed", verification_status="liveness_failed",
                                    liveness_status="missing_or_expired_challenge",
                                    detail="A valid, unexpired server-issued liveness challenge is required.")

    if not request.source.isdigit():
        source_path = os.path.abspath(request.source)
        temp_root = os.path.abspath(tempfile.gettempdir())
        is_controlled_temp_file = os.path.commonpath([source_path, temp_root]) == temp_root
        if not ALLOW_SERVER_FILE_PATHS and not is_controlled_temp_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Server-side video file paths are disabled. Use a webcam index or enable a configured local source directory.",
            )
        allowed_root = os.path.abspath(ALLOWED_SOURCE_DIR)
        if not is_controlled_temp_file and (
            not os.path.isfile(source_path)
            or os.path.commonpath([source_path, allowed_root]) != allowed_root
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video source must be an existing file under the configured source directory.",
            )

    # Check biometric reference template
    reference_template = None
    if request.enforce_verification:
        bio = child.biometrics
        if not bio or not bio.consent_obtained:
            logger.warning("Video assessment halted: Child %s has no enrolled biometric template.", child.id)
            return VideoAnalyzeResponse(
                status="missing_reference_template",
                verification_status="not_enrolled",
                detail="Participant has not enrolled a reference face template with caregiver consent.",
                recovery_guidance="Please enroll a reference photo for this child profile with consent, or proceed with questionnaire-only screening."
            )
        try:
            reference_template = deserialize_embedding(bio.face_template)
        except Exception as exc:
            logger.error("Failed deserializing template for child %s: %s", child.id, exc)
            return VideoAnalyzeResponse(
                status="error",
                detail="Corrupted biometric reference template.",
                recovery_guidance="Please re-enroll the child's reference photo."
            )

    source = request.source
    if source.isdigit():
        source = int(source)

    try:
        result = run_video_session(
            source=source,
            duration_seconds=request.duration_seconds,
            reference_template=reference_template,
            enforce_verification=request.enforce_verification,
            challenge_action=challenge_action,
        )
    except Exception as exc:
        logger.exception("Video session crashed")
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(
        "Video analysis for child %s: status=%s verification=%s score=%s",
        child.id, result.status, result.verification_status, result.video_score
    )

    # If successful video score produced, persist transaction
    if result.status == "success" and result.video_score is not None:
        screening_record = ScreeningTransaction(
            child_id=child.id,
            questionnaire_score=0.0,
            video_score=result.video_score,
            video_confidence=result.video_confidence,
            hybrid_risk_score=result.video_score,
            risk_tier=risk_level(result.video_score),
            verification_status="verified",
            models_evaluated="video_rf"
        )
        db.add(screening_record)
        db.commit()

    return VideoAnalyzeResponse(
        status=result.status,
        video_score=result.video_score,
        video_confidence=result.video_confidence,
        predicted_class=result.predicted_class,
        session_duration_seconds=result.session_duration_seconds,
        valid_frame_ratio=result.valid_frame_ratio,
        feature_count=result.feature_count,
        verification_status=result.verification_status,
        unverified_frames_excluded=result.unverified_frames_excluded,
        detail=result.detail,
        recovery_guidance=result.recovery_guidance,
        liveness_status=result.liveness_status,
    )


@app.post("/video/analyze/upload", response_model=VideoAnalyzeResponse)
async def analyze_uploaded_video(
    file: UploadFile = File(...),
    child_id: int = Form(...),
    duration_seconds: Optional[int] = Form(default=None),
    enforce_verification: bool = Form(default=True),
    challenge_token: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze a validated temporary upload and delete it after processing."""
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video type. Use MP4, MOV, AVI, MKV, or WebM.",
        )

    payload = await file.read(100 * 1024 * 1024 + 1)
    if len(payload) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded video exceeds the 100 MB limit.",
        )
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video is empty.",
        )

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension, prefix="mindbloom-video-", delete=False
        ) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name

        return analyze_video(
            VideoAnalyzeRequest(
                child_id=child_id,
                source=temp_path,
                duration_seconds=duration_seconds,
                enforce_verification=enforce_verification,
                challenge_token=challenge_token,
            ),
            current_user=current_user,
            db=db,
        )
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary uploaded video %s", temp_path)


# ========================================================
# COGNITIVE GAME SESSIONS & SCREENING HISTORY
# ========================================================

@app.post("/profiles/{child_id}/game-session/")
def log_cognitive_game_metrics(
    child_id: int,
    current_log: GameLogInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Log performance for a cognitive tracking game with ownership authorization."""
    child = verify_child_ownership(child_id, current_user, db)

    last_session = db.query(CognitiveGameLog)\
        .filter(CognitiveGameLog.child_id == child.id, CognitiveGameLog.game_name == current_log.game_name)\
        .order_by(desc(CognitiveGameLog.session_date))\
        .first()

    analysis = {"status": "Baseline data point logged", "accuracy_delta": 0.0, "speed_delta_seconds": 0.0}
    if last_session:
        acc_diff = current_log.accuracy - last_session.accuracy_score
        time_diff = last_session.completion_time_sec - current_log.completion_time_sec
        analysis["accuracy_delta"] = round(acc_diff, 2)
        analysis["speed_delta_seconds"] = round(time_diff, 2)
        analysis["status"] = "Cognitive Progress Verified" if (time_diff > 0 or acc_diff > 0) else "Stable Level Maintenance"

    new_log = CognitiveGameLog(
        child_id=child.id,
        game_name=current_log.game_name,
        accuracy_score=current_log.accuracy,
        completion_time_sec=current_log.completion_time_sec,
        reaction_time_ms=current_log.reaction_time_ms
    )
    db.add(new_log)
    db.commit()

    return {"status": "Success", "analysis": analysis}

@app.get("/profiles/{child_id}/screenings")
def get_child_screenings(
    child_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve historical screening results for a child profile."""
    child = verify_child_ownership(child_id, current_user, db)
    screenings = db.query(ScreeningTransaction).filter(ScreeningTransaction.child_id == child.id).order_by(desc(ScreeningTransaction.created_at)).all()
    return [
        {
            "id": s.id,
            "questionnaire_score": s.questionnaire_score,
            "video_score": s.video_score,
            "video_confidence": s.video_confidence,
            "hybrid_risk_score": s.hybrid_risk_score,
            "risk_tier": s.risk_tier,
            "verification_status": s.verification_status,
            "models_evaluated": s.models_evaluated,
            "created_at": s.created_at.isoformat() if s.created_at else None
        }
        for s in screenings
    ]

@app.get("/metrics")
def metrics():
    return app_state.get("metrics", {})

@app.get("/feature-importance")
def feature_importance(top_n: int = 10):
    fi = app_state.get("feature_import", {})
    if not fi:
        return {"message": "Not available"}
    return {"top_features": top_features(fi, n=top_n)}
