import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc


# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core Machine Learning Pipelines
from app.model import load_best_model, load_metrics, load_feature_importance
from app.preprocess import load_scaler, get_feature_list
from app.utils import risk_level, top_features, combine_scores
from app.questionnaire import build_feature_vector_from_answers, QUESTIONS

# Relational Storage Engine
from app.database import engine, Base, get_db, ChildProfile, ScreeningTransaction, CognitiveGameLog


from app.video.service import run_video_session
from app.video.schemas import VideoAnalyzeRequest, VideoAnalyzeResponse

# Initialize relational schemas seamlessly in the terminal on boot
try:
    Base.metadata.create_all(bind=engine)
    print("\n[Database] 🟢 Success: PostgreSQL structural tables synchronized flawlessly.\n")
except Exception as db_err:
    print(f"\n[Database] 🔴 Critical Connection Error: {str(db_err)}\n")

# LOGGING CONFIGURATION
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
    title="Autism Early Risk Screening API",
    description="Multimodal autism screening using questionnaire + ML with active DB memory",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================
# PYDANTIC REQ / RESP COMMUNICATORS
# ========================================================

class ProfileCreateInput(BaseModel):
    name: str
    age_months: int
    sex: str
    family_history_asd: bool = False
    jaundice_history: bool = False

class PredictRequest(BaseModel):
    child_id: int  # Linked directly to profile tracking tables
    features: Dict[str, Any]
    video_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

class FeatureItem(BaseModel):
    feature: str
    importance: float

class PredictResponse(BaseModel):
    questionnaire_score: float
    final_score: float
    risk_level: str
    top_features: List[FeatureItem]
    explanation: str

class GameLogInput(BaseModel):
    game_name: str
    accuracy: float
    completion_time_sec: float
    reaction_time_ms: float

# ========================================================
# FUNCTIONAL SERVER GATEWAYS
# ========================================================

@app.get("/health")
def health():
    return {
        "status": "ok" if app_state.get("model") else "model_not_loaded",
        "model_ready": app_state.get("model") is not None,
    }

@app.get("/questions")
def get_questions():
    return {"total": len(QUESTIONS), "questions": QUESTIONS}

@app.post("/profiles/", status_code=status.HTTP_201_CREATED)
def register_child_file(profile: ProfileCreateInput, db: Session = Depends(get_db)):
    db_profile = ChildProfile(**profile.model_dump())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    logger.info(f"Terminal Event: Registered profile successfully for ID {db_profile.id}")
    return {"status": "Success", "child_id": db_profile.id, "name": db_profile.name}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    if app_state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    child = db.query(ChildProfile).filter(ChildProfile.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=444, detail="Child profile execution requires active ID log.")

    model = app_state["model"]
    scaler = app_state["scaler"]
    feature_import = app_state.get("feature_import", {})

    try:
        X_raw = build_feature_vector_from_answers(request.features)
        X_scaled = scaler.transform([X_raw])
        q_score = float(model.predict_proba(X_scaled)[0][1])
        final_score = combine_scores(q_score, request.video_score)
        risk = risk_level(final_score)

        top_fi = []
        if feature_import:
            for f in top_features(feature_import, n=3):
                top_fi.append(FeatureItem(**f))

        explanation = (
            f"Risk level is {risk} with probability {final_score:.2f}. "
            f"Questionnaire score contributed {q_score:.2f} and video score {request.video_score:.2f}. "
        )

        # Permanent transactional archival commit
        screening_record = ScreeningTransaction(
            child_id=request.child_id,
            questionnaire_score=round(q_score, 4),
            video_score=round(request.video_score, 4),
            hybrid_risk_score=round(final_score, 4),
            risk_tier=risk
        )
        db.add(screening_record)
        db.commit()

        logger.info(f"Terminal Event: Screening saved for ID {request.child_id} -> Result: {risk}")

        return PredictResponse(
            questionnaire_score=round(q_score, 4),
            final_score=round(final_score, 4),
            risk_level=risk,
            top_features=top_fi,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/video/analyze", response_model=VideoAnalyzeResponse)
def analyze_video(
    request: VideoAnalyzeRequest,
    db: Session = Depends(get_db)
):
    """
    Run the complete video analysis pipeline.

    Webcam/file
        ↓
    OpenCV
        ↓
    MediaPipe
        ↓
    DeepFace
        ↓
    47-dimensional feature vector
        ↓
    Video Random Forest
        ↓
    Video risk score
    """

    # --------------------------------------------------
    # 1. Verify child exists
    # --------------------------------------------------

    child = (
        db.query(ChildProfile)
        .filter(ChildProfile.id == request.child_id)
        .first()
    )

    if not child:
        raise HTTPException(
            status_code=404,
            detail=f"Child profile {request.child_id} not found."
        )

    logger.info(
        "🎥 Terminal Event: Starting video analysis for child ID %s",
        request.child_id
    )

    # --------------------------------------------------
    # 2. Convert source
    # --------------------------------------------------

    source = request.source

    # VideoCaptureSession expects webcam index as int
    # when source is something like "0"
    if source.isdigit():
        source = int(source)

    logger.info(
        "🎥 Video source: %s | duration: %s seconds",
        source,
        request.duration_seconds
    )

    # --------------------------------------------------
    # 3. Run complete video pipeline
    # --------------------------------------------------

    try:

        if request.duration_seconds is not None:
            result = run_video_session(
                source=source,
                duration_seconds=request.duration_seconds,
            )
        else:
            result = run_video_session(
                source=source
            )

    except Exception as exc:
        logger.exception("🔥 Video analysis crashed")

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    # --------------------------------------------------
    # 4. Print result to terminal
    # --------------------------------------------------

    logger.info(
        "🎥 Video analysis finished | status=%s | score=%s | confidence=%s",
        result.status,
        result.video_score,
        result.video_confidence,
    )

    if result.feature_count is not None:
        logger.info(
            "🎥 Feature vector size: %s",
            result.feature_count
        )

    if result.valid_frame_ratio is not None:
        logger.info(
            "🎥 Valid frame ratio: %.3f",
            result.valid_frame_ratio
        )

    # --------------------------------------------------
    # 5. Return API response
    # --------------------------------------------------

    return VideoAnalyzeResponse(
        status=result.status,
        video_score=result.video_score,
        video_confidence=result.video_confidence,
        predicted_class=result.predicted_class,
        session_duration_seconds=result.session_duration_seconds,
        valid_frame_ratio=result.valid_frame_ratio,
        feature_count=result.feature_count,
        detail=result.detail,
    )

@app.post("/profiles/{child_id}/game-session/")
def log_cognitive_game_metrics(child_id: int, current_log: GameLogInput, db: Session = Depends(get_db)):
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Requested patient record missing.")

    # Query most recent performance delta to evaluate longitudinal progress tracker
    last_session = db.query(CognitiveGameLog)\
        .filter(CognitiveGameLog.child_id == child_id, CognitiveGameLog.game_name == current_log.game_name)\
        .order_by(desc(CognitiveGameLog.session_date))\
        .first()
    
    analysis = {"status": "Baseline data point logged", "accuracy_delta": 0.0, "speed_delta_seconds": 0.0}
    
    if last_session:
        acc_diff = current_log.accuracy - last_session.accuracy_score
        time_diff = last_session.completion_time_sec - current_log.completion_time_sec
        analysis["accuracy_delta"] = round(acc_diff, 2)
        analysis["speed_delta_seconds"] = round(time_diff, 2)
        analysis["status"] = "Cognitive Progress Verified" if (time_diff > 0 or acc_diff > 0) else "Stable Level Maintenance"

    # Save active transaction to memory tables
    new_log = CognitiveGameLog(
        child_id=child_id,
        game_name=current_log.game_name,
        accuracy_score=current_log.accuracy,
        completion_time_sec=current_log.completion_time_sec,
        reaction_time_ms=current_log.reaction_time_ms
    )
    db.add(new_log)
    db.commit()

    logger.info(f"Terminal Event: Game score saved for ID {child_id} -> {current_log.game_name}: {analysis['status']}")
    return {"status": "Success", "analysis": analysis}

@app.get("/metrics")
def metrics():
    return app_state.get("metrics", {})

@app.get("/feature-importance")
def feature_importance(top_n: int = 10):
    fi = app_state.get("feature_import", {})
    if not fi:
        return {"message": "Not available"}
    return {"top_features": top_features(fi, n=top_n)}