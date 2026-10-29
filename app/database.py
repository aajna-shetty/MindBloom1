import os
import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./mindbloom.db"
)

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================================================
# USER AUTHENTICATION & OWNERSHIP (RESOURCE-LEVEL AUTH)
# ========================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    child_profiles = relationship(
        "ChildProfile",
        back_populates="owner",
        cascade="all, delete-orphan"
    )


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token_hash = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(Float, nullable=False)
    revoked_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


# ========================================================
# RELATIONAL DATABASE SCHEMAS FOR CLINICAL PROGRESS
# ========================================================

class ChildProfile(Base):
    __tablename__ = "child_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    age_months = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    family_history_asd = Column(Boolean, default=False)
    jaundice_history = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="child_profiles")
    screenings = relationship("ScreeningTransaction", back_populates="child", cascade="all, delete-orphan")
    game_logs = relationship("CognitiveGameLog", back_populates="child", cascade="all, delete-orphan")
    biometrics = relationship("ParticipantBiometrics", back_populates="child", uselist=False, cascade="all, delete-orphan")


class ParticipantBiometrics(Base):
    """
    Caregiver consent-governed biometric reference face template.
    Protected template storage: Stores non-reversible float embedding vector
    (e.g., Facenet-512) rather than raw facial imagery.
    """
    __tablename__ = "participant_biometrics"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child_profiles.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)

    consent_obtained = Column(Boolean, nullable=False, default=False)
    consent_caregiver_name = Column(String, nullable=True)
    consent_timestamp = Column(DateTime, nullable=True)

    face_template = Column(Text, nullable=False)  # JSON-serialized normalized embedding array
    template_model = Column(String, nullable=False, default="facenet512")
    embedding_dimension = Column(Integer, nullable=False, default=512)

    enrolled_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    child = relationship("ChildProfile", back_populates="biometrics")


class ScreeningTransaction(Base):
    __tablename__ = "screening_transactions"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)

    questionnaire_score = Column(Float, nullable=False)
    video_score = Column(Float, nullable=True)
    video_confidence = Column(Float, nullable=True)
    hybrid_risk_score = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    verification_status = Column(String, nullable=True, default="not_applicable")
    models_evaluated = Column(String, nullable=True, default="questionnaire")
    mediapipe_summary = Column(Text, nullable=True)
    vit_summary = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    child = relationship("ChildProfile", back_populates="screenings")


class CognitiveGameLog(Base):
    __tablename__ = "cognitive_game_logs"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)

    game_name = Column(String, nullable=False)

    accuracy_score = Column(Float, nullable=False)
    completion_time_sec = Column(Float, nullable=False)
    reaction_time_ms = Column(Float, nullable=False)
    session_date = Column(DateTime, default=datetime.datetime.utcnow)

    child = relationship("ChildProfile", back_populates="game_logs")