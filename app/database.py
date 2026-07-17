import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Configured for your isolated docker container port 5435
DATABASE_URL = "postgresql://postgres:aajna@localhost:5435/mindbloom"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================================================
# RELATIONAL DATABASE SCHEMAS FOR CLINICAL PROGRESS
# ========================================================

class ChildProfile(Base):
    __tablename__ = "child_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age_months = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    family_history_asd = Column(Boolean, default=False)
    jaundice_history = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    screenings = relationship("ScreeningTransaction", back_populates="child", cascade="all, delete-orphan")
    game_logs = relationship("CognitiveGameLog", back_populates="child", cascade="all, delete-orphan")


class ScreeningTransaction(Base):
    __tablename__ = "screening_transactions"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child_profiles.id", ondelete="CASCADE"))
    
    questionnaire_score = Column(Float, nullable=False)
    video_score = Column(Float, nullable=False)
    hybrid_risk_score = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    child = relationship("ChildProfile", back_populates="screenings")


class CognitiveGameLog(Base):
    __tablename__ = "cognitive_game_logs"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child_profiles.id", ondelete="CASCADE"))
    
    # "Attention Arena", "Emotion Mirror", "Memory Match", "Eye Tracking Challenge", "Pattern Recall"
    game_name = Column(String, nullable=False)
    
    accuracy_score = Column(Float, nullable=False)
    completion_time_sec = Column(Float, nullable=False)
    reaction_time_ms = Column(Float, nullable=False)
    session_date = Column(DateTime, default=datetime.datetime.utcnow)

    child = relationship("ChildProfile", back_populates="game_logs")