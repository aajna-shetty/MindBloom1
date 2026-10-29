# MindBloom

MindBloom is an experimental FastAPI-based service for early autism-risk screening. It combines questionnaire-based machine learning with an experimental computer-vision pipeline using OpenCV, MediaPipe, DeepFace, video Random Forest classification, and dynamic hybrid fusion.

> **Clinical Disclaimer:** MindBloom is an experimental screening aid, NOT a diagnostic instrument. It does not replace comprehensive evaluation by licensed medical or clinical professionals.

---

## 1. Key Capabilities

- **Secure Authentication & Resource Ownership:**
  - Salted PBKDF2-HMAC-SHA256 password hashing.
  - Cryptographically signed Bearer token session management with expiration and logout revocation.
  - Resource-level authorization: users strictly access and manage only their own child profiles and screening records.
  - Eliminates Insecure Direct Object References (IDOR) without introducing role-based access control (RBAC).
- **Consent-Governed Participant Identity Verification:**
  - Prevents accidental screening of siblings or different individuals sitting before the camera.
  - Enrolls protected, normalized face templates based on explicit caregiver consent.
  - Never permanently stores raw facial photographs.
  - Multi-frame consensus verification before video feature collection begins.
  - Continuous session monitoring: automatically pauses collection and excludes unverified frames when the child leaves the camera or another person appears.
  - Non-biometric fallbacks: questionnaire-only screening and right to biometric erasure.
- **Machine Learning & Random Forest Integration:**
  - Preserved questionnaire model pipeline (`models/best_model.pkl`, `models/scaler.pkl`) predicting probability based on AQ-10, developmental milestones, and demographic risk factors.
  - Fixed 47-dimensional behavioral video feature vector aggregating eye openness (EAR), blink dynamics, horizontal/vertical gaze entropy, head pose angles, normalized facial landmark displacement, and sampled DeepFace affective probabilities.
  - Video Random Forest wrapper (`app/video/model.py`) that surfaces `video_model_not_available` when trained video weights are not yet present, rather than fabricating video risk scores.
- **Dynamic Confidence-Weighted Hybrid Fusion:**
  - $R_{hybrid} = \alpha R_q + (1 - \alpha) R_v$, where $\alpha$ dynamically balances model influence based on video confidence.
  - Automatically collapses to $R_q$ ($\alpha = 1.0$) when video analysis is unavailable or insufficient.

---

## 2. Project Layout

```text
app/
  ├── auth.py                  Password hashing, bearer tokens & IDOR prevention
  ├── database.py              SQLAlchemy models (User, ChildProfile, Biometrics, Screenings)
  ├── main.py                  FastAPI application routes and endpoint handlers
  ├── model.py                 Questionnaire model loading & evaluation
  ├── preprocess.py            Data cleaning, merging, and scaling
  ├── questionnaire.py         AQ-10 questions and feature vector builder
  ├── utils.py                 Risk banding and utility functions
  └── video/
      ├── capture.py           OpenCV video capture & frame generator
      ├── config.py            Session durations and threshold constants
      ├── deepface_processor.py Sampled affective expression extraction
      ├── features.py          Fixed 47-dimensional video feature schema
      ├── fusion.py            Confidence-weighted hybrid fusion
      ├── mediapipe_processor.py Landmark geometry (EAR, gaze, head pose, movement)
      ├── model.py             Video Random Forest wrapper
      ├── schemas.py           Pydantic request & response models
      ├── service.py           Video screening orchestration with verification
      ├── session.py           Time-series bookkeeping & data quality metrics
      └── verification.py      Participant biometric verification engine
data/                          Autism questionnaire datasets
models/                        Trained artifacts & MediaPipe face landmarker task
tests/                         Automated unit, integration, and security test suites
.env.example                   Configuration template
```

---

## 3. Setup and Installation

Python 3.10 or newer is recommended.

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the example environment configuration:

```powershell
cp .env.example .env
```

For local PowerShell sessions, configure the values explicitly:

```powershell
$env:MINDBLOOM_SECRET_KEY = "use-a-long-random-development-secret"
$env:MINDBLOOM_ALLOWED_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"
```

Production deployments must set `MINDBLOOM_SECRET_KEY`. The API rejects a
missing secret when `MINDBLOOM_ENV=production`.

### Database migrations

The repository includes Alembic configuration. Apply the schema through the
migration history rather than relying on `create_all()` for production
deployments:

```powershell
alembic upgrade head
```

The initial migration is in
`migrations/versions/0001_initial_schema.py`. Existing databases should be
backed up and reviewed before migrating.

---

## 4. Running the API

```powershell
uvicorn app.main:app --reload
```

Once running, interactive documentation is available at `http://127.0.0.1:8000/docs`.

### Primary API Endpoints:

#### Authentication
- `POST /auth/register` - Create a user account
- `POST /auth/login` - Authenticate with email/password and obtain Bearer token
- `POST /auth/logout` - Revoke current Bearer token
- `GET /auth/me` - View current authenticated user details

#### Child Profiles (Protected)
- `POST /profiles/` - Register a child profile owned by the authenticated user
- `GET /profiles/` - List child profiles owned by the user
- `GET /profiles/{child_id}` - Retrieve child profile details
- `DELETE /profiles/{child_id}` - Delete child profile and all associated records

#### Participant Identity Verification
- `POST /profiles/{child_id}/enroll-face` - Consent-based reference face template enrollment
- `POST /profiles/{child_id}/enroll-face/upload` - Multipart upload enrollment with image validation
- `POST /profiles/{child_id}/enroll-face/capture` - Base64 camera-capture enrollment with image validation
- `GET /profiles/{child_id}/biometrics` - Query biometric enrollment status
- `DELETE /profiles/{child_id}/biometrics` - Right to erasure (delete biometric template)

Enrollment requires a real image containing exactly one detectable face. The
JSON endpoint does not create synthetic templates. Surrogate embeddings are
disabled by default and must only be enabled explicitly for isolated tests with
`MINDBLOOM_ALLOW_SURROGATE_EMBEDDINGS=true`.

Server-side video file paths are disabled by default. To enable local file
analysis, set `MINDBLOOM_ALLOW_SERVER_FILE_PATHS=true` and configure
`MINDBLOOM_ALLOWED_SOURCE_DIR` to the directory containing approved files.
Uploaded videos are limited to 100 MB, are processed from a temporary file, and
are deleted after analysis. Passive motion checks reduce static-image/replay
risk but are not a validated liveness or presentation-attack detector.

#### Screening & Predictions
- `GET /questions` - Retrieve questionnaire questions (authenticated)
- `POST /predict` - Calculate questionnaire or hybrid risk score
- `POST /video/analyze` - Run webcam or uploaded video analysis with participant verification
- `POST /video/analyze/upload` - Validate and analyze a temporary video upload
- `GET /profiles/{child_id}/screenings` - Retrieve historical screening transactions

---

## 5. Running Automated Tests

Run the complete automated test suite:

```powershell
pytest -v
```

The test suite includes:
- Authentication & password hashing (`tests/test_auth.py`)
- Cross-user IDOR isolation (`tests/test_auth.py`)
- Participant identity verification (`tests/test_participant_verification.py`)
- Continuous session monitoring & frame exclusion (`tests/test_video_service_verification.py`)
- Hybrid fusion & API screening persistence (`tests/test_api_screening_fusion.py`)
- Core video capture, geometry, features, and session tests (`tests/video/`)