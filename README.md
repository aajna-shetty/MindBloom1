# MindBloom

MindBloom is a work-in-progress FastAPI service for early autism-risk screening. It combines questionnaire-based machine learning with an experimental video-analysis pipeline using OpenCV, MediaPipe, and DeepFace.

## Current scope

- Questionnaire and hybrid risk prediction API
- Child profile and screening-result persistence through SQLAlchemy
- Experimental webcam/file video analysis pipeline
- Model artifacts and training utilities included in the repository
- Automated tests for core video-processing components

This project is not a diagnostic tool. It is an experimental screening aid and still requires validation, usability work, and production hardening before real-world use.

## Project layout

```text
app/              FastAPI application and screening logic
app/video/        Video capture, feature extraction, fusion, and analysis
data/              Training datasets
models/            Trained model artifacts and MediaPipe task asset
tests/             Automated tests
train.py          Questionnaire model training script
train_video.py    Video model training script
```

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The video pipeline expects `models/face_landmarker.task` to be present. The current repository includes this asset.

## Configuration

The application uses the database configuration in `app/database.py`. Set the relevant environment variables for PostgreSQL before starting the API, or use the repository's Docker Compose configuration where appropriate.

## Run the API

```powershell
uvicorn app.main:app --reload
```

Once running, open the interactive API documentation at `http://127.0.0.1:8000/docs`.

Useful endpoints include:

- `GET /health` - check API and model readiness
- `GET /questions` - retrieve questionnaire questions
- `POST /profiles/` - create a child profile
- `POST /predict` - calculate a questionnaire or hybrid risk score
- `POST /video/analyze` - run video analysis for an existing profile

## Run tests

```powershell
pytest
```

Some video and DeepFace tests require a working camera or installed model/runtime assets. Hardware-dependent tests may need to be run separately in a configured environment.

## Train models

```powershell
python train.py
python train_video.py
```

Training scripts may update files under `models/`. Review generated artifacts before committing them.

## Status

MindBloom is not fully complete. Known areas still under development include end-to-end product flows, model validation, privacy and security controls, deployment configuration, and production-grade error handling.