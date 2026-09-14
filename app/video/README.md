# MindBloom Video Behavioral Analysis Module

Computer-vision behavioral feature extraction + screening-risk estimation,
added alongside the existing questionnaire pipeline. **This module does not
diagnose autism.** It produces measurable behavioral features and, only if
a properly trained model exists, a video-derived risk estimate `Rv` that
feeds into the existing hybrid fusion with the questionnaire score.

## Pipeline

```
Webcam/video file (capture.py, OpenCV)
    -> per-frame RGB frames
    -> MediaPipe Face Mesh (mediapipe_processor.py)
       -> gaze proxy, EAR/blink, head pose, landmark movement
    -> DeepFace, sampled every N frames (deepface_processor.py)
       -> emotion probability distribution
    -> SessionData (session.py): time series + data-quality counters
    -> build_feature_vector() (features.py): 47-dim aggregated vector
    -> VideoModel.predict() (model.py): Random Forest -> Rv, confidence
    -> hybrid_fusion() (fusion.py): R_hybrid = alpha*Rq + (1-alpha)*Rv
```

## Division of responsibility (why this matters)

- **OpenCV**: capture only. No detection, no features, no scores.
- **MediaPipe**: landmark detection + geometric feature derivation only.
  Never classifies anything.
- **DeepFace**: expression/affect analysis only. Never classifies autism.
- **Random Forest (model.py)**: the *only* place a classification decision
  is made, and only from the fixed 47-feature vector.

No code path anywhere in this module does `if <cv signal> then autism = ...`.
See `app/video/model.py` for where and how the actual prediction happens.

## Running the dev harness

```bash
python -m app.video.test
```

Runs a `VIDEO_DEV_SESSION_SECONDS`-long (default 30s) session, prints
per-frame diagnostics, session quality metrics, and the full 47-feature
vector. Does **not** require a trained `video_model.pkl`.

## Configuration

All knobs live in `app/video/config.py` and are overridable via environment
variables (e.g. `VIDEO_SESSION_SECONDS=300`, `VIDEO_DEEPFACE_EVERY_N_FRAMES=5`).
None of the defaults are clinical thresholds — they are engineering
defaults, documented as such in the config file itself.

## The 47-dimensional feature schema

Defined in `app/video/features.py::VIDEO_FEATURE_SCHEMA`, persisted to
`models/video_feature_schema.json` by `train_video.py`/`save_feature_schema()`
so training and inference can never silently use a different feature order.

| Group | Count | Features |
|---|---|---|
| Gaze | 8 | horizontal/vertical mean, variance, entropy; looking-away proportion; direction-change rate |
| Blink / eye openness | 6 | blink count, rate, duration mean/variance; EAR mean/variance |
| Head pose | 8 | yaw/pitch/roll mean & variance; movement magnitude mean & variance |
| Facial movement (landmark-based, **not FACS**) | 7 | overall/mouth/eyebrow movement mean & variance; movement frequency |
| DeepFace emotion | 17 | mean & variance per emotion (7×2=14); distribution entropy; dominant-emotion switch rate; DeepFace success ratio |
| Data quality | 1 | valid_frame_ratio |
| **Total** | **47** | |

Full descriptions/calculations/units are in the schema itself (each entry
has `description`, `source`, `calculation`, `units`) and in the JSON file
once `train_video.py` has run.

## Documented limitations (read before trusting any output)

- **Gaze** is a 2D, uncalibrated iris-position-in-eye-box proxy, not a
  validated eye-tracking measurement. It does not decouple head pose from
  eye position and has no camera calibration.
- **Head pose** uses a generic (non-personalized) 3D face model via
  `cv2.solvePnP` — absolute angles are approximate; within-session relative
  changes are more trustworthy than any single reading.
- **"Facial movement"** is raw landmark displacement, explicitly **not**
  FACS Action Units. Nothing in this codebase claims FACS-level validity.
- **Blink detection** is an EAR-threshold + consecutive-frame heuristic,
  not a clinically validated blink detector.
- **DeepFace confidence/accuracy** degrades on non-frontal faces, poor
  lighting, and low-resolution webcam frames — the `deepface_success_ratio`
  and `emotion_..._variance` features partially reflect this, but expression
  outputs should not be over-trusted frame-by-frame.
- **Video RF probabilities are not necessarily calibrated** (see the
  calibration note at the bottom of `app/video/model.py`).
- **No trained `video_model.pkl` currently exists in this project.** The
  three CSVs under `data/` are questionnaire/demographic data, not
  video-derived features, and `train_video.py` will refuse to train on
  them. Until real labeled video-derived data exists and `train_video.py`
  is run against it, `/video/analyze` returns `video_model_not_available`.
- **The fusion formula `get_alpha()` in `fusion.py`** is one explicit,
  documented, swappable engineering choice — not a formula derived from or
  validated against the source paper's own experiments. Replace it if you
  have better-justified data on how video confidence should trade off
  against questionnaire reliability.

## Where things live vs. the existing project

New: everything under `app/video/`, `train_video.py`, `tests/video/`.

Modified (see `app_modified/` for drop-in replacements — diff against your
real files before applying):
- `app/database.py` — added `ScreeningTransaction.video_confidence`
  (nullable) and a new `VideoSession` table for session-level metadata/QA
  (no raw video is stored, per spec §21).
- `app/main.py` — added `POST /video/analyze`; `PredictRequest.video_score`
  default changed from a placeholder `0.5` to `None` (video is now
  genuinely optional rather than silently defaulted); `/predict` now calls
  `app.video.fusion.hybrid_fusion()` instead of the fixed-weight
  `app.utils.combine_scores()` (left in place, untouched, as a fallback/
  reference implementation).

Untouched: `app/questionnaire.py`, `app/preprocess.py`, `app/model.py`,
`train.py`, `app/cli_test.py`, `app/utils.py`, `docker-compose.yml`, and
all three existing datasets.
