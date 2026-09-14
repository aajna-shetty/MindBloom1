"""
app/video/test.py
─────────────────────────────────────────────────────────────────────────────
Manual development harness — run with:

    python -m app.video.test

Exercises webcam capture, MediaPipe detection, DeepFace detection, and
feature extraction WITHOUT requiring a trained video_model.pkl (per §23 of
spec). Uses config.DEV_SESSION_DURATION_SECONDS (default 30s) instead of
the full 5-minute production session so you're not stuck waiting during
development.

This intentionally duplicates a slice of service.run_video_session's loop
rather than importing it, so you can see per-frame diagnostics live in the
terminal (face detected? landmarks ok? deepface hit or miss?) — useful for
debugging camera/lighting/model issues before wiring the full pipeline.
"""

from __future__ import annotations

import sys
import logging

from app.video.capture import VideoCaptureSession, CameraError
from app.video.mediapipe_processor import MediaPipeProcessor
from app.video.session import SessionData
from app.video.features import build_feature_vector, VIDEO_FEATURES
from app.video.config import DEV_SESSION_DURATION_SECONDS, DEEPFACE_EVERY_N_FRAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("video.test")


def main(duration_seconds: int = DEV_SESSION_DURATION_SECONDS, source=0):
    print("=" * 60)
    print("  MindBloom Video Module — Development Test Harness")
    print(f"  duration={duration_seconds}s  source={source}  "
          f"deepface_every_n_frames={DEEPFACE_EVERY_N_FRAMES}")
    print("=" * 60)

    try:
        df_proc = None
        try:
            from app.video.deepface_processor import DeepFaceProcessor
            df_proc = DeepFaceProcessor()
            print("[test] DeepFace available.")
        except ImportError as exc:
            print(f"[test] DeepFace NOT available ({exc}); continuing without emotion features.")

        session = SessionData()

        with VideoCaptureSession(source=source, duration_seconds=duration_seconds) as cap:
            with MediaPipeProcessor() as mp_proc:
                for frame in cap.frames():
                    if not frame.valid or frame.rgb_frame is None:
                        session.record_capture_failure()
                        print(f"  frame {frame.index:04d} | INVALID ({frame.error})")
                        continue

                    result = mp_proc.process_frame(frame.rgb_frame)
                    session.record_landmark_result(frame.timestamp, result)

                    line = f"  frame {frame.index:04d} t={frame.timestamp:5.2f}s "
                    if not result.face_detected:
                        line += "| no face"
                    else:
                        line += (
                            f"| faces={result.num_faces} "
                            f"gaze=({result.gaze_horizontal}, {result.gaze_vertical}) "
                            f"ear={result.ear} "
                            f"pose=(yaw={result.yaw}, pitch={result.pitch}, roll={result.roll})"
                        )

                    if df_proc is not None and frame.index % max(1, DEEPFACE_EVERY_N_FRAMES) == 0:
                        df_result = df_proc.analyze_frame(frame.rgb_frame)
                        session.record_deepface_result(df_result)
                        if df_result.success:
                            line += f" | deepface={df_result.dominant_emotion}"
                        else:
                            line += f" | deepface_failed={df_result.error}"

                    print(line)

        print("\n" + "-" * 60)
        print("SESSION QUALITY METRICS")
        print("-" * 60)
        for k, v in session.quality_metrics().items():
            print(f"  {k}: {v}")

        if not session.has_sufficient_data():
            print("\n[test] INSUFFICIENT_VIDEO_DATA — not building a feature vector.")
            return

        feats = build_feature_vector(session)
        print("\n" + "-" * 60)
        print(f"47-DIM FEATURE VECTOR ({len(VIDEO_FEATURES)} features)")
        print("-" * 60)
        for name in VIDEO_FEATURES:
            print(f"  {name}: {feats[name]:.4f}")

        print("\n[test] Feature extraction complete. No trained video_model.pkl "
              "is required for this harness — see train_video.py to train one "
              "once labeled video-derived data is available.")

    except CameraError as exc:
        print(f"[test] Camera error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
