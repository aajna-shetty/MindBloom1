"""
tests/test_emotion_visual.py
Interactive visual test for ViT Facial Expression Recognition.
"""

import os
import sys
import time
import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.video.emotion_processor import EmotionProcessor
from app.video.mediapipe_processor import MediaPipeProcessor


# Shared state for async ViT worker
frame_queue = queue.Queue(maxsize=1)
result_lock = threading.Lock()
latest_result = {
    "dominant_emotion": "neutral",
    "emotion_scores": {e: 0.0 for e in EmotionProcessor.STANDARD_EMOTIONS},
    "inference_time_ms": 0.0
}
stop_event = threading.Event()


def vit_worker(processor: EmotionProcessor):
    """Background worker that continuously processes the latest face crop."""
    global latest_result
    while not stop_event.is_set():
        try:
            face_crop = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        t0 = time.perf_counter()
        res = processor.analyze(face_crop)
        inf_ms = (time.perf_counter() - t0) * 1000.0

        if res["success"]:
            with result_lock:
                latest_result = {
                    "dominant_emotion": res["dominant_emotion"],
                    "emotion_scores": res["emotion_scores"],
                    "inference_time_ms": inf_ms
                }
            # Detailed debug printout
            scores = res["emotion_scores"]
            print(
                f"[ViT] Dom: {res['dominant_emotion']:<8} | "
                f"Ang: {scores['angry']*100:4.1f}% | "
                f"Dis: {scores['disgust']*100:4.1f}% | "
                f"Neu: {scores['neutral']*100:4.1f}% | "
                f"Sad: {scores['sad']*100:4.1f}% | "
                f"Lat: {inf_ms:4.0f}ms"
            )

        frame_queue.task_done()


def run_visual_test():
    print("=" * 60)
    print("STARTING ViT EMOTION VISUAL TEST")
    print("=" * 60)

    # 1. Initialize MediaPipe FaceLandmarker
    t0 = time.perf_counter()
    model_path = os.path.join("models", "face_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    print(f"[INIT] MediaPipe Task:      {(time.perf_counter() - t0) * 1000.0:6.1f} ms")

    # 2. Initialize ViT Emotion Processor
    t0 = time.perf_counter()
    emotion_proc = EmotionProcessor()
    print(f"[INIT] ViT Model & Warmup:  {(time.perf_counter() - t0) * 1000.0:6.1f} ms")

    # 3. Start background worker thread
    worker_thread = threading.Thread(target=vit_worker, args=(emotion_proc,), daemon=True)
    worker_thread.start()

    # 4. Open Camera
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(0)
    print(f"[INIT] Camera Open:         {(time.perf_counter() - t0) * 1000.0:6.1f} ms")

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        stop_event.set()
        return

    print("\n[INFO] Camera active. Press 'q' in the window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run MediaPipe face landmarker
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_result = landmarker.detect(mp_image)

        if mp_result.face_landmarks:
            landmarks = mp_result.face_landmarks[0]
            bbox = MediaPipeProcessor.extract_face_bbox(landmarks, frame.shape)
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)

                # Send face crop to worker thread if ready
                if frame_queue.empty():
                    crop = frame[y1:y2, x1:x2].copy()
                    frame_queue.put(crop)

        # Retrieve current snapshot of emotion scores
        with result_lock:
            dominant = latest_result["dominant_emotion"]
            scores = dict(latest_result["emotion_scores"])
            latency = latest_result["inference_time_ms"]

        # Render UI overlay
        # Dominant Emotion Header
        header_color = (0, 0, 255) if dominant == "disgust" else ((0, 255, 0) if dominant == "happy" else (255, 255, 255))
        cv2.putText(frame, f"Emotion: {dominant.upper()}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, header_color, 2)
        cv2.putText(frame, f"ViT latency: {latency:4.0f} ms", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # 7-Class breakdown bars
        y_offset = 85
        for emo, score in scores.items():
            pct = score * 100.0
            is_dominant = (emo == dominant)
            bar_color = (0, 200, 255) if is_dominant else (140, 140, 140)
            
            # Label
            cv2.putText(frame, f"{emo:<8}: {pct:5.1f}%", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1 if not is_dominant else 2)
            
            # Probability Bar
            bar_len = int(score * 120)
            cv2.rectangle(frame, (160, y_offset - 10), (160 + bar_len, y_offset - 2), bar_color, -1)
            cv2.rectangle(frame, (160, y_offset - 10), (280, y_offset - 2), (80, 80, 80), 1)

            y_offset += 22

        cv2.imshow("MindBloom - ViT Emotion Recognition Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    print("\nVisual test ended.")


if __name__ == "__main__":
    run_visual_test()