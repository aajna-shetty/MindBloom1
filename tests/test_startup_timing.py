"""
tests/test_startup_timing.py
Isolates and benchmarks startup latencies for Camera, MediaPipe, and ViT.
"""

import os
import time
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

def run_benchmark():
    print("=" * 60)
    print("STARTING MINDBLOOM STARTUP TIMING BENCHMARK")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Benchmark Camera Startup (Default vs DirectShow)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    cap_default = cv2.VideoCapture(0)
    _ = cap_default.read()
    t_default = (time.perf_counter() - t0) * 1000.0
    cap_default.release()

    t0 = time.perf_counter()
    cap_dshow = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    _ = cap_dshow.read()
    t_dshow = (time.perf_counter() - t0) * 1000.0
    cap_dshow.release()

    print(f"[TIMING] Camera Open (Default MSMF):   {t_default:8.2f} ms")
    print(f"[TIMING] Camera Open (DirectShow):     {t_dshow:8.2f} ms")

    # -------------------------------------------------------------
    # 2. Benchmark MediaPipe FaceLandmarker Initialization
    # -------------------------------------------------------------
    model_path = os.path.join("models", "face_landmarker.task")
    if not os.path.exists(model_path):
        print(f"[ERROR] MediaPipe task not found at {model_path}")
        return

    t0 = time.perf_counter()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    t_mp_init = (time.perf_counter() - t0) * 1000.0
    print(f"[TIMING] MediaPipe Task Init:          {t_mp_init:8.2f} ms")

    # -------------------------------------------------------------
    # 3. Benchmark ViT Model Download / Initialization
    # -------------------------------------------------------------
    model_id = "mo-thecreator/vit-Facial-Expression-Recognition"
    print(f"\n[INFO] Loading ViT: {model_id} ...")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    t_vit_load = (time.perf_counter() - t0) * 1000.0
    print(f"[TIMING] ViT Weights & Processor Load: {t_vit_load:8.2f} ms")

    # -------------------------------------------------------------
    # 4. Benchmark First Inference (Warm-up) vs Steady State
    # -------------------------------------------------------------
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # First Inference
    t0 = time.perf_counter()
    inputs = processor(images=dummy_img, return_tensors="pt")
    with torch.inference_mode():
        _ = model(**inputs)
    t_first_inf = (time.perf_counter() - t0) * 1000.0
    print(f"[TIMING] ViT First Inference (Warmup):  {t_first_inf:8.2f} ms")

    # Steady State Average (10 runs)
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        inputs = processor(images=dummy_img, return_tensors="pt")
        with torch.inference_mode():
            _ = model(**inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    print(f"[TIMING] ViT Average Steady Inference: {np.mean(latencies):8.2f} ms")
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()