"""
app.video
─────────────────────────────────────────────────────────────────────────────
Computer-vision behavioral analysis module for MindBloom.

This package ONLY produces measurable behavioral features from webcam/video
input (via OpenCV + MediaPipe + DeepFace) and, if a trained model artifact
exists, a video-derived ASD-screening risk estimate (NOT a diagnosis).

None of the CV libraries used here (OpenCV, MediaPipe, DeepFace) classify a
person as autistic. Classification only happens in app.video.model, using a
separately trained Random Forest that consumes the 47-dimensional feature
vector defined in app.video.features.
"""
