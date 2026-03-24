"""
Sign Language Recognition (SLR) pipeline.

This package implements a complete pipeline for sign language recognition using
MediaPipe landmarks and deep learning sequence models.

Pipeline stages:
    1. Extraction: Extract face, hand, and pose landmarks from sign videos
    2. Split: Generate persistent train/val/test split manifests
    3. Training: Train a sequence model to classify signs from landmark sequences
    4. Inference: Recognize signs in real-time from webcam or video files
    5. Validation: Verify preprocessing consistency between training and inference

Modules:
    config            - Configuration constants, paths, hyperparameters
    landmarks         - MediaPipe landmarker setup, detection, keypoint extraction
    extraction        - Video processing & keypoint extraction
    split             - Signer-independent or random train/val/test split generation
    dataset           - LazySignDataset, data scanning, scaler, DataLoaders
    train             - Training loop, evaluation, model saving
    inference         - Real-time and video inference (trigger, motion, continuous modes)
    validate          - Validation comparing training vs inference pipelines
    visualization     - Drawing utilities for landmarks and inference HUD
    convert_landmarks - One-time batch conversion of .npy files (478->83 face landmarks)
    main              - CLI entry point for all commands
"""
