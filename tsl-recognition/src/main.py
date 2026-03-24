#!/usr/bin/env python3
"""
CLI entry point for the Sign Language Recognition pipeline.

Usage::

    # From the project root directory:
    python -m src split                                       # BosphorusSign22k signer split
    python -m src split --dataset autsl                       # AUTSL predefined split
    python -m src split --split-mode random                   # stratified random split
    python -m src extract                                     # extract keypoints (Bosphorus)
    python -m src extract --dataset autsl                     # extract keypoints (AUTSL)
    python -m src convert                                     # dry-run .npy 1692->507 conversion
    python -m src convert --apply                             # apply .npy conversion
    python -m src train                                       # train with default GRU
    python -m src train --dataset autsl                       # train GRU on AUTSL
    python -m src evaluate                                    # re-evaluate latest run
    python -m src evaluate --all                              # re-evaluate ALL past runs
    python -m src infer --mode motion                         # real-time inference
    python -m src validate                                    # validate inference pipeline
    python -m src validate --run-dir trained-models/run_*     # validate specific run
    python -m src infer --mode video --video path/to/video.mp4

Add ``--test`` to any subcommand for a quick 10-class smoke test.
"""

from __future__ import annotations

import argparse
import sys

from .config import TrainConfig
from .datasets import DATASET_CHOICES


def _make_config(args) -> TrainConfig:
    """Build a TrainConfig from parsed CLI arguments."""
    dataset = getattr(args, "dataset", "bosphorus") or "bosphorus"
    if getattr(args, "test", False):
        cfg = TrainConfig.test(dataset=dataset)
    else:
        cfg = TrainConfig.full(dataset=dataset)
    return cfg


def cmd_split(args):
    """Execute the split generation command."""
    from .split import generate_split
    cfg = _make_config(args)
    # Use dataset default split mode unless explicitly overridden
    if args.split_mode is not None:
        cfg.split_mode = args.split_mode
    else:
        cfg.split_mode = cfg.dataset_info.default_split_mode
    generate_split(cfg)


def cmd_extract(args):
    """Execute the keypoint extraction command."""
    from .extraction import run_extraction
    cfg = _make_config(args)
    run_extraction(cfg, num_workers=args.num_workers)


def cmd_train(args):
    """Execute the model training command."""
    from .train import train
    cfg = _make_config(args)
    if hasattr(args, "split_mode") and args.split_mode:
        cfg.split_mode = args.split_mode
    if hasattr(args, "model") and args.model:
        cfg.model_arch = args.model
    if args.label_smoothing is not None:
        cfg.label_smoothing = args.label_smoothing
    if hasattr(args, "model_size") and args.model_size:
        cfg.model_size_override = args.model_size
    train(cfg)


def cmd_infer(args):
    """Execute the inference command."""
    from .inference import InferenceMode, run_inference
    mode_map = {
        "trigger": InferenceMode.TRIGGER,
        "motion": InferenceMode.MOTION,
        "continuous": InferenceMode.CONTINUOUS,
        "video": InferenceMode.VIDEO_FILE,
    }
    cfg = _make_config(args)
    run_inference(
        mode=mode_map[args.mode],
        cfg=cfg,
        video_path=args.video,
        run_dir=getattr(args, "run_dir", None),
        show_landmarks=not args.no_landmarks,
        headless=args.headless,
    )


def cmd_convert(args):
    """Execute the landmark conversion command."""
    from .convert_landmarks import run_conversion
    run_conversion(dry_run=not args.apply)


def cmd_evaluate(args):
    """Re-evaluate a saved model on the test set (top-1 & top-5 accuracy)."""
    from .evaluate import evaluate_all, evaluate_run
    dataset = getattr(args, "dataset", "bosphorus") or "bosphorus"
    if args.all:
        evaluate_all(dataset=dataset)
    else:
        evaluate_run(run_dir=getattr(args, "run_dir", None), dataset=dataset)


def cmd_validate(args):
    """Execute the validation command."""
    from .validate import run_validation
    dataset = getattr(args, "dataset", "bosphorus") or "bosphorus"
    run_validation(run_dir=getattr(args, "run_dir", None), n_samples=args.samples, dataset=dataset)


def main():
    """Main CLI entry point for the Sign Language Recognition pipeline."""
    parser = argparse.ArgumentParser(
        prog="slr",
        description="Sign Language Recognition pipeline",
    )

    # Shared parent parsers so flags work before OR after the subcommand
    shared_parent = argparse.ArgumentParser(add_help=False)
    shared_parent.add_argument(
        "--test", action="store_true",
        help="Use 10-class test config instead of full training",
    )
    shared_parent.add_argument(
        "--dataset", choices=DATASET_CHOICES, default="bosphorus",
        help="Dataset to use (default: bosphorus)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # split
    p_split = sub.add_parser("split", parents=[shared_parent],
                              help="Generate train/val/test split manifests")
    p_split.add_argument(
        "--split-mode", choices=["signer", "random", "predefined"], default=None,
        help="Split strategy (default: dataset-specific)",
    )

    # extract
    p_extract = sub.add_parser("extract", parents=[shared_parent],
                               help="Extract keypoints from sign-language videos")
    p_extract.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = sequential)",
    )

    # train
    p_train = sub.add_parser("train", parents=[shared_parent],
                              help="Train a sign language recognition model")
    p_train.add_argument(
        "--model",
        choices=["gru"],
        default=None,
        help="Model architecture (default: gru)",
    )
    p_train.add_argument(
        "--split-mode", choices=["signer", "random", "predefined"], default=None,
        help="Override split mode for scaler selection",
    )
    p_train.add_argument(
        "--label-smoothing", type=float, default=None,
        help="Label smoothing value (default: 0.1; use 0 to disable)",
    )
    p_train.add_argument(
        "--model-size", choices=["small", "large", "xlarge"], default=None,
        help="Override model size preset (default: auto-detect from class count)",
    )

    # infer
    p_infer = sub.add_parser("infer", parents=[shared_parent],
                              help="Run real-time or video inference")
    p_infer.add_argument(
        "--mode", choices=["trigger", "motion", "continuous", "video"],
        default="motion", help="Inference mode (default: motion)",
    )
    p_infer.add_argument("--video", type=str, default=None, help="Video path for video mode")
    p_infer.add_argument("--run-dir", type=str, default=None,
                          help="Path to a run_* directory (default: latest)")
    p_infer.add_argument("--no-landmarks", action="store_true", help="Hide landmark overlay")
    p_infer.add_argument("--headless", action="store_true",
                          help="Skip GUI window (auto-detected when no display)")

    # convert
    p_conv = sub.add_parser("convert", help="Convert .npy files from 1692-dim to 507-dim")
    p_conv.add_argument("--apply", action="store_true",
                         help="Actually overwrite files (default is dry-run)")

    # evaluate
    p_eval = sub.add_parser("evaluate", parents=[shared_parent],
                             help="Re-evaluate a saved model on the test set")
    p_eval.add_argument("--run-dir", type=str, default=None,
                         help="Path to a run_* directory (default: latest)")
    p_eval.add_argument("--all", action="store_true",
                         help="Re-evaluate ALL run_* directories")

    # validate
    p_val = sub.add_parser("validate", parents=[shared_parent],
                            help="Validate inference vs training pipeline")
    p_val.add_argument("--samples", type=int, default=100,
                       help="Number of samples to validate")
    p_val.add_argument("--run-dir", type=str, default=None,
                       help="Path to a run_* directory (default: latest)")

    args = parser.parse_args()

    dispatch = {
        "split": cmd_split,
        "extract": cmd_extract,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
        "convert": cmd_convert,
        "validate": cmd_validate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
