"""
Interpolate missing pose/hand/face landmark segments in sequences of flat keypoint vectors.

MediaPipe sometimes fails to detect hands or pose for a few frames (occlusion, motion blur).
Missing parts are stored as zeros. This module fills those gaps using linear interpolation
from the nearest valid frames so downstream models see continuous trajectories.

Layout (must match extract_keypoints output):
  pose:  POSE_LANDMARKS * 4   (x, y, z, visibility)
  face:  FACE_LANDMARKS * 3   (x, y, z)   — reduced subset (83 points)
  left:  HAND_LANDMARKS * 3
  right: HAND_LANDMARKS * 3
"""

from __future__ import annotations

import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 83  # reduced from 478 to linguistically relevant subset

# Offsets and sizes in the flat vector (same order as extract_keypoints)
_IDX_POSE = 0
_LEN_POSE = POSE_LANDMARKS * 4
_IDX_FACE = _IDX_POSE + _LEN_POSE
_LEN_FACE = FACE_LANDMARKS * 3
_IDX_LEFT = _IDX_FACE + _LEN_FACE
_LEN_LEFT = HAND_LANDMARKS * 3
_IDX_RIGHT = _IDX_LEFT + _LEN_LEFT
_LEN_RIGHT = HAND_LANDMARKS * 3
EXPECTED_DIM = _LEN_POSE + _LEN_FACE + _LEN_LEFT + _LEN_RIGHT


def _slice_part(seq: np.ndarray, start: int, length: int) -> np.ndarray:
    """(T, D) -> (T, length) for the segment [start:start+length]."""
    return seq[:, start : start + length].copy()


def _is_missing(part: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """(T, ...) -> (T,) bool True where that part is missing (all zeros or near-zero)."""
    if part.ndim == 2:
        # Consider missing if all coords (and visibility if present) are ~0
        return np.all(np.abs(part) <= tol, axis=1)
    return np.all(np.abs(part) <= tol)


def _interpolate_part(valid_mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Fill missing positions (where valid_mask is False) by linear interpolation.
    values: (T, D), valid_mask: (T,) bool. Interpolates each dimension across time.
    """
    T, D = values.shape
    out = values.copy()
    if np.all(valid_mask):
        return out

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return out
    if len(valid_idx) == 1:
        out[~valid_mask] = values[valid_idx[0]]
        return out

    # Per dimension: interpolate missing time indices from valid ones
    missing_idx = np.where(~valid_mask)[0]
    for d in range(D):
        out[missing_idx, d] = np.interp(
            missing_idx,
            valid_idx,
            values[valid_idx, d],
        )
    return out


def interpolate_missing_keypoints(
    seq: np.ndarray,
    *,
    missing_tol: float = 1e-6,
) -> np.ndarray:
    """
    Fill missing pose/face/left_hand/right_hand segments in a keypoint sequence by
    linear interpolation from neighboring valid frames.

    Parameters
    ----------
    seq : np.ndarray
        Shape (T, D) with D = EXPECTED_DIM (pose + face + left_hand + right_hand flat).
    missing_tol : float
        Treat a part as missing when all its values are <= this in absolute value.

    Returns
    -------
    np.ndarray
        (T, D) with missing segments filled by interpolation. Unchanged where already valid.
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Expected shape (T, {EXPECTED_DIM}); got {seq.shape}"
        )

    T = seq.shape[0]
    out = seq.copy()

    parts = [
        ("pose", _IDX_POSE, _LEN_POSE),
        ("face", _IDX_FACE, _LEN_FACE),
        ("left_hand", _IDX_LEFT, _LEN_LEFT),
        ("right_hand", _IDX_RIGHT, _LEN_RIGHT),
    ]
    for _name, start, length in parts:
        part = _slice_part(out, start, length)
        missing = _is_missing(part, tol=missing_tol)
        if not np.any(missing):
            continue
        filled = _interpolate_part(~missing, part)
        out[:, start : start + length] = filled

    return out


def count_missing_per_frame(seq: np.ndarray, missing_tol: float = 1e-6) -> dict[str, np.ndarray]:
    """
    For debugging: report which parts are missing in each frame.

    Returns
    -------
    dict
        Keys "pose", "face", "left_hand", "right_hand". Values are (T,) bool arrays
        True where that part is missing.
    """
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected shape (T, {EXPECTED_DIM}); got {seq.shape}")

    result = {}
    for name, start, length in [
        ("pose", _IDX_POSE, _LEN_POSE),
        ("face", _IDX_FACE, _LEN_FACE),
        ("left_hand", _IDX_LEFT, _LEN_LEFT),
        ("right_hand", _IDX_RIGHT, _LEN_RIGHT),
    ]:
        part = _slice_part(seq, start, length)
        result[name] = _is_missing(part, tol=missing_tol)
    return result
