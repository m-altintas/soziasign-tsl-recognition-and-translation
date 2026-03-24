"""
One-time batch conversion of existing .npy keypoint files from the old
1692-dimensional layout (478 face landmarks) to the new 507-dimensional
layout (83 face landmarks).

The column mapping is:
    OLD [1692]:  pose(33*4=132) | face(478*3=1434) | left_hand(21*3=63) | right_hand(21*3=63)
    NEW [507]:   pose(33*4=132) | face(83*3=249)   | left_hand(21*3=63) | right_hand(21*3=63)

Only the face columns change — pose and hand columns are copied verbatim.
Selected face columns correspond to ``FACE_LANDMARK_INDICES`` defined in
``src.config``.

After conversion the cached scaler files are deleted so they will be
recomputed on the next training run.

Usage::

    python -m src.convert_landmarks            # dry-run (default)
    python -m src.convert_landmarks --apply     # actually overwrite files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import (
    DATA_PATH,
    FACE_LANDMARK_INDICES,
    FACE_LANDMARKS_FULL,
    HAND_LANDMARKS,
    POSE_LANDMARKS,
)

# ---------------------------------------------------------------------------
# Old layout constants (before the reduction)
# ---------------------------------------------------------------------------
_OLD_POSE_LEN = POSE_LANDMARKS * 4          # 132
_OLD_FACE_LEN = FACE_LANDMARKS_FULL * 3     # 1434
_OLD_LH_LEN = HAND_LANDMARKS * 3            # 63
_OLD_RH_LEN = HAND_LANDMARKS * 3            # 63
_OLD_DIM = _OLD_POSE_LEN + _OLD_FACE_LEN + _OLD_LH_LEN + _OLD_RH_LEN  # 1692

# Compute the column indices to keep from the old 1692-dim vector
_POSE_COLS = list(range(0, _OLD_POSE_LEN))                                       # 0..131
_FACE_COLS = [_OLD_POSE_LEN + i * 3 + c
              for i in FACE_LANDMARK_INDICES for c in range(3)]                   # selected face xyz
_LH_COLS = list(range(_OLD_POSE_LEN + _OLD_FACE_LEN,
                       _OLD_POSE_LEN + _OLD_FACE_LEN + _OLD_LH_LEN))             # 1566..1628
_RH_COLS = list(range(_OLD_POSE_LEN + _OLD_FACE_LEN + _OLD_LH_LEN, _OLD_DIM))   # 1629..1691
KEEP_COLS = np.array(_POSE_COLS + _FACE_COLS + _LH_COLS + _RH_COLS, dtype=np.intp)

_NEW_DIM = len(KEEP_COLS)  # 507


def convert_file(path: Path, *, dry_run: bool = True) -> str:
    """Convert a single .npy file in-place.

    Returns a status string: ``"converted"``, ``"already_new"``, ``"skipped"``,
    or ``"error: ..."``.
    """
    try:
        data = np.load(path)
    except Exception as exc:
        return f"error: load failed ({exc})"

    if data.ndim != 2:
        return f"skipped (ndim={data.ndim})"

    _, feat = data.shape

    if feat == _NEW_DIM:
        return "already_new"

    if feat != _OLD_DIM:
        return f"skipped (dim={feat})"

    new_data = data[:, KEEP_COLS].copy()

    if not dry_run:
        np.save(path, new_data)

    return "converted"


def run_conversion(
    data_path: Path | None = None,
    *,
    dry_run: bool = True,
) -> dict[str, int]:
    """Walk *data_path* and convert every .npy file.

    Returns a dict of status counts.
    """
    data_path = data_path or DATA_PATH

    npy_files = sorted(data_path.rglob("*.npy"))
    if not npy_files:
        print(f"No .npy files found under {data_path}")
        return {}

    mode_label = "DRY-RUN" if dry_run else "CONVERTING"
    print(f"\n{mode_label}: {len(npy_files)} .npy files under {data_path}")
    print(f"Old dim: {_OLD_DIM} -> New dim: {_NEW_DIM}")
    print(f"Keeping {len(KEEP_COLS)} columns "
          f"(pose={len(_POSE_COLS)}, face={len(_FACE_COLS)}, "
          f"lh={len(_LH_COLS)}, rh={len(_RH_COLS)})\n")

    stats: dict[str, int] = {}

    for fpath in tqdm(npy_files, desc=mode_label):
        status = convert_file(fpath, dry_run=dry_run)
        stats[status] = stats.get(status, 0) + 1

    print(f"\nResults:")
    for status, count in sorted(stats.items()):
        print(f"  {status}: {count}")

    # Delete cached scaler files (they encode the old dimension)
    if not dry_run and stats.get("converted", 0) > 0:
        for scaler_path in data_path.glob("scaler_*.pkl"):
            scaler_path.unlink()
            print(f"Deleted stale scaler: {scaler_path.name}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .npy keypoint files from 1692-dim to 507-dim",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually overwrite files (default is dry-run)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=f"Override processed data directory (default: {DATA_PATH})",
    )
    args = parser.parse_args()
    run_conversion(args.data_path, dry_run=not args.apply)
