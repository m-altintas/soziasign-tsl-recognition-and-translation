"""
Data Preparation: Raw JSONL -> Processed Train/Valid Split

Reads the raw gloss-Turkish sentence pairs, applies Turkish-aware
lowercasing, wraps glosses in <gloss> tags, and splits into
train/valid sets.

If processed files already exist, the script skips regeneration
to preserve the exact split used in experiments. Use --force to
overwrite.

Usage:
    python src/prepare_data.py          # safe: skips if files exist
    python src/prepare_data.py --force  # regenerate from scratch
"""

import argparse
import json
import os
import random

# --- Paths (relative to project root) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "slr_gloss_tr_cleaned.jsonl")
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.jsonl")
VALID_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "valid.jsonl")

VALID_RATIO = 0.1
SEED = 42


def turkish_lower(text):
    """Turkish-aware lowercasing (handles İ->i and I->ı)."""
    if not text:
        return ""
    return text.replace("İ", "i").replace("I", "ı").lower()


def process_entry(entry):
    """Lowercase the gloss and wrap it in <gloss> tags."""
    gloss = turkish_lower(entry["input"]).strip()
    return {
        "input": f"<gloss> {gloss} </gloss>",
        "output": entry["output"]
    }


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare train/valid splits from raw data.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing processed files.")
    args = parser.parse_args()

    # Safety check: do not overwrite the split used in experiments
    if os.path.exists(TRAIN_PATH) and os.path.exists(VALID_PATH) and not args.force:
        print(f"[SKIP] Processed files already exist. Use --force to regenerate.")
        print(f"       {TRAIN_PATH}")
        print(f"       {VALID_PATH}")
        return

    # 1. Read raw data
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded {len(data)} samples from {RAW_PATH}")

    # 2. Process all entries
    processed = [process_entry(entry) for entry in data]

    # 3. Shuffle and split
    random.seed(SEED)
    random.shuffle(processed)

    split_idx = len(processed) - int(len(processed) * VALID_RATIO)
    train_data = processed[:split_idx]
    valid_data = processed[split_idx:]

    # 4. Write output
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    write_jsonl(TRAIN_PATH, train_data)
    write_jsonl(VALID_PATH, valid_data)

    print(f"[INFO] Train: {len(train_data)} samples -> {TRAIN_PATH}")
    print(f"[INFO] Valid: {len(valid_data)} samples -> {VALID_PATH}")


if __name__ == "__main__":
    main()
