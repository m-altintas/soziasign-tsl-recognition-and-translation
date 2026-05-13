"""
Depth × width ablation sweep over the GRU architecture on AUTSL.

Runs 7 cells sequentially. Each cell is a full training run with a unique
--run-tag so results are self-identifying in the output directory tree.

Usage (from tsl-recognition/):
    python scripts/run_ablation.py

Logs for each cell land in models/recognition/_ablation_logs/<tag>.log.
After all cells finish, run scripts/aggregate_ablation.py to produce a
summary table.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CELLS = [
    # Depth sweep — hidden=512 fixed (large preset width), layers vary
    {"tag": "ablation-depth-3", "gru_layers": 3, "gru_hidden": 512},
    {"tag": "ablation-depth-4", "gru_layers": 4, "gru_hidden": 512},
    {"tag": "ablation-baseline-5x512", "gru_layers": 5, "gru_hidden": 512},
    {"tag": "ablation-depth-6", "gru_layers": 6, "gru_hidden": 512},
    # Width sweep — layers=5 fixed (large preset depth), hidden varies
    {"tag": "ablation-width-256", "gru_layers": 5, "gru_hidden": 256},
    {"tag": "ablation-width-384", "gru_layers": 5, "gru_hidden": 384},
    {"tag": "ablation-width-768", "gru_layers": 5, "gru_hidden": 768},
]

LOG_DIR = REPO_ROOT / "models" / "recognition" / "_ablation_logs"


def run_cell(cell: dict) -> bool:
    tag = cell["tag"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{tag}.log"

    cmd = [
        sys.executable, "-m", "tsl_recognition", "train",
        "--dataset", "autsl",
        "--model-size", "large",
        "--gru-layers", str(cell["gru_layers"]),
        "--gru-hidden", str(cell["gru_hidden"]),
        "--run-tag", tag,
        "--min-epochs", "150",
    ]

    print(f"\n{'=' * 60}")
    print(f"Starting cell: {tag}")
    print(f"  gru_layers={cell['gru_layers']}  gru_hidden={cell['gru_hidden']}")
    print(f"  log -> {log_path}")
    print(f"{'=' * 60}")

    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_fh.write(result.stdout)

    # Mirror the tail to stdout so the tmux session shows progress
    lines = result.stdout.splitlines()
    for line in lines[-20:]:
        print(line)

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n[{tag}] {status}")
    return ok


def main() -> None:
    results: list[tuple[str, bool]] = []
    for cell in CELLS:
        ok = run_cell(cell)
        results.append((cell["tag"], ok))

    print(f"\n{'=' * 60}")
    print("ABLATION SWEEP SUMMARY")
    print(f"{'=' * 60}")
    for tag, ok in results:
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {tag}")

    failed = [tag for tag, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} cell(s) failed. Check logs in {LOG_DIR}.")
        sys.exit(1)
    else:
        print("\nAll cells completed. Run scripts/aggregate_ablation.py for the table.")


if __name__ == "__main__":
    main()
