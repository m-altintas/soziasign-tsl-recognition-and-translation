"""
Depth x width ablation sweep over the GRU architecture on AUTSL.

Launches all 7 cells in parallel so the full sweep uses the available GPU
memory instead of running one cell at a time.

Usage (from tsl-recognition/):
    python scripts/run_ablation.py

Each cell writes its own log to models/recognition/_ablation_logs/<tag>.log.
After all cells finish, run scripts/aggregate_ablation.py to produce a
summary table.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CELLS = [
    # Depth sweep -- hidden=512 fixed (large preset width), layers vary
    {"tag": "ablation-depth-3", "gru_layers": 3, "gru_hidden": 512},
    {"tag": "ablation-depth-4", "gru_layers": 4, "gru_hidden": 512},
    {"tag": "ablation-baseline-5x512", "gru_layers": 5, "gru_hidden": 512},
    {"tag": "ablation-depth-6", "gru_layers": 6, "gru_hidden": 512},
    # Width sweep -- layers=5 fixed (large preset depth), hidden varies
    {"tag": "ablation-width-256", "gru_layers": 5, "gru_hidden": 256},
    {"tag": "ablation-width-384", "gru_layers": 5, "gru_hidden": 384},
    {"tag": "ablation-width-768", "gru_layers": 5, "gru_hidden": 768},
]

LOG_DIR = REPO_ROOT / "models" / "recognition" / "_ablation_logs"


def launch_cell(cell: dict) -> tuple[str, subprocess.Popen, object]:
    tag = cell["tag"]
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

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    return tag, proc, log_fh


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Launching {len(CELLS)} cells in parallel")
    print(f"{'=' * 60}")

    running: list[tuple[str, subprocess.Popen, object]] = []
    for cell in CELLS:
        tag, proc, log_fh = launch_cell(cell)
        running.append((tag, proc, log_fh))
        print(
            f"  [started] {tag}  "
            f"layers={cell['gru_layers']}  hidden={cell['gru_hidden']}  "
            f"pid={proc.pid}  log={LOG_DIR / (tag + '.log')}"
        )

    print(f"\nAll cells running. Waiting for completion...\n")

    results: list[tuple[str, bool]] = []
    for tag, proc, log_fh in running:
        proc.wait()
        log_fh.close()
        ok = proc.returncode == 0
        status = "OK" if ok else f"FAILED (exit {proc.returncode})"
        print(f"  [{status}] {tag}")
        results.append((tag, ok))

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
