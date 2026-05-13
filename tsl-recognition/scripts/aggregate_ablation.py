"""
Aggregate ablation sweep results into a markdown table.

Walks models/recognition/run_*_ablation-* directories, reads config.json
and scores.json from each, and prints a formatted table suitable for
copy-pasting into the paper.

Usage (from tsl-recognition/):
    python scripts/aggregate_ablation.py
    python scripts/aggregate_ablation.py --out ablation_results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models" / "recognition"


def load_run(run_dir: Path) -> dict | None:
    config_path = run_dir / "config.json"
    scores_path = run_dir / "scores.json"
    if not config_path.exists() or not scores_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)
    with open(scores_path) as f:
        scores = json.load(f)

    run_tag = config.get("run_tag", "")
    if not run_tag or "ablation" not in run_tag:
        return None

    return {
        "tag": run_tag,
        "run_dir": run_dir.name,
        "num_layers": config.get("resolved_num_layers") or config.get("gru_num_layers"),
        "hidden_size": config.get("resolved_hidden_size") or config.get("gru_hidden_size"),
        "total_params": config.get("total_params"),
        "top1": scores.get("test_accuracy"),
        "top5": scores.get("test_top5_accuracy"),
        "best_val": scores.get("best_val_accuracy"),
    }


def format_table(rows: list[dict]) -> str:
    def pct(v: float | None) -> str:
        return f"{v * 100:.2f}%" if v is not None else "—"

    def params(v: int | None) -> str:
        if v is None:
            return "—"
        return f"{v / 1_000_000:.1f}M"

    def mark(row: dict) -> str:
        return "*" if "baseline" in row["tag"] else " "

    lines = [
        "| Tag | Layers | Hidden | Params | Val top-1 | Test top-1 | Test top-5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tag']}{mark(row)} "
            f"| {row['num_layers'] or '—'} "
            f"| {row['hidden_size'] or '—'} "
            f"| {params(row['total_params'])} "
            f"| {pct(row['best_val'])} "
            f"| {pct(row['top1'])} "
            f"| {pct(row['top5'])} |"
        )
    lines.append("\n\\* baseline cell")
    return "\n".join(lines)


def sort_key(row: dict) -> tuple:
    tag = row["tag"]
    layers = row["num_layers"] or 0
    hidden = row["hidden_size"] or 0
    # depth cells first (share hidden=512), then width cells
    if "depth" in tag or "baseline" in tag:
        return (0, layers, hidden)
    return (1, layers, hidden)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation sweep results")
    parser.add_argument("--out", type=str, default=None, help="Write table to this file")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Path to the models/recognition directory",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    run_dirs = sorted(models_dir.glob("run_*_ablation-*"))

    if not run_dirs:
        print(f"No ablation run directories found under {models_dir}.")
        print("Run scripts/run_ablation.py first.")
        return

    rows = []
    for run_dir in run_dirs:
        row = load_run(run_dir)
        if row:
            rows.append(row)
        else:
            print(f"  Skipping {run_dir.name} (incomplete or not an ablation run)")

    if not rows:
        print("No valid ablation runs found.")
        return

    rows.sort(key=sort_key)
    table = format_table(rows)

    print(table)

    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nTable written to {args.out}")


if __name__ == "__main__":
    main()
