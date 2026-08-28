#!/usr/bin/env python3
"""Split a dataset's task.json and batch.json into train/test partitions on disk.

This produces *physically separated* train/test files so that generate_adversarial_agent.py
can be pointed at the train files and have **no path** to test data — eliminating
any risk of accidental leakage at Stage 1 (skill mining / association rules),
Stage 2 (strategy-template initial population) or Stage 3 (GA fitness sampling).

The split is:

    rng = random.Random(split_seed)
    indices = list(range(N))
    rng.shuffle(indices)
    n_test = max(1, int(N * test_ratio))
    test_indices = sorted(indices[:n_test])
    train_indices = sorted(indices[n_test:])

This matches the formula previously used inside ``FitnessEvaluator.__init__``
so split_seed=42 / test_ratio=0.4 reproduces earlier experiments exactly.

Usage:
    python split_dataset.py \\
        --task-json Datasets/finance-agent-benchmark/task.json \\
        --batch-json results/finance-agent-benchmark_batch.json \\
        --test-ratio 0.4 \\
        --split-seed 42 \\
        -o Datasets/finance-agent-benchmark/split_42_40

Outputs (under -o output dir):
    task.train.json       — train split of task.json
    task.test.json        — test split of task.json
    batch.train.json      — train split of batch.json (index-aligned)
    batch.test.json       — test split of batch.json (index-aligned)
    split_metadata.json   — {seed, ratio, train_indices, test_indices, src_*}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def compute_split_indices(
    n: int, test_ratio: float, split_seed: int,
) -> tuple[list[int], list[int]]:
    """Reproduce the exact split previously used inside FitnessEvaluator."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    rng = random.Random(split_seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_test = max(1, int(n * test_ratio))
    test_indices = sorted(indices[:n_test])
    train_indices = sorted(indices[n_test:])
    return train_indices, test_indices


def assert_aligned_or_reorder(
    batch: list[dict], n_tasks: int,
) -> list[dict]:
    """Ensure batch is sorted by ``task_index`` matching task.json positions.

    Orchestrator output already preserves order, but we reorder defensively.
    """
    if len(batch) != n_tasks:
        raise ValueError(
            f"Length mismatch: batch has {len(batch)} entries but task.json has {n_tasks}"
        )

    # If task_index is present, sort by it; otherwise trust the natural order
    has_index = all("task_index" in e for e in batch)
    if has_index:
        ordered = sorted(batch, key=lambda e: e["task_index"])
        for i, e in enumerate(ordered):
            if e["task_index"] != i:
                raise ValueError(
                    f"task_index gap at position {i}: got {e['task_index']}"
                )
        return ordered
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physically split a dataset's task.json and batch.json into train/test partitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task-json", required=True,
        help="Path to the dataset's task.json (e.g. Datasets/finance-agent-benchmark/task.json)",
    )
    parser.add_argument(
        "--batch-json", required=True,
        help="Path to the orchestrator batch result JSON (e.g. results/finance-agent-benchmark_batch.json)",
    )
    parser.add_argument(
        "--test-ratio", type=float, required=True,
        help="Fraction of tasks to put in the test split (e.g. 0.4)",
    )
    parser.add_argument(
        "--split-seed", type=int, required=True,
        help="Random seed for the deterministic shuffle",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory where split files will be written (created if missing)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files in --output-dir",
    )
    args = parser.parse_args()

    task_path = Path(args.task_json)
    batch_path = Path(args.batch_json)
    out_dir = Path(args.output_dir)

    if not task_path.exists():
        sys.exit(f"ERROR: task.json not found: {task_path}")
    if not batch_path.exists():
        sys.exit(f"ERROR: batch.json not found: {batch_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check existing outputs
    out_files = [
        out_dir / "task.train.json",
        out_dir / "task.test.json",
        out_dir / "batch.train.json",
        out_dir / "batch.test.json",
        out_dir / "split_metadata.json",
    ]
    existing = [p for p in out_files if p.exists()]
    if existing and not args.force:
        sys.exit(
            "ERROR: output files already exist (use --force to overwrite):\n  "
            + "\n  ".join(str(p) for p in existing)
        )

    # Load
    tasks = load_json(task_path)
    batch = load_json(batch_path)
    n = len(tasks)

    print(f"Loaded {n} task entries from {task_path}", file=sys.stderr)
    print(f"Loaded {len(batch)} batch entries from {batch_path}", file=sys.stderr)

    # Align batch to task indices
    batch = assert_aligned_or_reorder(batch, n)
    print("Batch index alignment OK", file=sys.stderr)

    # Compute split
    train_idx, test_idx = compute_split_indices(n, args.test_ratio, args.split_seed)
    print(
        f"Split (seed={args.split_seed} ratio={args.test_ratio}): "
        f"train={len(train_idx)} test={len(test_idx)}",
        file=sys.stderr,
    )
    if set(train_idx) & set(test_idx):
        sys.exit("INTERNAL ERROR: train and test indices overlap")
    if set(train_idx) | set(test_idx) != set(range(n)):
        sys.exit("INTERNAL ERROR: split does not cover all indices")

    # Slice
    task_train = [tasks[i] for i in train_idx]
    task_test = [tasks[i] for i in test_idx]
    batch_train = [batch[i] for i in train_idx]
    batch_test = [batch[i] for i in test_idx]

    # Write
    write_json(out_dir / "task.train.json", task_train)
    write_json(out_dir / "task.test.json", task_test)
    write_json(out_dir / "batch.train.json", batch_train)
    write_json(out_dir / "batch.test.json", batch_test)

    metadata = {
        "split_seed": args.split_seed,
        "test_ratio": args.test_ratio,
        "n_total": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_indices": train_idx,
        "test_indices": test_idx,
        "src_task_json": str(task_path.resolve()),
        "src_batch_json": str(batch_path.resolve()),
        "split_formula": (
            "rng = random.Random(seed); indices = list(range(n)); "
            "rng.shuffle(indices); n_test = max(1, int(n * ratio)); "
            "test = sorted(indices[:n_test]); train = sorted(indices[n_test:])"
        ),
    }
    write_json(out_dir / "split_metadata.json", metadata)

    print("\nDone. Outputs written:", file=sys.stderr)
    for p in out_files:
        size = p.stat().st_size if p.exists() else 0
        print(f"  {p}  ({size} bytes)", file=sys.stderr)
    print(
        "\nNext steps:\n"
        f"  generate_adversarial_agent.py --shadow-dataset {out_dir / 'task.train.json'}\n"
        f"  evaluate.py       -d {out_dir / 'task.test.json'}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
