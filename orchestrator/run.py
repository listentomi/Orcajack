#!/usr/bin/env python3
"""CLI entry point for the MAS Orchestrator.

Usage examples:
    # Single task
    python -m orchestrator.run --task "Analyze Netflix revenue trends"

    # Batch – run all tasks in a dataset (with cost estimate + confirmation)
    python -m orchestrator.run --dataset finance-agent-benchmark

    # Resume an interrupted batch run (auto-detected from existing output file)
    python -m orchestrator.run --dataset finance-agent-benchmark --yes

    # Interactive mode
    python -m orchestrator.run --interactive

    # Custom config
    python -m orchestrator.run --config orchestrator/config.yaml --task "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .cost import format_batch_estimate
from .orchestrator import Orchestrator
from .schemas import OrchestratorOutput


def _save_result(result: OrchestratorOutput, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = output_dir / f"{name}.json"
    fp.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return fp


def _print_usage_summary(orch: Orchestrator) -> None:
    """Print token usage and cost summary after execution."""
    usage = orch.usage
    if usage.num_calls == 0:
        return
    cost = orch.get_cost_estimate()
    print("\n" + "=" * 60, file=sys.stderr)
    print("  运行统计  /  Execution Summary", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  LLM 调用次数:    {usage.num_calls}", file=sys.stderr)
    print(f"  Prompt tokens:   {usage.total_prompt_tokens:,}", file=sys.stderr)
    print(f"  Completion tokens: {usage.total_completion_tokens:,}", file=sys.stderr)
    print(f"  Total tokens:    {usage.total_tokens:,}", file=sys.stderr)
    print(f"  实际费用 (Cost): ¥{cost.total_cost:.4f}", file=sys.stderr)
    print(f"    ├─ input:  ¥{cost.input_cost:.4f}", file=sys.stderr)
    print(f"    └─ output: ¥{cost.output_cost:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def _count_completed(output_path: Path) -> int:
    """Count how many tasks are already completed in the output file."""
    if not output_path.exists():
        return 0
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        return len(data)
    except (json.JSONDecodeError, TypeError):
        return 0


AVG_COMPLETION_TOKENS = 500


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS Orchestrator Agent")
    parser.add_argument("--task", type=str, help="Single task to orchestrate")
    parser.add_argument("--dataset", type=str, help="Dataset name to batch-process (e.g. finance-agent-benchmark)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation for batch runs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    orch = Orchestrator(config_path=args.config)
    output_dir = Path(args.output_dir) if args.output_dir else (Path(__file__).parent / "../results").resolve()

    if args.task:
        result = orch.run(args.task)
        fp = _save_result(result, output_dir, "single_task")
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        print(f"\nSaved to {fp}", file=sys.stderr)
        _print_usage_summary(orch)

    elif args.dataset:
        tasks = orch.load_dataset_tasks(args.dataset)
        num_tasks = len(tasks)
        batch_path = output_dir / f"{args.dataset}_batch.json"
        done_count = _count_completed(batch_path)

        remaining = num_tasks - done_count
        avg_prompt = orch.estimate_prompt_tokens()

        if done_count > 0:
            print(f"\n  发现已有进度: {done_count}/{num_tasks} 已完成，"
                  f"将续跑剩余 {remaining} 个任务", file=sys.stderr)
            print(f"  结果文件: {batch_path}\n", file=sys.stderr)

        print(format_batch_estimate(
            orch._model, remaining, avg_prompt, AVG_COMPLETION_TOKENS,
        ), file=sys.stderr)

        if not args.yes:
            try:
                prompt_text = "是否继续运行？(y/N) >>> " if done_count == 0 else "是否继续未完成的任务？(y/N) >>> "
                answer = input(f"\n{prompt_text}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n已取消。", file=sys.stderr)
                sys.exit(0)
            if answer not in ("y", "yes"):
                print("已取消。", file=sys.stderr)
                sys.exit(0)

        workers = args.workers or orch.max_workers
        mode = f"并发 {workers} workers" if workers > 1 else "串行"
        print(f"\n开始处理（共 {num_tasks} 任务，跳过 {done_count} 已完成，{mode}）...\n", file=sys.stderr)
        results = orch.run_dataset(args.dataset, output_path=batch_path, max_workers=workers)
        print(f"\nDone! {len(results)} tasks saved to {batch_path}", file=sys.stderr)
        _print_usage_summary(orch)

    elif args.interactive:
        print("MAS Orchestrator – Interactive Mode  (type 'quit' to exit)")
        print(f"Model: {orch._model}  |  Agents loaded: {len(orch._registry)}")
        print("-" * 60)
        idx = 0
        while True:
            try:
                task = input("\n[Task] >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not task or task.lower() in ("quit", "exit", "q"):
                break
            idx += 1
            result = orch.run(task)
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

            cost = orch.get_cost_estimate()
            last_usage = orch.usage.calls[-1]
            print(
                f"  [Token] prompt={last_usage.prompt_tokens:,} "
                f"completion={last_usage.completion_tokens:,}  |  "
                f"累计费用: ¥{cost.total_cost:.4f}",
                file=sys.stderr,
            )
            fp = _save_result(result, output_dir, f"interactive_{idx}")
            print(f"  -> Saved to {fp}", file=sys.stderr)

        _print_usage_summary(orch)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
