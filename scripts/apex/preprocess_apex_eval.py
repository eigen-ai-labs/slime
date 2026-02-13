#!/usr/bin/env python3
"""Convert APEX JSON files into two eval JSONL files for zero-shot inference.

Reads local JSON files (tasks_and_rubrics.json + world_descriptions.json) and
produces two JSONL datasets — one per agent design (vanilla, plan-then-act).

Usage:
    # Default paths
    python preprocess_apex_eval.py

    # Custom paths
    python preprocess_apex_eval.py \
        --tasks /path/to/tasks_and_rubrics.json \
        --worlds /path/to/world_descriptions.json \
        --output-dir /path/to/output/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AGENT_DESIGNS = ("vanilla", "plan_then_act")

DEFAULT_TASKS_PATH = "/root/apex_data/tasks_and_rubrics.json"
DEFAULT_WORLDS_PATH = "/root/apex_data/world_descriptions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess APEX-Agents data for zero-shot eval (vanilla + plan-then-act)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS_PATH,
        help="Path to tasks_and_rubrics.json",
    )
    parser.add_argument(
        "--worlds",
        type=str,
        default=DEFAULT_WORLDS_PATH,
        help="Path to world_descriptions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory for output JSONL files (default: same dir as this script)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge model to embed in metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and first record without writing files",
    )
    return parser.parse_args()


def load_json(path: str) -> list[dict]:
    """Load a JSON array from file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def build_world_lookup(worlds: list[dict]) -> dict[str, dict]:
    """Build world_id -> world dict lookup."""
    return {w["world_id"]: w for w in worlds}


def convert_task(
    task: dict,
    world_lookup: dict[str, dict],
    agent_design: str,
    judge_model: str,
) -> dict | None:
    """Convert a single APEX task into a SLIME eval JSONL record.

    Returns None if the task is missing required fields.
    """
    prompt = task.get("prompt")
    if not prompt:
        return None

    gold_response = task.get("gold_response", "")
    task_id = task.get("task_id", "")
    world_id = task.get("world_id", "")
    domain = task.get("domain", "")
    expected_output = task.get("expected_output", "message_in_console")

    # Build multi_rubrics from rubric list
    rubric_list = task.get("rubric", [])
    multi_rubrics = []
    if isinstance(rubric_list, list):
        for i, item in enumerate(rubric_list):
            if isinstance(item, dict):
                criteria = item.get("criteria", "")
            else:
                criteria = str(item)
            multi_rubrics.append({
                "name": f"criterion_{i}",
                "weight": 1.0,
                "criteria": criteria,
            })

    # Resolve world description
    world = world_lookup.get(world_id, {})
    world_description = world.get("world_description", "")

    record = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "label": gold_response,
        "metadata": {
            "task_id": task_id,
            "world_id": world_id,
            "domain": domain,
            "expected_output": expected_output,
            "agent_design": agent_design,
            "world_description": world_description,
            "multi_rubrics": multi_rubrics,
            "reference_answer": gold_response,
            "query": prompt,
            "judge_model": judge_model,
        },
    }
    return record


def main() -> None:
    args = parse_args()

    # Load source data
    print(f"Loading tasks from: {args.tasks}")
    tasks = load_json(args.tasks)
    print(f"  -> {len(tasks)} tasks")

    print(f"Loading worlds from: {args.worlds}")
    worlds = load_json(args.worlds)
    print(f"  -> {len(worlds)} worlds")

    world_lookup = build_world_lookup(worlds)

    # Collect stats
    output_types: dict[str, int] = {}
    domains: dict[str, int] = {}
    for t in tasks:
        eo = t.get("expected_output", "unknown")
        output_types[eo] = output_types.get(eo, 0) + 1
        d = t.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1

    print(f"\nExpected output distribution:")
    for k, v in sorted(output_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100 * v / len(tasks):.1f}%)")

    print(f"\nDomain distribution:")
    for k, v in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n--- Dry run: showing first converted record (vanilla) ---")
        if tasks:
            record = convert_task(tasks[0], world_lookup, "vanilla", args.judge_model)
            if record:
                print(json.dumps(record, indent=2, ensure_ascii=False)[:2000])
        print(f"\nDry run complete. Would write {len(tasks)} records x {len(AGENT_DESIGNS)} designs.")
        return

    # Write one JSONL per agent design
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for design in AGENT_DESIGNS:
        output_path = output_dir / f"apex_eval_{design}.jsonl"
        written = 0
        skipped = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for task in tasks:
                record = convert_task(task, world_lookup, design, args.judge_model)
                if record is None:
                    skipped += 1
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        print(f"\nWrote {written} records to {output_path}")
        if skipped:
            print(f"  Skipped {skipped} tasks (missing required fields)")

    print("\nDone.")


if __name__ == "__main__":
    main()
