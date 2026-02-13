#!/usr/bin/env python3
"""Convert HuggingFace mercor/apex-agents dataset into SLIME training JSONL.

Usage:
    # Dry run - inspect dataset structure without writing
    python preprocess_apex_data.py --dry-run

    # Full conversion
    python preprocess_apex_data.py

    # Custom output path
    python preprocess_apex_data.py --output /path/to/output.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess APEX-Agents data for SLIME training")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "apex_train.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mercor/apex-agents",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect dataset structure without writing output",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge model to embed in metadata",
    )
    return parser.parse_args()


def inspect_dataset(dataset) -> None:
    """Print dataset structure for debugging."""
    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")
    print(f"\nFirst example:")
    example = dataset[0]
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        elif isinstance(value, list) and len(value) > 5:
            print(f"  {key}: [{value[0]!r}, ... ({len(value)} items)]")
        else:
            print(f"  {key}: {value!r}")


def convert_example(example: dict, judge_model: str) -> dict | None:
    """Convert a single HF dataset example to SLIME JSONL format.

    Returns None if the example is missing required fields.
    """
    prompt = example.get("prompt")
    if not prompt:
        return None

    gold_output = example.get("gold_output", "")
    rubric = example.get("rubric", [])

    # Build multi_rubrics from rubric list
    multi_rubrics = []
    if isinstance(rubric, list):
        for i, criterion in enumerate(rubric):
            multi_rubrics.append({
                "name": f"criterion_{i}",
                "weight": 1.0,
                "criteria": str(criterion),
            })
    elif isinstance(rubric, str) and rubric:
        multi_rubrics.append({
            "name": "criterion_0",
            "weight": 1.0,
            "criteria": rubric,
        })

    # Extract metadata fields (try common field names)
    task_id = example.get("task_id", example.get("id", ""))
    job_category = example.get("job_category", example.get("job_type", example.get("domain", "")))
    world_id = example.get("world_id", example.get("world", ""))

    record = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "label": gold_output,
        "metadata": {
            "rm_type": "llm_judge",
            "judge_model": judge_model,
            "reference_answer": gold_output,
            "multi_rubrics": multi_rubrics,
            "task_id": str(task_id),
            "job_category": str(job_category),
            "world_id": str(world_id),
        },
    }
    return record


def main() -> None:
    args = parse_args()

    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, split=args.split)
    except Exception:
        # Some datasets have no explicit split
        print(f"Failed to load split '{args.split}', trying without split...")
        ds = load_dataset(args.dataset)
        available_splits = list(ds.keys())
        print(f"Available splits: {available_splits}")
        dataset = ds[available_splits[0]]

    inspect_dataset(dataset)

    if args.dry_run:
        print("\n--- Dry run: showing first 3 converted examples ---")
        for i, example in enumerate(dataset):
            if i >= 3:
                break
            record = convert_example(example, args.judge_model)
            if record:
                print(json.dumps(record, indent=2, ensure_ascii=False)[:1000])
                print("---")
        print(f"\nDry run complete. Would write {len(dataset)} examples to {args.output}")
        return

    # Full conversion
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            record = convert_example(example, args.judge_model)
            if record is None:
                skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone: wrote {written} examples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} examples (missing required fields)")


if __name__ == "__main__":
    main()
