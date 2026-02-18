"""
Prepare Mercor APEX-Agents dataset for slime RL training.

Downloads from HuggingFace mercor/apex-agents and converts to slime JSONL format.

Usage:
    uv run python -m slime.rollout.mercor_data_prep \
        --output mercor_train.jsonl \
        --cache-dir /data/mercor_cache

    # Filter by domain:
    uv run python -m slime.rollout.mercor_data_prep \
        --output mercor_ib.jsonl \
        --domain "Investment Banking"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from huggingface_hub import hf_hub_download

HF_DATASET = "mercor/apex-agents"


def prepare_dataset(
    output_path: str,
    cache_dir: str = "/tmp/mercor_cache",
    domain: str | None = None,
    max_tasks: int | None = None,
) -> None:
    """Convert HuggingFace mercor/apex-agents to slime JSONL."""
    os.makedirs(cache_dir, exist_ok=True)
    snapshots_dir = os.path.join(cache_dir, "world_snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    # Download task and world metadata
    print("Downloading tasks_and_rubrics.json...")
    tasks_path = hf_hub_download(HF_DATASET, "tasks_and_rubrics.json", repo_type="dataset")
    with open(tasks_path) as f:
        tasks = json.load(f)

    print("Downloading world_descriptions.json...")
    worlds_path = hf_hub_download(HF_DATASET, "world_descriptions.json", repo_type="dataset")
    with open(worlds_path) as f:
        worlds = {w["world_id"]: w for w in json.load(f)}

    # Filter by domain if specified
    if domain:
        tasks = [t for t in tasks if t.get("domain", "").lower() == domain.lower()]
        print(f"Filtered to {len(tasks)} tasks for domain '{domain}'")

    if max_tasks:
        tasks = tasks[:max_tasks]

    # Download world snapshots
    world_ids_needed = {t["world_id"] for t in tasks}
    print(f"Need {len(world_ids_needed)} world snapshots")

    for world_id in world_ids_needed:
        local_path = os.path.join(snapshots_dir, f"{world_id}.zip")
        if os.path.exists(local_path):
            print(f"  {world_id}: cached")
            continue

        print(f"  {world_id}: downloading...")
        hf_path = hf_hub_download(
            HF_DATASET,
            f"world_files_zipped/{world_id}.zip",
            repo_type="dataset",
        )
        shutil.copy(hf_path, local_path)

    # Generate JSONL
    print(f"Writing {len(tasks)} tasks to {output_path}")
    with open(output_path, "w") as f:
        for task in tasks:
            world_id = task["world_id"]
            world = worlds.get(world_id, {})

            entry = {
                "messages": [
                    {"role": "user", "content": task["prompt"]},
                ],
                "metadata": {
                    "world_id": world_id,
                    "task_id": task["task_id"],
                    "task_name": task.get("task_name", ""),
                    "domain": task.get("domain", ""),
                    "world_name": world.get("world_name", ""),
                    "world_snapshot_path": os.path.join(snapshots_dir, f"{world_id}.zip"),
                    "rubric": task.get("rubric", []),
                    "gold_response": task.get("gold_response", ""),
                    "expected_output": task.get("expected_output", ""),
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done! {len(tasks)} tasks written to {output_path}")
    print(f"World snapshots cached in {snapshots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Mercor APEX-Agents dataset for slime")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--cache-dir", type=str, default="/tmp/mercor_cache", help="Cache directory for world snapshots")
    parser.add_argument("--domain", type=str, default=None, help="Filter by domain (e.g. 'Investment Banking', 'Law', 'Management Consulting')")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max number of tasks")
    args = parser.parse_args()

    prepare_dataset(
        output_path=args.output,
        cache_dir=args.cache_dir,
        domain=args.domain,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
