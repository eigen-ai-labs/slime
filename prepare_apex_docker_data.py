#!/usr/bin/env python3
"""
Prepare APEX Docker task data for slime RL training.

Converts Docker world exports + training verifiers into slime JSONL format.
Each line contains the task prompt, metadata (task_id, world_id, docker_dir,
verifiers, domain), and rm_type="archipelago" so the reward model dispatcher
routes to the Archipelago grading reward.

Usage:
    python prepare_apex_docker_data.py \
        --docker-root /path/to/docker/exports \
        --training-root /path/to/training/tasks \
        --output apex_tasks.jsonl

Reuses scanning/loading functions from distill_apex_docker_v4.py.
"""

import argparse
import json
import os
import re
import sys
import zipfile


# ── Docker/training scanning functions (adapted from distill_apex_docker_v4.py) ─


def scan_docker_worlds(docker_root):
    """Scan docker root for world exports. Returns {world_id: docker_dir}."""
    world_dirs = {}
    if not os.path.isdir(docker_root):
        return world_dirs
    for entry in os.listdir(docker_root):
        entry_path = os.path.join(docker_root, entry)
        if not os.path.isdir(entry_path):
            continue
        m = re.search(r'\((world_[a-f0-9]+)\)', entry)
        if m:
            world_dirs[m.group(1)] = entry_path
    return world_dirs


def load_docker_tasks(docker_dir, world_id):
    """Extract task dicts from export.zip's task JSONs.

    Returns list of dicts with task_id, task_slug, world_id, world_name,
    prompt, domain, etc.
    """
    tasks = []
    zip_path = os.path.join(docker_dir, "export.zip")
    if not os.path.exists(zip_path):
        return tasks
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("tasks/") and name.endswith(".json"):
                data = json.loads(zf.read(name))
                extra = data.get("extra", {})
                tid = extra.get("task_id")
                slug = extra.get("task_slug")
                if not tid or not slug:
                    continue
                prompt_blocks = data.get("prompt", [])
                prompt_text = " ".join(
                    b.get("content", "")
                    for b in prompt_blocks
                    if b.get("type") == "text"
                )
                wname = extra.get("world_name", "")
                if "law" in wname.lower():
                    domain = "Law"
                elif "investment" in wname.lower():
                    domain = "Investment Banking"
                elif "consulting" in wname.lower() or "management" in wname.lower():
                    domain = "Management Consulting"
                else:
                    domain = "Unknown"
                tasks.append({
                    "task_id": tid,
                    "task_slug": slug,
                    "task_name": extra.get("task_name", slug),
                    "world_id": world_id,
                    "world_name": wname,
                    "domain": domain,
                    "prompt": prompt_text,
                })
    return tasks


def load_training_verifiers(training_root):
    """Load full task_verifiers from training task.json files.

    Returns {task_id: {"verifiers": [...], "domain": str, "prompt": str}}
    """
    verifier_map = {}
    if not os.path.isdir(training_root):
        return verifier_map
    for entry in os.listdir(training_root):
        m = re.search(r'\((task_[a-f0-9]+)\)', entry)
        if not m:
            continue
        tid = m.group(1)
        task_json = os.path.join(training_root, entry, "task.json")
        if not os.path.exists(task_json):
            continue
        try:
            with open(task_json) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        verifiers = data.get("task_verifiers", [])
        if not verifiers:
            continue

        prompt = ""
        for msg in data.get("task_prompt_messages", []):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        domain = ""
        cf = data.get("custom_fields", {})
        for v in cf.values():
            if isinstance(v, str) and v in (
                "Law", "Investment Banking", "Management Consulting",
                "Survey/Interview Analysis", "Operations/Process Improvement",
            ):
                domain = v
                break

        verifier_map[tid] = {
            "verifiers": verifiers,
            "domain": domain,
            "prompt": prompt,
        }
    return verifier_map


def main():
    parser = argparse.ArgumentParser(
        description="Prepare APEX Docker tasks for slime RL training"
    )
    parser.add_argument(
        "--docker-root",
        required=True,
        help="Root dir containing world Docker exports (folders with export.zip + image.tar)",
    )
    parser.add_argument(
        "--training-root",
        required=True,
        help="Root dir containing training task folders with task.json verifiers",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--world",
        help="Only include tasks for a specific world ID",
    )
    parser.add_argument(
        "--require-verifiers",
        action="store_true",
        default=True,
        help="Skip tasks without training verifiers (default: True)",
    )
    parser.add_argument(
        "--no-require-verifiers",
        action="store_false",
        dest="require_verifiers",
        help="Include tasks even without training verifiers",
    )
    args = parser.parse_args()

    # Scan Docker worlds
    print(f"Scanning Docker worlds in {args.docker_root}...")
    docker_worlds = scan_docker_worlds(args.docker_root)
    print(f"  Found {len(docker_worlds)} worlds")

    if not docker_worlds:
        print(f"ERROR: No world Docker exports found in {args.docker_root}")
        sys.exit(1)

    # Filter to specific world if requested
    if args.world:
        if args.world in docker_worlds:
            docker_worlds = {args.world: docker_worlds[args.world]}
        else:
            print(f"ERROR: World {args.world} not found")
            sys.exit(1)

    # Load tasks from Docker exports
    print("Loading tasks from Docker exports...")
    all_tasks = []
    for wid, wdir in sorted(docker_worlds.items()):
        tasks = load_docker_tasks(wdir, wid)
        all_tasks.extend(tasks)
        print(f"  {wid}: {len(tasks)} tasks")
    print(f"  Total: {len(all_tasks)} tasks")

    # Load verifiers from training folders
    print(f"Loading verifiers from {args.training_root}...")
    verifier_map = load_training_verifiers(args.training_root)
    print(f"  Found verifiers for {len(verifier_map)} tasks")

    # Build JSONL records
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    skipped_no_verifiers = 0
    skipped_no_prompt = 0

    with open(args.output, "w") as f:
        for task in all_tasks:
            tid = task["task_id"]
            docker_dir = docker_worlds[task["world_id"]]

            # Get verifiers (prefer training verifiers, they have full structure)
            tv = verifier_map.get(tid)
            verifiers = tv["verifiers"] if tv else []
            # Prefer training prompt if available (may be refined)
            prompt = (tv["prompt"] if tv and tv["prompt"] else task["prompt"]).strip()
            # Prefer training domain if available
            domain = (tv["domain"] if tv and tv["domain"] else task["domain"])

            if not prompt:
                skipped_no_prompt += 1
                continue

            if args.require_verifiers and not verifiers:
                skipped_no_verifiers += 1
                continue

            record = {
                "messages": [{"role": "user", "content": prompt}],
                "metadata": {
                    "task_id": tid,
                    "task_slug": task["task_slug"],
                    "world_id": task["world_id"],
                    "docker_dir": docker_dir,
                    "verifiers": verifiers,
                    "domain": domain,
                    "rm_type": "archipelago",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written} tasks to {args.output}")
    if skipped_no_verifiers:
        print(f"  Skipped {skipped_no_verifiers} tasks without verifiers")
    if skipped_no_prompt:
        print(f"  Skipped {skipped_no_prompt} tasks without prompts")


if __name__ == "__main__":
    main()
