"""
Archipelago grading reward model for slime RL training.

Wraps the Archipelago grading CLI as a slime-compatible reward function.
Used via: --custom-rm-path slime.rollout.rm_hub.archipelago.compute_archipelago_reward

The function reads snapshot paths and verifiers from sample.metadata,
builds grading config files, runs the grading CLI, and returns a float
score in [0, 1].

Environment variables:
    APEX_JUDGE_MODEL    - LLM judge model (default: openai/google/gemini-3-flash-preview)
    APEX_JUDGE_API_BASE - API base URL for judge (default: https://api.gmi-serving.com/v1)
    APEX_JUDGE_API_KEY  - API key for judge
    APEX_GRADING_DIR    - Path to Archipelago grading runner directory
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    return str(obj)


def _build_grading_inputs(task_id, world_id, verifiers, trajectory_messages,
                          final_answer, elapsed, status, tmpdir,
                          judge_model, judge_extra_args=None):
    """Create all JSON config files needed by the Archipelago grading CLI.

    Replicates build_grading_inputs() from distill_apex_docker_v4.py.

    Returns dict of file paths keyed by config name.
    """
    paths = {}

    # grading_settings.json
    grading_settings = {
        "llm_judge_model": judge_model,
        "llm_judge_extra_args": judge_extra_args,
    }
    paths["grading_settings"] = os.path.join(tmpdir, "grading_settings.json")
    with open(paths["grading_settings"], "w") as f:
        json.dump(grading_settings, f, indent=2)

    # verifiers.json
    verifier_configs = []
    unique_config_ids = set()
    for i, tv in enumerate(verifiers):
        config_id = tv.get("config_id", "output_llm")
        eval_config_id = f"ec_{config_id}" if config_id != "output_llm" else "ec_output_llm"
        unique_config_ids.add((config_id, eval_config_id))

        verifier_configs.append({
            "verifier_id": tv["verifier_id"],
            "verifier_version": tv.get("verifier_version", 1),
            "world_id": world_id,
            "task_id": task_id,
            "eval_config_id": eval_config_id,
            "verifier_values": tv.get("config_input", {}),
            "verifier_index": tv.get("verifier_index", i),
            "verifier_dependencies": tv.get("verifier_dependencies") or None,
        })
    paths["verifiers"] = os.path.join(tmpdir, "verifiers.json")
    with open(paths["verifiers"], "w") as f:
        json.dump(verifier_configs, f, indent=2)

    # eval_configs.json
    eval_configs = []
    for config_id, eval_config_id in sorted(unique_config_ids):
        eval_configs.append({
            "eval_config_id": eval_config_id,
            "eval_config_name": f"Output LLM Verifier ({config_id})",
            "eval_defn_id": "output_llm",
            "eval_config_values": {},
        })
    paths["eval_configs"] = os.path.join(tmpdir, "eval_configs.json")
    with open(paths["eval_configs"], "w") as f:
        json.dump(eval_configs, f, indent=2)

    # scoring_config.json
    scoring_config = {
        "scoring_config_id": "sc_default",
        "scoring_config_name": "Default Scoring",
        "scoring_defn_id": "apex_v1_grade_score",
        "scoring_config_values": {},
    }
    paths["scoring_config"] = os.path.join(tmpdir, "scoring_config.json")
    with open(paths["scoring_config"], "w") as f:
        json.dump(scoring_config, f, indent=2)

    # trajectory.json — build from agent messages
    clean_messages = []
    if isinstance(trajectory_messages, list):
        for msg in trajectory_messages:
            if isinstance(msg, dict):
                clean_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
                clean_messages.append(clean_msg)

    trajectory = {
        "messages": clean_messages,
        "output": {"final_answer": final_answer},
        "status": status,
        "time_elapsed": elapsed,
    }
    paths["trajectory"] = os.path.join(tmpdir, "trajectory.json")
    with open(paths["trajectory"], "w") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False, default=_json_default)

    return paths


def _run_grading_cli(task_id, initial_snapshot, final_snapshot, grading_inputs,
                     grading_dir, rollout_idx=0):
    """Run the Archipelago grading CLI and parse results.

    Returns (final_score, grading_output_dict).
    """
    output_path = os.path.join(os.path.dirname(grading_inputs["trajectory"]), "results.json")

    uv_bin = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    cmd = [
        uv_bin, "run", "python", "-m", "runner.main",
        "--grading-run-id", f"{task_id}_r{rollout_idx}",
        "--trajectory-id", f"traj_{task_id}",
        "--initial-snapshot", initial_snapshot,
        "--final-snapshot", final_snapshot,
        "--trajectory", grading_inputs["trajectory"],
        "--grading-settings", grading_inputs["grading_settings"],
        "--verifiers", grading_inputs["verifiers"],
        "--eval-configs", grading_inputs["eval_configs"],
        "--scoring-config", grading_inputs["scoring_config"],
        "--output", output_path,
    ]

    result = subprocess.run(
        cmd, cwd=grading_dir,
        capture_output=True, text=True, timeout=3600,
    )

    if result.returncode != 0:
        logger.error("Grading CLI failed for %s: exit=%d stderr=%s",
                      task_id, result.returncode, result.stderr[:500])
        return 0.0, {"error": f"CLI exit {result.returncode}: {result.stderr[:500]}"}

    with open(output_path) as f:
        grading_output = json.load(f)

    scoring_results = grading_output.get("scoring_results", {})
    final_score = scoring_results.get("final_score", 0.0)

    return final_score, grading_output


async def compute_archipelago_reward(
    args: Any,
    samples: Sample | list[Sample],
    **kwargs,
) -> float | list[float]:
    """Compute reward using Archipelago grading.

    Invoked by slime's reward model dispatcher via:
        --custom-rm-path slime.rollout.rm_hub.archipelago.compute_archipelago_reward

    Accepts both single Sample (from async_rm) and list[Sample] (from batched_async_rm).

    Reads from sample.metadata:
        - initial_snapshot: path to initial state ZIP
        - final_snapshot: path to final state ZIP
        - verifiers: list of verifier dicts (from training task.json)
        - task_id, world_id: identifiers

    Returns:
        float score in [0, 1] (single) or list[float] (batch)
    """
    # Handle batch mode: batched_async_rm passes list[Sample]
    if isinstance(samples, list):
        tasks = [_compute_single_reward(args, s, **kwargs) for s in samples]
        return await asyncio.gather(*tasks)

    # Single sample mode: async_rm passes a single Sample
    return await _compute_single_reward(args, samples, **kwargs)


async def _compute_single_reward(
    args: Any,
    sample: Sample,
    **kwargs,
) -> float:
    """Compute reward for a single sample."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    initial_snapshot = metadata.get("initial_snapshot")
    final_snapshot = metadata.get("final_snapshot")
    verifiers = metadata.get("verifiers", [])
    task_id = metadata.get("task_id", "unknown")
    world_id = metadata.get("world_id", "unknown")

    if not initial_snapshot or not final_snapshot:
        logger.warning("Missing snapshot paths for task %s, returning 0.0", task_id)
        return 0.0

    if not os.path.exists(initial_snapshot) or not os.path.exists(final_snapshot):
        logger.warning("Snapshot files not found for task %s, returning 0.0", task_id)
        return 0.0

    if not verifiers:
        logger.warning("No verifiers for task %s, returning 0.0", task_id)
        return 0.0

    # Read grading config from env vars
    judge_model = os.environ.get("APEX_JUDGE_MODEL", "openai/google/gemini-3-flash-preview")
    judge_api_base = os.environ.get("APEX_JUDGE_API_BASE", "https://api.gmi-serving.com/v1")
    judge_api_key = os.environ.get("APEX_JUDGE_API_KEY", "")
    grading_dir = os.environ.get("APEX_GRADING_DIR", "/data/mingye_b200-1/archipelago/grading")

    judge_extra_args = {}
    if judge_api_base:
        judge_extra_args["api_base"] = judge_api_base
    if judge_api_key:
        judge_extra_args["api_key"] = judge_api_key
    if not judge_extra_args:
        judge_extra_args = None

    # Build trajectory from sample conversation history
    # sample.prompt is the full chat message list (including all tool calls/results)
    trajectory_messages = sample.prompt if isinstance(sample.prompt, list) else []
    final_answer = sample.response or ""
    elapsed = metadata.get("elapsed_seconds", 0.0)
    status = "completed" if sample.status == Sample.Status.COMPLETED else "error"

    try:
        tmpdir = tempfile.mkdtemp(prefix=f"apex_grade_{task_id}_")
        try:
            grading_inputs = _build_grading_inputs(
                task_id=task_id,
                world_id=world_id,
                verifiers=verifiers,
                trajectory_messages=trajectory_messages,
                final_answer=final_answer,
                elapsed=elapsed,
                status=status,
                tmpdir=tmpdir,
                judge_model=judge_model,
                judge_extra_args=judge_extra_args,
            )

            # Run grading CLI in a thread to avoid blocking the event loop
            final_score, grading_output = await asyncio.to_thread(
                _run_grading_cli,
                task_id,
                initial_snapshot,
                final_snapshot,
                grading_inputs,
                grading_dir,
            )

            logger.info("Archipelago reward for %s: %.3f", task_id, final_score)
            return float(final_score)

        finally:
            # Clean up grading temp dir
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        logger.error("Archipelago grading failed for %s: %s", task_id, e)
        return 0.0

    finally:
        # Clean up snapshot temp files
        for snap_path in (initial_snapshot, final_snapshot):
            if snap_path and os.path.exists(snap_path):
                try:
                    os.unlink(snap_path)
                except OSError:
                    pass
