"""
Mercor grading reward function for slime RL training.

Uses archipelago's grading system (snapshot diff + LLM verifiers).
Falls back to a simple LLM judge if archipelago is not installed.

Usage:
    --custom-rm-path slime.rollout.mercor_grading.grade

Dataset metadata format:
    {
        "world_snapshot_path": "/path/to/initial_world.zip",
        "final_snapshot_path": "/path/to/final_snapshot.tar.gz",  # set by mercor_agent_rollout
        "rubric": [
            {"verifier_id": "ver_xxx", "criteria": "Agent created a report.pdf with analysis"}
        ]
    }
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tarfile
import zipfile
from typing import Any

logger = logging.getLogger(__name__)


def _tar_gz_to_zip_bytes(tar_gz_path: str) -> io.BytesIO:
    """Convert a tar.gz file to a zip BytesIO (what archipelago grading expects)."""
    buf = io.BytesIO()
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        zf.writestr(member.name, f.read())
    buf.seek(0)
    return buf


def _load_snapshot_bytes(path: str) -> io.BytesIO:
    """Load a snapshot file as BytesIO. Handles .zip and .tar.gz."""
    if path.endswith(".zip"):
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    elif path.endswith(".tar.gz") or path.endswith(".gz"):
        return _tar_gz_to_zip_bytes(path)
    else:
        with open(path, "rb") as f:
            return io.BytesIO(f.read())


def grade(sample: Any, item: Any) -> float:
    """Grade a Mercor agent trajectory using archipelago's grading system.

    This function follows the OpenAI grader format: grade(sample, item) -> float.
    It is called by slime's batched_async_rm via --custom-rm-path.

    Args:
        sample: Agent output dict with 'trajectories' list
        item: Reference data dict with 'prompt' and 'metadata'

    Returns:
        Score between 0.0 and 1.0
    """
    metadata = item.get("metadata", {})
    rubric = metadata.get("rubric", [])
    initial_path = metadata.get("world_snapshot_path")
    final_path = metadata.get("final_snapshot_path")

    if not rubric:
        logger.warning("No rubric in metadata, returning 0.0")
        return 0.0

    if not initial_path or not final_path:
        logger.warning("Missing snapshot paths, returning 0.0")
        return 0.0

    if not os.path.exists(final_path):
        logger.warning("Final snapshot not found: %s, returning 0.0", final_path)
        return 0.0

    try:
        return _grade_with_archipelago(sample, item, initial_path, final_path, rubric)
    except ImportError:
        logger.info("Archipelago grading not installed, falling back to simple LLM judge")
        return _grade_with_llm_judge(sample, item, rubric)


def _grade_with_archipelago(
    sample: Any,
    item: Any,
    initial_path: str,
    final_path: str,
    rubric: list[dict],
) -> float:
    """Grade using archipelago's full grading pipeline (snapshot diff + LLM verifiers)."""
    from runner.evals.models import EvalConfig
    from runner.main import main as grading_main
    from runner.models import (
        AgentTrajectoryOutput,
        GradingSettings,
        Verifier,
    )
    from runner.scoring_methods.models import ScoringConfig

    import uuid

    initial_bytes = _load_snapshot_bytes(initial_path)
    final_bytes = _load_snapshot_bytes(final_path)

    # Build trajectory from sample data
    messages = []
    if item.get("prompt"):
        for msg in item["prompt"]:
            messages.append(msg)
    for step in sample.get("trajectories", []):
        messages.append({"role": "assistant", "content": step.get("response", "")})

    trajectory = AgentTrajectoryOutput(
        messages=messages,
        status="completed",
        time_elapsed=0.0,
    )

    # Build verifiers from rubric
    verifiers = [
        Verifier(
            verifier_id=v.get("verifier_id", f"ver_{i}"),
            verifier_version=1,
            world_id=item["metadata"].get("world_id", ""),
            task_id=item["metadata"].get("task_id", ""),
            eval_config_id="ec_output_llm",
            verifier_values={
                "criteria": v["criteria"],
                "is_primary_objective": i == 0,
            },
            verifier_index=i,
            verifier_dependencies=None,
        )
        for i, v in enumerate(rubric)
    ]

    # Default eval and scoring configs (same as archipelago examples)
    eval_configs = [
        EvalConfig(
            eval_config_id="ec_output_llm",
            eval_defn_id="output_llm",
            eval_config_values={
                "model": os.environ.get("MERCOR_GRADING_MODEL", "anthropic/claude-sonnet-4-20250514"),
            },
        )
    ]

    scoring_config = ScoringConfig(
        scoring_defn_id="apex_v1_grade_score",
        scoring_config_values={},
    )

    grading_settings = GradingSettings()

    grading_run_id = f"gr_{uuid.uuid4().hex[:8]}"
    trajectory_id = f"traj_{uuid.uuid4().hex[:8]}"

    _, _, _, scoring_results = asyncio.run(
        grading_main(
            grading_run_id=grading_run_id,
            trajectory_id=trajectory_id,
            initial_snapshot_bytes=initial_bytes,
            final_snapshot_bytes=final_bytes,
            trajectory=trajectory,
            grading_settings=grading_settings,
            verifiers=verifiers,
            eval_configs=eval_configs,
            scoring_config=scoring_config,
        )
    )

    score = scoring_results.final_score
    logger.info("Archipelago grading score: %.3f", score)
    return score


def _grade_with_llm_judge(
    sample: Any,
    item: Any,
    rubric: list[dict],
) -> float:
    """Fallback: simple LLM judge grading based on trajectory only (no snapshot diff).

    Uses litellm to call an LLM judge that scores the agent's trajectory
    against the rubric criteria.
    """
    try:
        import litellm
    except ImportError:
        logger.warning("litellm not installed, returning 0.0")
        return 0.0

    trajectories = sample.get("trajectories", [])
    if not trajectories:
        return 0.0

    # Build a summary of what the agent did
    steps_summary = []
    for i, step in enumerate(trajectories):
        resp = step.get("response", "")[:500]
        tools = step.get("tool_calls") or []
        tool_names = [t.get("name", "?") for t in tools] if tools else []
        results = step.get("tool_results") or []
        errors = [r for r in results if r.get("is_error")]

        step_desc = f"Step {i + 1}: {resp}"
        if tool_names:
            step_desc += f"\n  Tools used: {', '.join(tool_names)}"
        if errors:
            step_desc += f"\n  Errors: {len(errors)}"
        steps_summary.append(step_desc)

    criteria_list = "\n".join(f"- {v['criteria']}" for v in rubric)

    judge_prompt = f"""You are evaluating an AI agent that was given a professional services task.

Task prompt: {item.get("prompt", [{}])[0].get("content", "") if isinstance(item.get("prompt"), list) else item.get("prompt", "")}

Agent trajectory:
{chr(10).join(steps_summary)}

Evaluation criteria:
{criteria_list}

For each criterion, determine if the agent's work satisfies it.
Then provide an overall score from 0.0 to 1.0 where:
- 0.0 = none of the criteria met
- 0.5 = some criteria partially met
- 1.0 = all criteria fully met

Respond with ONLY a JSON object: {{"score": <float>, "rationale": "<brief explanation>"}}"""

    try:
        model = os.environ.get("MERCOR_GRADING_MODEL", "openai/gpt-4o-mini")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        score = float(result.get("score", 0.0))
        logger.info("LLM judge score: %.3f (%s)", score, result.get("rationale", ""))
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error("LLM judge grading failed: %s", e)
        return 0.0
