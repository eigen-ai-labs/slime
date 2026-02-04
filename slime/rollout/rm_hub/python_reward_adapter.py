"""
Python Reward Function Adapter for Agent RL (OpenAI Grader Format Compatible).

This module provides an adapter that converts Slime's Sample list format
to OpenAI grader format for custom Python reward functions.

The adapter supports two function signatures:

1. OpenAI grader format (recommended):
   def grade(sample: Any, item: Any) -> float

2. Traditional Slime signature (for backwards compatibility):
   async def reward_function(args, sample: Sample) -> float

OpenAI grader format reference:
https://platform.openai.com/docs/guides/rft-use-cases?runloop=grader
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import Namespace

    from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def samples_to_grader_input(samples: list[Sample]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert Slime Sample list to OpenAI grader format (sample, item).

    This function transforms the internal Sample representation used by Slime
    into OpenAI grader format that's compatible with OpenAI RFT graders.

    Args:
        samples: List of Sample objects from the agent rollout.
                Each Sample represents one step in the trajectory.

    Returns:
        A tuple of (sample, item) in OpenAI grader format:
        - sample: Dictionary containing agent outputs:
            - trajectories: List of step dictionaries with:
                - response: The agent's response text for this step
                - tool_calls: Parsed tool calls (if any)
                - tool_results: Results from tool execution (if any)
                - thinking: Content from <think> tags (if any)
        - item: Dictionary containing reference data:
            - prompt: The original prompt (chat format)
            - metadata: User-defined metadata from the dataset
    """
    trajectories: list[dict[str, Any]] = []

    for s in samples:
        metadata = s.metadata if isinstance(s.metadata, dict) else {}

        step = {
            "response": s.response,
            "tool_calls": metadata.get("tool_calls"),
            "tool_results": metadata.get("tool_results"),
            "thinking": metadata.get("thinking"),
        }
        trajectories.append(step)

    # Build sample (agent output) in OpenAI format
    sample: dict[str, Any] = {
        "trajectories": trajectories,
    }

    # Build item (reference data) from the first sample
    first = samples[0] if samples else None
    item: dict[str, Any] = {
        "prompt": first.prompt if first else [],
        "metadata": (first.metadata if isinstance(first.metadata, dict) else {}) if first else {},
    }

    return sample, item


def is_openai_grader_signature(func) -> bool:
    """Check if the function uses OpenAI grader signature: grade(sample, item).

    Args:
        func: The reward function to check.

    Returns:
        True if the function uses OpenAI grader signature, False otherwise.
    """
    try:
        # Check function name first
        if func.__name__ == "grade":
            return True

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check for OpenAI grader signature: (sample, item)
        if len(params) >= 2:
            first_param = params[0].lower()
            second_param = params[1].lower()

            if first_param == "sample" and second_param == "item":
                return True

        return False
    except Exception as e:
        logger.warning("Could not inspect function signature: %s", e)
        return False


async def call_reward_function(
    func,
    args: Namespace,
    samples: list[Sample],
    **kwargs,
) -> float | list[float]:
    """Call the reward function with the appropriate signature.

    This function automatically detects the signature of the reward function
    and calls it with the appropriate arguments.

    Supports:
    1. OpenAI grader format: def grade(sample, item) -> float
    2. Traditional Slime format: async def reward_function(args, sample) -> float

    Args:
        func: The reward function to call.
        args: Slime training arguments (for traditional signature).
        samples: List of Sample objects from the trajectory.
        **kwargs: Additional keyword arguments.

    Returns:
        Reward value(s) from the function.
    """
    if is_openai_grader_signature(func):
        # Use OpenAI grader format: grade(sample, item)
        sample, item = samples_to_grader_input(samples)

        # Check if function is async or sync
        if asyncio.iscoroutinefunction(func):
            return await func(sample, item)
        else:
            # OpenAI graders are typically sync functions
            return func(sample, item)
    else:
        # Use traditional Slime signature
        # For group/batch mode, pass all samples
        if len(samples) > 1 and _is_batch_function(func):
            return await func(args, samples, **kwargs)
        else:
            # Single sample mode
            return await func(args, samples[0] if samples else None, **kwargs)


def _is_batch_function(func) -> bool:
    """Check if the function expects a batch of samples.

    Batch functions have 'samples' (plural) as the second parameter,
    while single-sample functions have 'sample' (singular).
    """
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if len(params) >= 2:
            second_param = params[1].lower()
            return second_param == "samples"
        return False
    except Exception:
        return False


# Template functions for users to reference (OpenAI grader format)
# These are not used directly but serve as documentation


def template_task_completion(sample: Any, item: Any) -> float:
    """Template: Score based on task completion markers.

    OpenAI grader format compatible.

    Args:
        sample: Agent output containing trajectories
        item: Reference data with prompt and metadata

    Returns:
        1.0 if completed, 0.5 if tools used, 0.0 otherwise
    """
    trajectories = sample.get("trajectories", [])
    final_response = "".join(step["response"] for step in trajectories).lower()

    if "task completed" in final_response or "successfully" in final_response:
        return 1.0
    elif any(step.get("tool_calls") for step in trajectories):
        return 0.5
    return 0.0


def template_json_validation(sample: Any, item: Any) -> float:
    """Template: Validate JSON output format.

    OpenAI grader format compatible.

    Args:
        sample: Agent output containing trajectories
        item: Reference data with expected_keys in metadata

    Returns:
        Fraction of expected keys found in JSON output
    """
    import json

    trajectories = sample.get("trajectories", [])
    metadata = item.get("metadata", {})

    final_response = trajectories[-1]["response"] if trajectories else ""
    expected_keys = metadata.get("expected_keys", [])

    try:
        result = json.loads(final_response)
        if not expected_keys:
            return 1.0
        matched = sum(1 for k in expected_keys if k in result)
        return matched / len(expected_keys)
    except json.JSONDecodeError:
        return 0.0


def template_tool_usage(sample: Any, item: Any) -> float:
    """Template: Evaluate tool usage accuracy.

    OpenAI grader format compatible.

    Args:
        sample: Agent output containing trajectories
        item: Reference data with expected_tools in metadata

    Returns:
        Tool match score minus error penalty
    """
    trajectories = sample.get("trajectories", [])
    metadata = item.get("metadata", {})
    expected_tools = set(metadata.get("expected_tools", []))

    used_tools = set()
    tool_errors = 0

    for step in trajectories:
        for call in step.get("tool_calls") or []:
            used_tools.add(call.get("name"))
        for result in step.get("tool_results") or []:
            if result.get("is_error"):
                tool_errors += 1

    if expected_tools:
        tool_match = len(used_tools & expected_tools) / len(expected_tools)
    else:
        tool_match = 1.0 if used_tools else 0.5

    error_penalty = min(tool_errors * 0.1, 0.5)
    return max(0.0, tool_match - error_penalty)


def template_multi_criteria(sample: Any, item: Any) -> float:
    """Template: Multi-criteria weighted scoring.

    OpenAI grader format compatible.

    Args:
        sample: Agent output containing trajectories
        item: Reference data with weights and criteria in metadata

    Returns:
        Weighted sum of completion, efficiency, and accuracy scores
    """
    trajectories = sample.get("trajectories", [])
    metadata = item.get("metadata", {})
    weights = metadata.get("weights", {"completion": 0.5, "efficiency": 0.3, "accuracy": 0.2})

    final_response = "".join(step["response"] for step in trajectories).lower()

    scores = {}

    # Completion score
    completion_keywords = metadata.get("completion_keywords", ["done", "completed"])
    scores["completion"] = 1.0 if any(kw in final_response for kw in completion_keywords) else 0.0

    # Efficiency score (fewer steps is better)
    max_steps = metadata.get("max_steps", 10)
    scores["efficiency"] = max(0, 1 - len(trajectories) / max_steps)

    # Accuracy score (compare with expected answer)
    if "expected_answer" in metadata:
        scores["accuracy"] = 1.0 if metadata["expected_answer"].lower() in final_response else 0.0
    else:
        scores["accuracy"] = 0.5

    # Weighted average
    total = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights)
    return total
