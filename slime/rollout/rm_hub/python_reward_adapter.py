"""
Python Reward Function Adapter for Agent RL.

This module provides an adapter that converts Slime's Sample list format
to a user-friendly format for custom Python reward functions.

The adapter supports two function signatures:

1. New simplified signature (recommended for Agent RL):
   async def reward_function(trajectories: list[dict], reference: dict) -> float

2. Traditional Slime signature (for backwards compatibility):
   async def reward_function(args, sample: Sample) -> float
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import Namespace

    from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def samples_to_reward_input(samples: list[Sample]) -> tuple[list[dict], dict]:
    """Convert Slime Sample list to reward function input format.

    This function transforms the internal Sample representation used by Slime
    into a user-friendly format that's easier to work with in custom reward functions.

    Args:
        samples: List of Sample objects from the agent rollout.
                Each Sample represents one step in the trajectory.

    Returns:
        A tuple of (trajectories, reference):
        - trajectories: List of step dictionaries containing:
            - response: The agent's response text for this step
            - tool_calls: Parsed tool calls (if any)
            - tool_results: Results from tool execution (if any)
            - thinking: Content from <think> tags (if any)
        - reference: Dictionary containing:
            - prompt: The original prompt (chat format)
            - metadata: User-defined metadata from the dataset
    """
    trajectories: list[dict[str, Any]] = []

    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

        step = {
            "response": sample.response,
            "tool_calls": metadata.get("tool_calls"),
            "tool_results": metadata.get("tool_results"),
            "thinking": metadata.get("thinking"),
        }
        trajectories.append(step)

    # Build reference from the first sample (all samples share the same prompt/metadata)
    first = samples[0] if samples else None
    reference = {
        "prompt": first.prompt if first else [],
        "metadata": (first.metadata if isinstance(first.metadata, dict) else {}) if first else {},
    }

    return trajectories, reference


def is_simplified_signature(func) -> bool:
    """Check if the function uses the new simplified signature.

    The simplified signature is:
        async def reward_function(trajectories: list[dict], reference: dict) -> float

    The traditional signature is:
        async def reward_function(args, sample: Sample) -> float

    Args:
        func: The reward function to check.

    Returns:
        True if the function uses the simplified signature, False otherwise.
    """
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check for simplified signature: first param is 'trajectories' or 'trajectory'
        # or second param is 'reference'
        if len(params) >= 2:
            first_param = params[0].lower()
            second_param = params[1].lower()

            # New signature indicators
            if first_param in ("trajectories", "trajectory", "steps"):
                return True
            if second_param == "reference":
                return True

        # Also check type hints if available
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                annotation_str = str(param.annotation)
                # Check for list[dict] or List[dict] type hints
                if "list[dict]" in annotation_str.lower() or "list" in annotation_str.lower():
                    if param_name.lower() in ("trajectories", "trajectory", "steps"):
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

    Args:
        func: The reward function to call.
        args: Slime training arguments (for traditional signature).
        samples: List of Sample objects from the trajectory.
        **kwargs: Additional keyword arguments.

    Returns:
        Reward value(s) from the function.
    """
    if is_simplified_signature(func):
        # Use new simplified signature
        trajectories, reference = samples_to_reward_input(samples)
        return await func(trajectories, reference)
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


# Template functions for users to reference
# These are not used directly but serve as documentation


async def template_task_completion(trajectories: list[dict], reference: dict) -> float:
    """Template: Score based on task completion markers.

    Args:
        trajectories: Agent execution steps
        reference: Original prompt + metadata

    Returns:
        1.0 if completed, 0.5 if tools used, 0.0 otherwise
    """
    final_response = "".join(step["response"] for step in trajectories).lower()

    if "task completed" in final_response or "successfully" in final_response:
        return 1.0
    elif any(step.get("tool_calls") for step in trajectories):
        return 0.5
    return 0.0


async def template_json_validation(trajectories: list[dict], reference: dict) -> float:
    """Template: Validate JSON output format.

    Args:
        trajectories: Agent execution steps
        reference: Contains expected_keys in metadata

    Returns:
        Fraction of expected keys found in JSON output
    """
    import json

    final_response = trajectories[-1]["response"] if trajectories else ""
    expected_keys = reference["metadata"].get("expected_keys", [])

    try:
        result = json.loads(final_response)
        if not expected_keys:
            return 1.0
        matched = sum(1 for k in expected_keys if k in result)
        return matched / len(expected_keys)
    except json.JSONDecodeError:
        return 0.0


async def template_tool_usage(trajectories: list[dict], reference: dict) -> float:
    """Template: Evaluate tool usage accuracy.

    Args:
        trajectories: Agent execution steps
        reference: Contains expected_tools in metadata

    Returns:
        Tool match score minus error penalty
    """
    expected_tools = set(reference["metadata"].get("expected_tools", []))

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


async def template_multi_criteria(trajectories: list[dict], reference: dict) -> float:
    """Template: Multi-criteria weighted scoring.

    Args:
        trajectories: Agent execution steps
        reference: Contains weights and criteria in metadata

    Returns:
        Weighted sum of completion, efficiency, and accuracy scores
    """
    metadata = reference["metadata"]
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
