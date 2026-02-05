"""
LLM-as-a-Judge Reward Model for RL training.

Implements reward computation using an LLM to evaluate response quality:
- Supports OpenAI-compatible APIs (OpenAI, Anthropic, vLLM, etc.)
- Flexible scoring criteria via prompt templates (rubrics)
- Binary or scaled reward output
- Multi-rubric evaluation with weighted averaging

Example metadata format (single rubrics):
{
    "rm_type": "llm_judge",
    "rubrics": "Rate 0-1 for accuracy and clarity",  # Evaluation criteria
    "judge_model": "gpt-4o-mini",  # Optional, defaults to gpt-4o-mini
    "reference_answer": "Optional reference for comparison",
    "judge_api_base": "https://api.openai.com/v1"  # Optional
}

Example metadata format (multi-rubrics):
{
    "rm_type": "llm_judge",
    "judge_model": "gpt-4o-mini",
    "multi_rubrics": [
        {"name": "tool_usage", "weight": 0.3, "criteria": "Correct tool selection and usage"},
        {"name": "task_completion", "weight": 0.4, "criteria": "Successfully completed the task"},
        {"name": "reasoning", "weight": 0.2, "criteria": "Clear and logical reasoning"},
        {"name": "efficiency", "weight": 0.1, "criteria": "Minimal steps to complete"},
    ],
}

Alternative field names supported:
- rubrics OR judge_criteria: Evaluation criteria (rubrics preferred)
- judge_prompt: Custom prompt template (overrides default)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, TypedDict

import aiohttp


class RubricConfig(TypedDict, total=False):
    """Configuration for a single evaluation rubric."""

    name: str  # Rubric identifier
    weight: float  # Weight for averaging (0.0-1.0)
    criteria: str  # Evaluation criteria text


logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"  # Use a stable, working model
DEFAULT_JUDGE_API_BASE = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_TOKENS = 256

# Default judge prompt template
DEFAULT_JUDGE_PROMPT = """You are an expert evaluator. Rate the following response on a scale from 0.0 to 1.0.

**User Query:**
{query}

**Response to Evaluate:**
{response}

{reference_section}

**Evaluation Criteria:**
{criteria}

**Instructions:**
- Provide a single score between 0.0 (poor) and 1.0 (excellent)
- Consider all criteria listed above
- Output ONLY a JSON object with "score" and "reason" keys
- Example: {{"score": 0.85, "reason": "Clear and accurate response"}}

**Your Evaluation:**"""


def _format_criteria(criteria: list[str] | str | None) -> str:
    """Format evaluation criteria for the prompt."""
    if criteria is None:
        return "- Overall quality and helpfulness"
    if isinstance(criteria, str):
        return f"- {criteria}"
    return "\n".join(f"- {c}" for c in criteria)


def _format_reference_section(reference: str | None) -> str:
    """Format reference answer section if provided."""
    if reference:
        return f"""**Reference Answer (for comparison):**
{reference}
"""
    return ""


def _extract_score_from_response(response_text: str) -> float | None:
    """Extract numerical score from LLM response.

    Handles various response formats:
    - JSON: {"score": 0.85, "reason": "..."}
    - Plain number: 0.85
    - Text with number: "Score: 0.85"
    """
    # Try JSON parsing first
    try:
        # Find JSON object in response
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "score" in data:
                score = float(data["score"])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Try extracting plain number
    number_patterns = [
        r"(?:score|rating|evaluation)[\s:]*([0-9]*\.?[0-9]+)",  # Score: 0.85
        r"^([0-9]*\.?[0-9]+)$",  # Plain number
        r"([0-9]*\.?[0-9]+)\s*(?:/\s*1(?:\.0)?)?",  # 0.85 or 0.85/1
    ]

    for pattern in number_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                score = float(match.group(1))
                # If score is in 0-10 range, normalize to 0-1
                if score > 1.0:
                    score = score / 10.0
                return max(0.0, min(1.0, score))
            except ValueError:
                continue

    return None


async def compute_llm_judge_reward(
    response: str,
    label: Any,  # Can be used as query or ignored
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> float:
    """
    Compute reward using an LLM as a judge.

    Workflow:
    1. Extract configuration from metadata
    2. Format the judge prompt with query and response
    3. Call the judge LLM API
    4. Parse and return the score

    Args:
        response: The LLM-generated response to evaluate
        label: Optional query/prompt that generated the response
        metadata: Dictionary containing:
            - rubrics: Evaluation criteria/rubrics (preferred field)
            - judge_model: Model to use as judge (default: gpt-4o-mini)
            - judge_prompt: Custom prompt template (optional)
            - judge_criteria: Legacy alias for rubrics
            - reference_answer: Optional reference for comparison
            - judge_api_base: API base URL (default: OpenAI)
            - judge_api_key: API key (defaults to env var)
            - query: The original user query (if not in label)

    Returns:
        Float score between 0.0 and 1.0
    """
    if metadata is None:
        metadata = {}

    judge_model = metadata.get("judge_model") or os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    api_base = metadata.get("judge_api_base") or os.environ.get("OPENAI_API_BASE", DEFAULT_JUDGE_API_BASE)
    api_key = metadata.get("judge_api_key") or os.environ.get("OPENAI_API_KEY")
    request_timeout = timeout or metadata.get("judge_timeout", DEFAULT_TIMEOUT)

    if not api_key:
        logger.warning("No API key provided for LLM judge, returning 0.0")
        return 0.0

    # Get the query (original user input)
    query = metadata.get("query") or label or "[No query provided]"

    # Format the prompt
    custom_prompt = metadata.get("judge_prompt")
    # Support both "rubrics" (preferred) and "judge_criteria" (legacy)
    rubrics = metadata.get("rubrics") or metadata.get("judge_criteria")
    criteria = _format_criteria(rubrics)
    reference = _format_reference_section(metadata.get("reference_answer"))

    if custom_prompt:
        # Use custom prompt with basic variable substitution
        prompt = custom_prompt.format(
            query=query,
            response=response,
            reference=reference,
            criteria=criteria,
        )
    else:
        prompt = DEFAULT_JUDGE_PROMPT.format(
            query=query,
            response=response,
            reference_section=reference,
            criteria=criteria,
        )

    # Prepare API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": DEFAULT_MAX_TOKENS,  # Use max_completion_tokens for newer OpenAI models
        # Note: temperature omitted for compatibility with newer OpenAI models (o1, gpt-5-mini) that don't support it
    }

    # Make API call
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{api_base.rstrip('/')}/chat/completions"
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request_timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "LLM judge API error (status %d): %s",
                        resp.status,
                        error_text[:200],
                    )
                    return 0.0

                result = await resp.json()

        # Extract response content
        judge_response = result["choices"][0]["message"]["content"]
        logger.debug("Judge response: %s", judge_response[:200])

        # Parse score
        score = _extract_score_from_response(judge_response)
        if score is None:
            logger.warning(
                "Could not extract score from judge response: %s",
                judge_response[:100],
            )
            return 0.0

        logger.debug("LLM judge score: %.3f", score)
        return score

    except aiohttp.ClientError as e:
        logger.error("LLM judge network error: %s", str(e))
        return 0.0
    except KeyError as e:
        logger.error("LLM judge response parsing error: %s", str(e))
        return 0.0
    except TimeoutError:
        logger.error("LLM judge request timed out")
        return 0.0
    except Exception as e:
        logger.error("LLM judge unexpected error: %s", str(e))
        return 0.0


async def compute_multi_rubric_reward(
    response: str,
    label: Any,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> float | dict[str, float]:
    """
    Compute reward using multiple rubrics with weighted averaging.

    This function evaluates the response against multiple criteria in parallel
    and returns a weighted average of the scores.

    Args:
        response: The LLM-generated response to evaluate
        label: Optional query/prompt that generated the response
        metadata: Dictionary containing:
            - multi_rubrics: List of RubricConfig dicts with name, weight, criteria
            - judge_model: Model to use as judge (default: gpt-4o-mini)
            - judge_api_base: API base URL (default: OpenAI)
            - judge_api_key: API key (defaults to env var)
            - query: The original user query (if not in label)
            - return_breakdown: If True, return dict with per-rubric scores

    Returns:
        Float score (weighted average) or dict with breakdown if return_breakdown=True

    Example metadata:
        {
            "multi_rubrics": [
                {"name": "accuracy", "weight": 0.5, "criteria": "Factual accuracy"},
                {"name": "clarity", "weight": 0.3, "criteria": "Clear explanation"},
                {"name": "completeness", "weight": 0.2, "criteria": "Complete answer"},
            ],
            "return_breakdown": True,
        }
    """
    if metadata is None:
        metadata = {}

    multi_rubrics: list[RubricConfig] = metadata.get("multi_rubrics", [])

    # If no multi_rubrics, fall back to single rubrics evaluation
    if not multi_rubrics:
        return await compute_llm_judge_reward(response, label, metadata, timeout)

    # Validate rubrics
    total_weight = sum(r.get("weight", 1.0) for r in multi_rubrics)
    if total_weight <= 0:
        logger.warning("Total rubric weight is 0, returning 0.0")
        return 0.0

    # Create tasks for parallel evaluation
    tasks = []
    rubric_names = []

    for rubric in multi_rubrics:
        rubric_name = rubric.get("name", "unnamed")
        rubric_criteria = rubric.get("criteria", "")

        # Create metadata for this specific rubric
        rubric_metadata = {
            **metadata,
            "rubrics": rubric_criteria,
            # Remove multi_rubrics to avoid recursion
            "multi_rubrics": None,
        }

        tasks.append(compute_llm_judge_reward(response, label, rubric_metadata, timeout))
        rubric_names.append(rubric_name)

    # Execute all evaluations in parallel
    try:
        scores = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error("Multi-rubric evaluation failed: %s", e)
        return 0.0

    # Process results and compute weighted average
    weighted_sum = 0.0
    valid_weight = 0.0
    score_breakdown = {}

    for i, (rubric, score) in enumerate(zip(multi_rubrics, scores, strict=False)):
        rubric_name = rubric.get("name", f"rubric_{i}")
        weight = rubric.get("weight", 1.0)

        if isinstance(score, Exception):
            logger.warning("Rubric '%s' evaluation failed: %s", rubric_name, score)
            score_breakdown[rubric_name] = 0.0
            continue

        if not isinstance(score, (int, float)):
            logger.warning("Rubric '%s' returned non-numeric score: %s", rubric_name, score)
            score_breakdown[rubric_name] = 0.0
            continue

        score_breakdown[rubric_name] = float(score)
        weighted_sum += float(score) * weight
        valid_weight += weight

    # Compute final weighted average
    if valid_weight > 0:
        final_score = weighted_sum / valid_weight
    else:
        final_score = 0.0

    logger.debug(
        "Multi-rubric scores: %s, weighted average: %.3f",
        score_breakdown,
        final_score,
    )

    # Return breakdown if requested
    if metadata.get("return_breakdown", False):
        return {
            "score": final_score,
            "breakdown": score_breakdown,
            "weights": {r.get("name", f"rubric_{i}"): r.get("weight", 1.0) for i, r in enumerate(multi_rubrics)},
        }

    return final_score


async def compute_agent_trajectory_reward(
    samples: list[Any],
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> float:
    """
    Compute reward for an agent trajectory (list of samples).

    This function evaluates the entire trajectory and assigns reward
    only to the final sample. Designed for multi-step agent workflows.

    Args:
        samples: List of Sample objects representing the trajectory
        metadata: Evaluation configuration (same as compute_multi_rubric_reward)
        timeout: Request timeout

    Returns:
        Float score for the trajectory
    """
    if not samples:
        return 0.0

    # Get the final sample's response (complete trajectory)
    final_sample = samples[-1]
    response = getattr(final_sample, "response", str(final_sample))

    # Get query from first sample or metadata
    query = None
    if hasattr(samples[0], "prompt"):
        prompt = samples[0].prompt
        if isinstance(prompt, list):
            # Chat format - find user message
            for msg in prompt:
                if msg.get("role") == "user":
                    query = msg.get("content")
                    break
        else:
            query = prompt

    # Build metadata with query
    eval_metadata = {**(metadata or {})}
    if query and "query" not in eval_metadata:
        eval_metadata["query"] = query

    # Add trajectory context to evaluation
    trajectory_context = []
    for i, sample in enumerate(samples):
        step_info = {
            "step": i + 1,
            "has_tool_calls": sample.metadata.get("has_tool_calls", False) if hasattr(sample, "metadata") else False,
        }
        trajectory_context.append(step_info)

    eval_metadata["trajectory_steps"] = len(samples)
    eval_metadata["trajectory_context"] = trajectory_context

    # Use multi-rubric if configured, otherwise single rubric
    if eval_metadata.get("multi_rubrics"):
        return await compute_multi_rubric_reward(response, query, eval_metadata, timeout)
    else:
        return await compute_llm_judge_reward(response, query, eval_metadata, timeout)


# Aliases for consistency
llm_judge_reward = compute_llm_judge_reward
multi_rubric_reward = compute_multi_rubric_reward
agent_trajectory_reward = compute_agent_trajectory_reward
