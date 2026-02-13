"""APEX-Agents Binary Rubric Reward Function.

Implements strict binary (Met/Not Met) evaluation matching the APEX benchmark
grading methodology. Each rubric criterion is evaluated independently via an
LLM judge, and the final reward is the fraction of criteria met.

Registration:
    --custom-rm-path slime.rollout.rm_hub.apex_reward.compute_apex_reward
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_API_BASE = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 128

# Binary judge prompt — enforces strict Met / Not Met output
BINARY_JUDGE_PROMPT = """You are evaluating an agent's output against a reference answer.

**Task:** {query}

**Agent Output:** {response}

**Reference Output:** {reference_answer}

**Criterion:** {criterion}

Is this criterion met? Answer ONLY with a JSON object: {{"result": "Met"}} or {{"result": "Not Met"}}."""


def _parse_binary_result(response_text: str) -> float:
    """Parse a binary Met/Not Met response from the judge.

    Returns 1.0 for Met, 0.0 for Not Met or parse failure.
    """
    # Try JSON extraction first
    try:
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            result = data.get("result", "").strip().lower()
            if result == "met":
                return 1.0
            if result in ("not met", "not_met"):
                return 0.0
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # Fallback: look for plain text Met/Not Met
    text = response_text.strip().lower()
    if "not met" in text or "not_met" in text:
        return 0.0
    if "met" in text:
        return 1.0

    logger.warning("Could not parse binary result from judge response: %s", response_text[:100])
    return 0.0


async def _call_binary_judge(
    query: str,
    response: str,
    reference_answer: str,
    criterion: str,
    judge_model: str,
    api_base: str,
    api_key: str,
    timeout: float,
) -> float:
    """Call the LLM judge API for a single binary criterion.

    Returns 1.0 (Met) or 0.0 (Not Met).
    """
    prompt = BINARY_JUDGE_PROMPT.format(
        query=query,
        response=response,
        reference_answer=reference_answer,
        criterion=criterion,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": DEFAULT_MAX_TOKENS,
    }

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{api_base.rstrip('/')}/chat/completions"
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "APEX judge API error (status %d) for criterion '%s': %s",
                        resp.status,
                        criterion[:50],
                        error_text[:200],
                    )
                    return 0.0

                result = await resp.json()

        judge_response = result["choices"][0]["message"]["content"]
        logger.debug("APEX judge response for '%s': %s", criterion[:50], judge_response[:100])
        return _parse_binary_result(judge_response)

    except aiohttp.ClientError as e:
        logger.error("APEX judge network error: %s", str(e))
        return 0.0
    except KeyError as e:
        logger.error("APEX judge response parsing error: %s", str(e))
        return 0.0
    except TimeoutError:
        logger.error("APEX judge request timed out for criterion: %s", criterion[:50])
        return 0.0
    except Exception as e:
        logger.error("APEX judge unexpected error: %s", str(e))
        return 0.0


async def compute_apex_reward(
    args: Any,
    sample: Any,
    **kwargs: Any,
) -> float:
    """Compute APEX binary rubric reward for a sample.

    Evaluates each criterion in metadata["multi_rubrics"] with a binary
    Met/Not Met judge call, then returns the fraction of criteria met.

    This function follows the custom RM signature expected by async_rm():
        async def custom_rm(args, sample, **kwargs) -> float

    Args:
        args: Training arguments namespace (unused, kept for interface compatibility).
        sample: A Sample object with .response, .label, and .metadata attributes.

    Returns:
        Float between 0.0 and 1.0 — fraction of rubric criteria met.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    # Config cascade: metadata → env var → default
    judge_model = metadata.get("judge_model") or os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    api_base = metadata.get("judge_api_base") or os.environ.get("OPENAI_API_BASE", DEFAULT_JUDGE_API_BASE)
    api_key = metadata.get("judge_api_key") or os.environ.get("OPENAI_API_KEY")
    timeout = metadata.get("judge_timeout", DEFAULT_TIMEOUT)

    if not api_key:
        logger.warning("No API key provided for APEX judge, returning 0.0")
        return 0.0

    multi_rubrics = metadata.get("multi_rubrics", [])
    if not multi_rubrics:
        logger.warning("No multi_rubrics in metadata, returning 0.0")
        return 0.0

    response = sample.response or ""
    reference_answer = metadata.get("reference_answer", "")
    query = metadata.get("query") or sample.label or "[No query provided]"

    # If query is a list (chat messages), extract the user message
    if isinstance(query, list):
        for msg in query:
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = msg.get("content", "")
                break
        else:
            query = str(query)

    # Evaluate all criteria in parallel
    tasks = [
        _call_binary_judge(
            query=query,
            response=response,
            reference_answer=reference_answer,
            criterion=rubric.get("criteria", ""),
            judge_model=judge_model,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
        )
        for rubric in multi_rubrics
    ]

    scores = await asyncio.gather(*tasks, return_exceptions=True)

    # Compute fraction of criteria met
    met_count = 0
    total = len(multi_rubrics)

    for i, score in enumerate(scores):
        if isinstance(score, Exception):
            rubric_name = multi_rubrics[i].get("name", f"criterion_{i}")
            logger.warning("APEX criterion '%s' evaluation failed: %s", rubric_name, score)
            continue
        if score == 1.0:
            met_count += 1

    reward = met_count / total if total > 0 else 0.0

    logger.debug(
        "APEX reward: %d/%d criteria met = %.3f",
        met_count,
        total,
        reward,
    )

    return reward
