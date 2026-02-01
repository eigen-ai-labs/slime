"""
LLM-as-a-Judge Reward Model for RL training.

Implements reward computation using an LLM to evaluate response quality:
- Supports OpenAI-compatible APIs (OpenAI, Anthropic, vLLM, etc.)
- Flexible scoring criteria via prompt templates (rubrics)
- Binary or scaled reward output

Example metadata format:
{
    "rm_type": "llm_judge",
    "rubrics": "Rate 0-1 for accuracy and clarity",  # Evaluation criteria
    "judge_model": "gpt-4o-mini",  # Optional, defaults to gpt-4o-mini
    "reference_answer": "Optional reference for comparison",
    "judge_api_base": "https://api.openai.com/v1"  # Optional
}

Alternative field names supported:
- rubrics OR judge_criteria: Evaluation criteria (rubrics preferred)
- judge_prompt: Custom prompt template (overrides default)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
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

    # Get configuration from metadata
    judge_model = metadata.get("judge_model", DEFAULT_JUDGE_MODEL)
    api_base = metadata.get("judge_api_base", DEFAULT_JUDGE_API_BASE)
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
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": 0.0,  # Deterministic for consistency
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


# Alias for consistency
llm_judge_reward = compute_llm_judge_reward
