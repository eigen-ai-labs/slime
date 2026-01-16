"""
OpenRouter LLM Judge Reward Model

Uses OpenRouter API to evaluate math answers with LLM-as-a-Judge.
"""

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from slime.utils.types import Sample

# ==================== Configuration ====================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ==================== Prompts ====================
JUDGE_SYSTEM_PROMPT = """You are a math answer evaluator. Your task is to determine if a student's answer is correct.

Instructions:
1. Extract the final answer from the student's response (look for \\boxed{}, "answer is", or the last number)
2. Compare it with the expected answer (consider mathematical equivalence: 1/2 = 0.5 = 50%)
3. Return JSON: {"score": 1} if correct, {"score": 0} if wrong

Only output the JSON, nothing else."""

JUDGE_USER_TEMPLATE = """Problem: {prompt}

Expected Answer: {label}

Student Response: {response}

Evaluate and return JSON:"""

# ==================== Thread Pool ====================
_thread_pool = ThreadPoolExecutor(max_workers=10)


def _sync_call_openrouter(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    retry_count: int,
) -> str:
    """Synchronous OpenRouter API call with retry."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 429:
                wait_time = 2 ** (attempt + retry_count)
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    return '{"score": 0}'


async def call_openrouter(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 128,
    retry_count: int = 0,
) -> str:
    """Async wrapper for OpenRouter API call."""
    model = model or OPENROUTER_MODEL
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _thread_pool,
        _sync_call_openrouter,
        messages,
        model,
        temperature,
        max_tokens,
        retry_count,
    )


# ==================== Response Parsing ====================
def parse_judge_response(response: str) -> dict:
    """Parse LLM judge response to extract score."""
    # Try direct JSON parse
    try:
        result = json.loads(response.strip())
        if "score" in result:
            return {"score": 1 if result["score"] == 1 else 0}
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if "score" in result:
                return {"score": 1 if result["score"] == 1 else 0}
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", response)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            return {"score": 1 if result.get("score") == 1 else 0}
        except json.JSONDecodeError:
            pass

    # Fallback: look for keywords
    response_lower = response.lower()
    if "correct" in response_lower and "incorrect" not in response_lower:
        return {"score": 1}
    if "incorrect" in response_lower or "wrong" in response_lower:
        return {"score": 0}

    return {"score": 0}


# ==================== Stats ====================
_stats = {
    "total": 0,
    "success": 0,
    "error": 0,
    "total_time": 0.0,
}


# ==================== Main Reward Function ====================
async def llm_judge_rm(args, sample: Sample, **kwargs) -> dict:
    """
    LLM Judge reward function for slime.

    Args:
        args: Training arguments
        sample: Sample containing prompt, response, label

    Returns:
        dict with score and metrics for wandb logging
    """
    global _stats

    prompt = sample.prompt
    response = sample.response
    label = getattr(sample, "label", "") or ""

    _stats["total"] += 1
    start_time = time.time()

    try:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(prompt=prompt, label=label, response=response),
            },
        ]

        llm_response = await call_openrouter(messages)
        result = parse_judge_response(llm_response)
        score = result["score"]
        _stats["success"] += 1

    except Exception:
        score = 0
        _stats["error"] += 1

    elapsed = time.time() - start_time
    _stats["total_time"] += elapsed

    avg_time = _stats["total_time"] / max(_stats["total"], 1)
    success_rate = _stats["success"] / max(_stats["total"], 1)

    return {
        "score": score,
        "llm_judge/latency_ms": elapsed * 1000,
        "llm_judge/avg_latency_ms": avg_time * 1000,
        "llm_judge/success_rate": success_rate,
    }


async def batched_llm_judge_rm(args, samples: list[Sample], **kwargs) -> list[dict]:
    """Batched version with concurrency control."""
    semaphore = asyncio.Semaphore(10)

    async def limited_call(sample):
        async with semaphore:
            return await llm_judge_rm(args, sample, **kwargs)

    return await asyncio.gather(*[limited_call(s) for s in samples])
