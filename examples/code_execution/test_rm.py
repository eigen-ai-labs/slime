#!/usr/bin/env python3
"""
Test script for Code Execution Reward Model.

This script demonstrates how to use the code execution RM with sample data,
simulating the reward evaluation process that happens during RL training.

Usage:
    uv run python examples/code_execution/test_rm.py
"""

import asyncio
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Load modules directly to avoid aiohttp dependency in __init__.py
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the code execution modules
utils_module = load_module(
    "slime.rollout.rm_hub.code_execution_utils",
    project_root / "slime/rollout/rm_hub/code_execution_utils.py",
)
sandbox_module = load_module(
    "slime.rollout.rm_hub.code_execution_sandbox",
    project_root / "slime/rollout/rm_hub/code_execution_sandbox.py",
)
exec_module = load_module(
    "slime.rollout.rm_hub.code_execution",
    project_root / "slime/rollout/rm_hub/code_execution.py",
)

compute_code_execution_reward = exec_module.compute_code_execution_reward


@dataclass
class Sample:
    """Simplified Sample class for testing."""

    prompt: str | list[dict[str, str]] = ""
    response: str = ""
    label: str | None = None
    reward: float | None = None
    metadata: dict = field(default_factory=dict)


# Simulated model responses (in real training, these come from the policy model)
SIMULATED_RESPONSES = {
    0: """I'll solve this step by step.

```python
a = int(input())
b = int(input())
print(a + b)
```
""",
    1: """Here's my solution:

```python
n = int(input())
total = 0
for _ in range(n):
    x = int(input())
    total += x
print(total)
```
""",
    2: """```python
n = int(input())
if n % 2 == 0:
    print("YES")
else:
    print("NO")
```
""",
    3: """```python
s = input()
print(s[::-1])
```
""",
    4: """```python
n = int(input())
result = 1
for i in range(1, n + 1):
    result *= i
print(result)
```
""",
}

# Some intentionally wrong responses for testing
WRONG_RESPONSES = {
    0: """```python
a = int(input())
b = int(input())
print(a - b)  # Wrong: subtracting instead of adding
```
""",
    2: """```python
n = int(input())
print("YES")  # Always prints YES, wrong for odd numbers
```
""",
}


async def test_with_sample_data():
    """Test the RM with sample data from the JSONL file."""
    data_path = Path(__file__).parent / "sample_data.jsonl"

    print("=" * 60)
    print("Testing Code Execution RM with Sample Data")
    print("=" * 60)

    with open(data_path) as f:
        samples_data = [json.loads(line) for line in f]

    print(f"\nLoaded {len(samples_data)} samples from {data_path}\n")

    # Test with correct responses
    print("-" * 60)
    print("TEST 1: Evaluating CORRECT responses")
    print("-" * 60)

    for i, sample_data in enumerate(samples_data):
        metadata = sample_data["metadata"]
        prompt = sample_data["messages"][0]["content"]

        # Use simulated correct response
        response = SIMULATED_RESPONSES.get(i, "# No response")

        reward = await compute_code_execution_reward(
            response=response,
            label=sample_data.get("label"),
            metadata=metadata,
        )

        print(f"\nSample {i + 1}:")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Response: {response[:50].strip()}...")
        print(f"  Reward: {reward}")
        assert reward == 1.0, f"Expected 1.0 for correct response, got {reward}"

    print("\n✓ All correct responses got reward = 1.0")

    # Test with wrong responses
    print("\n" + "-" * 60)
    print("TEST 2: Evaluating WRONG responses")
    print("-" * 60)

    for i, response in WRONG_RESPONSES.items():
        sample_data = samples_data[i]
        metadata = sample_data["metadata"]
        prompt = sample_data["messages"][0]["content"]

        reward = await compute_code_execution_reward(
            response=response,
            label=sample_data.get("label"),
            metadata=metadata,
        )

        print(f"\nSample {i + 1} (wrong):")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Response: {response[:50].strip()}...")
        print(f"  Reward: {reward}")
        assert reward == 0.0, f"Expected 0.0 for wrong response, got {reward}"

    print("\n✓ All wrong responses got reward = 0.0")

    # Test edge cases
    print("\n" + "-" * 60)
    print("TEST 3: Edge cases")
    print("-" * 60)

    # No code in response
    no_code_response = "I don't know how to solve this problem."
    reward = await compute_code_execution_reward(
        response=no_code_response,
        label=None,
        metadata=samples_data[0]["metadata"],
    )
    print(f"\n  No code response: reward = {reward}")
    assert reward == 0.0

    # Empty response
    reward = await compute_code_execution_reward(
        response="",
        label=None,
        metadata=samples_data[0]["metadata"],
    )
    print(f"  Empty response: reward = {reward}")
    assert reward == 0.0

    # Syntax error in code
    syntax_error_response = """```python
def broken(
    print("missing paren"
```"""
    reward = await compute_code_execution_reward(
        response=syntax_error_response,
        label=None,
        metadata=samples_data[0]["metadata"],
    )
    print(f"  Syntax error response: reward = {reward}")
    assert reward == 0.0

    print("\n✓ All edge cases handled correctly")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


async def simulate_batch_evaluation():
    """Simulate batch evaluation as would happen during training."""
    data_path = Path(__file__).parent / "sample_data.jsonl"

    print("\n" + "=" * 60)
    print("Simulating Batch Evaluation (Training Mode)")
    print("=" * 60)

    with open(data_path) as f:
        samples_data = [json.loads(line) for line in f]

    # Create samples with simulated responses
    samples = []
    for i, sample_data in enumerate(samples_data):
        sample = Sample(
            prompt=sample_data["messages"],
            response=SIMULATED_RESPONSES.get(i, ""),
            label=sample_data.get("label"),
            metadata=sample_data["metadata"],
        )
        samples.append(sample)

    # Evaluate rewards in parallel (as would happen in training)
    tasks = [
        compute_code_execution_reward(
            response=sample.response,
            label=sample.label,
            metadata=sample.metadata,
        )
        for sample in samples
    ]

    rewards = await asyncio.gather(*tasks)

    print("\nBatch evaluation results:")
    for i, (sample, reward) in enumerate(zip(samples, rewards, strict=False)):
        print(f"  Sample {i + 1}: reward = {reward}")
        sample.reward = reward

    avg_reward = sum(rewards) / len(rewards)
    print(f"\nAverage reward: {avg_reward:.2f}")
    print(f"Pass rate: {sum(1 for r in rewards if r == 1.0) / len(rewards) * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(test_with_sample_data())
    asyncio.run(simulate_batch_evaluation())
