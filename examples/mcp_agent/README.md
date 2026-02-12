# MCP Agent Rollout

This example demonstrates how to train an agent with multi-step tool calling capabilities using MCP (Model Context Protocol) servers.

## Quick Start (Simplest Way)

Just provide the MCP server URL directly:

```bash
python -m slime.train \
    --rollout-function-path="slime.rollout.mcp_agent_rollout:generate_mcp_rollout" \
    --mcp-server-url http://localhost:8007/sse \
    # ... other training args
```

For multiple MCP servers:
```bash
python -m slime.train \
    --rollout-function-path="slime.rollout.mcp_agent_rollout:generate_mcp_rollout" \
    --mcp-server-url http://localhost:8007/sse http://localhost:8008/sse \
    # ... other training args
```

## Overview

The MCP Agent Rollout enables:
- **Single-turn multi-step** interactions: The agent can call tools multiple times within a single conversation turn
- **Multiple MCP servers**: Connect to various tool providers (search, code execution, databases, etc.)
- **Per-step samples**: Each reasoning step generates a separate sample for fine-grained training

## Architecture

```
User Query → Agent Loop → MCP Servers
                ↓
        [Step 1: Think + Tool Call]
                ↓
        [Step 2: Process Result + Tool Call]
                ↓
        [Step N: Final Answer]
                ↓
        List of Samples (for training)
```

## Prerequisites

1. Install MCP package:
   ```bash
   pip install "mcp[cli]"
   ```

2. Have one or more MCP servers running (SSE or Stdio transport)

## Configuration

### 1. Create MCP Server Config

Create a configuration function that returns a list of `MCPClientConfig` objects:

```python
# my_config.py
from slime.rollout.mcp import MCPClientConfig, MCPTransport

def mcp_server_config_fn() -> list[MCPClientConfig]:
    return [
        MCPClientConfig(
            name="WebSearch",
            transport=MCPTransport.SSE,
            url="http://localhost:8007/sse",
            concurrency_limit=16,
        ),
    ]
```

### 2. Prepare Training Data

Your dataset should have prompts that benefit from tool use. The metadata can include:
- `rm_type`: Set to `"llm_judge"` for LLM-based reward
- `rubrics`: Evaluation criteria for the judge

```json
{
  "prompt": "Search for the latest news about AI and summarize the top 3 stories.",
  "label": null,
  "metadata": {
    "rm_type": "llm_judge",
    "rubrics": "Evaluate based on: 1) Correct tool usage 2) Quality of summary 3) Accuracy of information"
  }
}
```

## Training

### Basic Usage (Simple URL)

```bash
python -m slime.train \
    --rollout-function-path="slime.rollout.mcp_agent_rollout:generate_mcp_rollout" \
    --mcp-server-url http://localhost:8007/sse \
    --mcp-max-steps 5 \
    # ... other training args
```

### Advanced Usage (Config File)

For more control over MCP server settings:

```bash
python -m slime.train \
    --rollout-function-path="slime.rollout.mcp_agent_rollout:generate_mcp_rollout" \
    --mcp-server-config-path="examples.mcp_agent.config:mcp_server_config_fn" \
    --mcp-max-steps 5 \
    --mcp-tool-parser qwen \
    # ... other training args
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mcp-server-url` | Direct URL(s) to MCP server(s) - simplest way | None |
| `--mcp-server-config-path` | Path to config function for advanced setup | None |
| `--mcp-max-steps` | Maximum tool-calling steps per query | 5 |
| `--mcp-tool-parser` | Parser for tool calls (currently: "qwen") | "qwen" |
| `--mcp-system-prompt-template` | Custom system prompt template | Built-in |

**Note:** Either `--mcp-server-url` or `--mcp-server-config-path` is required. Use `--mcp-server-url` for quick setup, or `--mcp-server-config-path` for advanced configurations (custom timeouts, blocklists, etc.).

## Tool Call Format

The agent uses Qwen-style tool calling format:

```
<think>
I need to search for information about AI news.
</think>

<tool_call>
{"name": "search", "arguments": {"query": "latest AI news 2024"}}
</tool_call>
```

After receiving tool results, the agent continues reasoning or provides a final answer.

## Reward Computation

Rewards are computed only for the **final step** of each trajectory using LLM-as-a-Judge:

1. Intermediate steps receive `reward = 0`
2. The final step is evaluated based on:
   - Tool usage correctness
   - Task completion
   - Reasoning quality
   - Efficiency (fewer steps is better)

### Multi-Rubric Evaluation

For more nuanced evaluation, use multiple rubrics:

```python
metadata = {
    "rm_type": "llm_judge",
    "multi_rubrics": [
        {"name": "tool_usage", "weight": 0.3, "criteria": "Correct selection and use of tools"},
        {"name": "task_completion", "weight": 0.4, "criteria": "Successfully completed the user's request"},
        {"name": "reasoning", "weight": 0.2, "criteria": "Clear and logical reasoning steps"},
        {"name": "efficiency", "weight": 0.1, "criteria": "Minimal steps to complete the task"},
    ],
}
```

## MCP Server Examples

### Starting a Simple SSE Server

Using the MCP filesystem server:
```bash
npx -y @modelcontextprotocol/server-filesystem /path/to/dir --sse --port 8007
```

### Custom MCP Server

See the [MCP documentation](https://modelcontextprotocol.io/) for creating custom servers.

## Troubleshooting

### Connection Issues

1. Verify MCP server is running: `curl http://localhost:8007/sse`
2. Check firewall settings
3. Ensure correct URL format (include `/sse` suffix for SSE servers)

### Tool Call Parsing Errors

1. Check model output format matches expected pattern
2. Try adjusting `rollout_temperature` (lower values = more deterministic)
3. Review logs for specific parsing errors

### Memory Issues

1. Reduce `concurrency_limit` in config
2. Decrease `mcp_max_steps`
3. Use `blocklist` to disable unused tools

## Evaluation (SimpleQA Verified Benchmark)

Run the [SimpleQA Verified](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified) benchmark against a deployed model:

```bash
uv run --with kagglehub --with pandas --with numpy --with openai \
  python simpleqa_verified_benchmark.py \
  --test-model "your-model-name" \
  --test-base-url "https://your-endpoint/v1" \
  --test-api-key "$YOUR_API_KEY" \
  --parallel 16 \
  --output results.csv
```

### Decontaminated Evaluation

To exclude benchmark prompts that overlap with training data (exact match), use `--decontaminate`:

```bash
uv run --with kagglehub --with pandas --with numpy --with openai \
  python simpleqa_verified_benchmark.py \
  --test-model "your-model-name" \
  --test-base-url "https://your-endpoint/v1" \
  --test-api-key "$YOUR_API_KEY" \
  --decontaminate simpleqa_1k.jsonl \
  --parallel 16 \
  --output results_decontaminated.csv
```

This loads the training JSONL, normalizes all user messages, and filters out any benchmark questions that exact-match a training prompt before evaluation.

### Key Eval Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--test-model` | Model name for the model under test | (required) |
| `--test-base-url` | OpenAI-compatible API base URL | OpenAI default |
| `--test-api-key` | API key for the test model | env `TEST_API_KEY` |
| `--judge-api-key` | OpenAI API key for the judge (GPT-4.1) | env `OPENAI_API_KEY` |
| `--num-examples` | Number of benchmark examples to evaluate | all |
| `--decontaminate` | Path to training JSONL to exclude overlapping prompts | None |
| `--parallel` | Number of parallel workers | 1 |
| `--output` | Path to save results CSV | None |
| `--extra-header` | Extra HTTP headers for test API (repeatable `KEY=VALUE`) | None |

## SimpleQA Verified Benchmark Results

### Why Decontamination?

Our training data (`simpleqa_1k.jsonl`) contains 1,000 examples sourced from the SimpleQA domain. Of these, **215 questions exactly match** prompts in the 1,000-question SimpleQA Verified benchmark. Evaluating on overlapping prompts inflates scores because the model may have memorized answers during training. To report fair results, we run two evaluations:

1. **Non-decontaminated (full):** All 1,000 benchmark questions — useful as a reference but potentially inflated.
2. **Decontaminated (clean):** The 785 benchmark questions that do **not** appear in training data — the reliable metric.

Decontamination uses exact match after text normalization (lowercase, strip punctuation, collapse whitespace). The `--decontaminate` flag automates this filtering at eval time.

### Decontaminated (Clean — Recommended)

Evaluated on 785 benchmark questions after removing 215 that overlap with `simpleqa_1k.jsonl`. Judge model: GPT-4.1.

| Model | Base | Eval Size | Correct | Incorrect | Not Attempted | Accuracy (Attempted) | F1 Score |
|-------|------|----------:|--------:|----------:|--------------:|---------------------:|---------:|
| Qwen3-4B (EigenTrain) | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | 785 | 55 | 702 | 27 | 7.27% | 0.0714 |

### Non-Decontaminated (Full Benchmark)

Evaluated on the full 1,000-question [SimpleQA Verified](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified) dataset. Judge model: GPT-4.1.

| Model | Base | Correct | Incorrect | Not Attempted | Accuracy (Attempted) | F1 Score |
|-------|------|--------:|----------:|--------------:|---------------------:|---------:|
| Qwen3-4B (EigenTrain) | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | 75 | 899 | 26 | 7.70% | 0.0760 |

## Example Output

```
Step 1:
  Thinking: "I need to search for AI news..."
  Tool Call: search(query="AI news 2024")

Step 2:
  Thinking: "Got 10 results, need to fetch details..."
  Tool Call: fetch(url="https://...")

Step 3:
  Final Answer: "Here are the top 3 AI news stories..."
  Reward: 0.85
```
