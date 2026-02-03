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
