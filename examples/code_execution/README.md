# Code Execution Reward Model

This module implements a code execution-based reward model for RL training, supporting LiveCodeBench-style evaluation.

## Overview

The Code Execution RM evaluates model-generated code by:
1. Extracting Python code from the model's response
2. Running the code against test cases in a secure sandbox
3. Returning a binary reward (1.0 if all tests pass, 0.0 otherwise)

## Features

- **Secure Sandbox Execution**: Memory limits, CPU time limits, and execution timeout
- **Multiple Test Modes**: STDIN (read from stdin, write to stdout) and FUNCTIONAL (call a function)
- **Compressed Test Cases**: Support for LiveCodeBench-style base64+zlib compressed test cases
- **Binary Reward**: Simple pass/fail evaluation (configurable)

## Data Format

Each training sample should be a JSON object with:

```json
{
  "messages": [
    {"role": "user", "content": "Write a function that..."}
  ],
  "label": null,
  "metadata": {
    "rm_type": "code_execution",
    "test_type": "STDIN",
    "test_cases": [
      {"input": "1\n2", "output": "3"},
      {"input": "5\n10", "output": "15"}
    ]
  }
}
```

### Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `rm_type` | Yes | Must be `"code_execution"` |
| `test_type` | No | `"STDIN"` (default) or `"FUNCTIONAL"` |
| `test_cases` | Yes | List of test cases or compressed string |
| `function_name` | For FUNCTIONAL | Name of the function to call |
| `starter_code` | No | Template code to prepend |
| `code_exec_timeout` | No | Timeout per test in seconds (default: 5.0) |
| `code_exec_memory_mb` | No | Memory limit in MB (default: 512) |

### Test Case Format

```json
{
  "input": "5\n3",
  "output": "8"
}
```

For LiveCodeBench compatibility, test cases can also be base64+zlib compressed:

```python
import base64, json, zlib

test_cases = [{"input": "1\n2", "output": "3"}]
compressed = base64.b64encode(zlib.compress(json.dumps(test_cases).encode())).decode()
```

## Usage

### In Training Config

```yaml
rm_type: "code_execution"
```

### In Eval Config

```yaml
eval:
  datasets:
    - name: livecodebench
      path: /data/livecodebench.jsonl
      rm_type: code_execution
      n_samples_per_eval_prompt: 8
      metadata_overrides:
        code_exec_timeout: 10.0
```

### Programmatic Usage

```python
from slime.rollout.rm_hub.code_execution import compute_code_execution_reward

response = '''
```python
x = int(input())
y = int(input())
print(x + y)
```
'''

metadata = {
    "test_type": "STDIN",
    "test_cases": [
        {"input": "1\n2", "output": "3"},
        {"input": "5\n5", "output": "10"},
    ],
}

reward = await compute_code_execution_reward(response, None, metadata)
# reward = 1.0 if all tests pass, 0.0 otherwise
```

## Testing

Run the test script:

```bash
uv run python examples/code_execution/test_rm.py
```

## Files

- `sample_data.jsonl` - Sample training data
- `eval_config.yaml` - Example evaluation configuration
- `test_rm.py` - Test script demonstrating usage

## Security

The sandbox implements multiple security measures:
- **Memory limit**: `resource.setrlimit(RLIMIT_AS, ...)`
- **CPU time limit**: `resource.setrlimit(RLIMIT_CPU, ...)`
- **Process limit**: `resource.setrlimit(RLIMIT_NPROC, 0)` (prevents fork bombs)
- **Execution timeout**: `asyncio.wait_for(..., timeout=...)`
- **Temporary directory isolation**: Each execution runs in a fresh temp directory

## Limitations

- **Python only**: Currently only supports Python code execution
- **STDIN/FUNCTIONAL modes**: Does not support file I/O or network-based testing
- **Platform**: Resource limits use the `resource` module (Linux/macOS only)
