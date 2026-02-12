# MCP Agent Rollout

This example demonstrates how to train an agent with multi-step tool calling capabilities using MCP (Model Context Protocol) servers.

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

All evaluations use GPT-4.1 as judge. **Tools** = No means plain chat completion without MCP tools. **Decontaminated** = 785 questions after removing 215 that overlap with training data; **Full** = all 1,000 benchmark questions.

| Model | Base | RL | Tools | Decontaminated | Eval Size | Correct | Incorrect | Not Attempted | Accuracy (Attempted) | F1 Score |
|-------|------|:--:|:-----:|:--------------:|----------:|--------:|----------:|--------------:|---------------------:|---------:|
| Qwen3-4B-Thinking-2507 | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | - | No | Yes | 785 | 55 | 702 | 27 | 7.27% | 0.0714 |
| Qwen3-4B-Thinking-2507 | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | - | No | No | 1000 | 75 | 899 | 26 | 7.70% | 0.0760 |
| Qwen3-4B + MCP Agent RL (epoch 1) | [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | Epoch 1 | No | Yes | 785 | 71 | 686 | 28 | 9.38% | 0.0921 |

