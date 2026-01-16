# OpenRouter LLM Judge 使用指南

使用 OpenRouter API 调用大语言模型（如 GPT-4o）来评判数学答案的正确性。

---

## 工作原理

### 判断流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   模型生成的    │     │   LLM Judge     │     │    返回分数     │
│   数学解答      │ ──▶ │   评估答案      │ ──▶ │   0 或 1       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 评判标准

LLM Judge 会收到三个输入：

| 输入 | 来源 | 说明 |
|------|------|------|
| `prompt` | 数据集 | 原始数学题目 |
| `response` | 模型生成 | 模型的完整回答（包含解题过程） |
| `label` | 数据集 | 正确答案（ground truth） |

LLM Judge 的评判逻辑：

1. **提取最终答案**：从模型回答中找到最终答案
   - 优先查找 `\boxed{...}` 格式
   - 其次查找 "answer is"、"=" 后的数值
   - 最后取回答中的最后一个数字

2. **与标准答案对比**：
   - 支持等价形式：`1/2` = `0.5` = `50%`
   - 支持不同表示：`π` = `3.14159...`
   - 忽略格式差异：空格、逗号等

3. **返回分数**：
   - **1 分**：最终答案正确
   - **0 分**：答案错误、缺失或无法判断

### 实际示例

**示例 1：正确答案**

```
题目: What is 2 + 2?
标准答案: 4
模型回答: Let me calculate... 2 + 2 = 4. So the answer is \boxed{4}.

LLM Judge 判断:
- 提取答案: 4
- 标准答案: 4
- 结果: score = 1 ✓
```

**示例 2：错误答案**

```
题目: What is the derivative of x^2?
标准答案: 2x
模型回答: The derivative is x^2 / 2 = x.

LLM Judge 判断:
- 提取答案: x
- 标准答案: 2x
- 结果: score = 0 ✗
```

**示例 3：等价形式**

```
题目: What fraction is 50%?
标准答案: 1/2
模型回答: 50% equals 0.5 or \boxed{0.5}.

LLM Judge 判断:
- 提取答案: 0.5
- 标准答案: 1/2
- 0.5 = 1/2 是等价的
- 结果: score = 1 ✓
```

### 与 Deepscaler 的对比

| 特性 | Deepscaler (规则) | LLM Judge |
|------|-------------------|-----------|
| 速度 | 极快 (~0.1ms) | 较慢 (~2-5s) |
| 成本 | 免费 | 需要 API 费用 |
| 等价形式识别 | 有限 | 强 |
| 推理过程评估 | 不支持 | 可选支持 |
| 复杂表达式 | 可能失败 | 更鲁棒 |

**建议**：
- 简单数值答案：用 Deepscaler
- 复杂表达式/需要语义理解：用 LLM Judge

---

## 快速开始

### 1. 设置环境变量

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxx"
export OPENROUTER_MODEL="openai/gpt-4o-mini"  # 可选，默认 gpt-4o-mini
```

### 2. 运行训练

```bash
python3 train.py \
  --custom-rm-path slime_plugins.rm.openrouter_llm_judge.llm_judge_rm \
  --reward-key score \
  # ... 其他参数
```

### 完整示例脚本

```bash
#!/bin/bash

export OPENROUTER_API_KEY="your-api-key"
export OPENROUTER_MODEL="openai/gpt-4o"

python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
  --custom-rm-path slime_plugins.rm.openrouter_llm_judge.llm_judge_rm \
  --reward-key score \
  --prompt-data your-dataset.jsonl \
  --input-key prompt \
  --label-key label \
  --use-wandb \
  --wandb-project my-project \
  --wandb-group llm-judge
```

---

## 配置选项

### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `OPENROUTER_API_KEY` | ✓ | - | OpenRouter API 密钥 |
| `OPENROUTER_MODEL` | - | `openai/gpt-4o-mini` | 使用的模型 |

### 推荐模型

| 模型 | 速度 | 准确性 | 成本 |
|------|------|--------|------|
| `openai/gpt-4o-mini` | 快 | 高 | 低 |
| `openai/gpt-4o` | 中 | 很高 | 中 |
| `anthropic/claude-3-haiku` | 快 | 高 | 低 |

---

## Wandb 监控

LLM Judge 会自动记录以下指标到 wandb：

| 指标 | 说明 |
|------|------|
| `llm_judge/latency_ms` | 单次调用延迟 (毫秒) |
| `llm_judge/avg_latency_ms` | 平均延迟 (毫秒) |
| `llm_judge/success_rate` | API 调用成功率 |

---

## 常见问题

### Q: API 调用失败怎么办？

A: LLM Judge 内置了重试机制，会自动处理速率限制和临时网络错误。如果持续失败，检查：
- API Key 是否有效
- 账户余额是否充足
- 网络是否可访问 openrouter.ai

### Q: 如何降低成本？

A:
- 使用更便宜的模型（如 gpt-4o-mini）
- 减少 `--n-samples-per-prompt`
- 减少 `--rollout-batch-size`

### Q: 可以用于非数学任务吗？

A: 可以，但需要修改 `JUDGE_SYSTEM_PROMPT` 来适配你的任务评判标准。
