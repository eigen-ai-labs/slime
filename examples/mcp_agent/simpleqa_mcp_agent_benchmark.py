"""
SimpleQA Verified Benchmark — MCP Agent Evaluation

Evaluates LLM factuality on the SimpleQA Verified benchmark using multi-step
tool calling via MCP servers. Unlike the plain benchmark (simpleqa_verified_benchmark.py)
which tests direct knowledge, this script measures how well a model can use tools
(e.g., web search) to find correct answers.

Uses the MCP protocol (via the `mcp` Python package) to:
1. Connect to MCP servers and discover available tools
2. Run a multi-step agent loop (generate -> parse -> tool call -> generate -> ...)
3. Grade final answers against gold targets using GPT-4.1 judge

This script is standalone — it does NOT import from slime, so no heavy deps
(ray, torch, etc.) are needed. Only requires: mcp, openai, kagglehub, pandas.

Research Paper: https://arxiv.org/abs/2509.07968

Usage:
    # Uses default SerpAPI MCP server (same as training)
    uv run python examples/mcp_agent/simpleqa_mcp_agent_benchmark.py \
      --test-model "Qwen/Qwen3-4B-Thinking-2507" \
      --test-base-url "http://localhost:30000/v1" \
      --max-steps 10 \
      --parallel 8 \
      --output results_mcp.csv

    # With custom MCP server
    uv run python examples/mcp_agent/simpleqa_mcp_agent_benchmark.py \
      --test-model "Qwen/Qwen3-4B-Thinking-2507" \
      --test-base-url "http://localhost:30000/v1" \
      --mcp-server-url "https://your-mcp-server.com/mcp" \
      --max-steps 10 \
      --parallel 8 \
      --output results_mcp.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import string
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import kagglehub
import pandas as pd
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default MCP server (same SerpAPI server used in Agent RL training)
DEFAULT_MCP_SERVER_URL = "https://mcp.serpapi.com/3486fbdce5cc091c7edcaaa1a222c0335744587fe8f88fa7c572dc75cdcc177c/mcp"

# ============================================================================
# Inline MCP client (extracted from slime.rollout.mcp — no slime import needed)
# ============================================================================


class MCPTransport(str, Enum):
    SSE = "sse"
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str | dict[str, Any]
    is_error: bool = False

    def to_message(self) -> dict[str, Any]:
        content = self.content if isinstance(self.content, str) else str(self.content)
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": content,
        }


@dataclass
class MCPClientConfig:
    name: str
    transport: MCPTransport | str = MCPTransport.SSE
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    concurrency_limit: int = 16
    timeout: float = 30.0

    def __post_init__(self):
        if isinstance(self.transport, str):
            self.transport = MCPTransport(self.transport)


class MCPClient:
    """Lightweight MCP client using the mcp package."""

    def __init__(self, config: MCPClientConfig) -> None:
        self.config = config
        self.name = config.name
        self._session = None
        self._exit_stack: AsyncExitStack | None = None
        self._semaphore = asyncio.Semaphore(config.concurrency_limit)
        self._tools_cache: list[MCPTool] | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> None:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamablehttp_client

        if self._connected:
            return

        self._exit_stack = AsyncExitStack()
        try:
            if self.config.transport == MCPTransport.SSE:
                transport = await self._exit_stack.enter_async_context(sse_client(self.config.url))
                read, write = transport
            elif self.config.transport == MCPTransport.STREAMABLE_HTTP:
                transport = await self._exit_stack.enter_async_context(streamablehttp_client(self.config.url, timeout=self.config.timeout))
                read, write, _ = transport
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            self._connected = True
            logger.info("Connected to MCP server: %s", self.name)
        except Exception as e:
            logger.error("Failed to connect to %s: %s", self.name, e)
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None
            raise

    async def disconnect(self) -> None:
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("Error during cleanup for %s: %s", self.name, e)
            self._exit_stack = None
        self._session = None
        self._tools_cache = None
        self._connected = False

    async def list_tools(self) -> list[MCPTool]:
        if self._tools_cache is not None:
            return self._tools_cache
        assert self._session is not None
        response = await self._session.list_tools()
        tools = [
            MCPTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
                server_name=self.name,
            )
            for t in response.tools
        ]
        self._tools_cache = tools
        return tools

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        if not self.is_connected:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: {self.name} not connected",
                is_error=True,
            )
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(tool_call.name, arguments=tool_call.arguments),
                    timeout=self.config.timeout,
                )
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif hasattr(item, "data"):
                        parts.append(json.dumps(item.data))
                    else:
                        parts.append(str(item))
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content="\n".join(parts) if parts else "",
                    is_error=result.isError if hasattr(result, "isError") else False,
                )
            except asyncio.TimeoutError:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error: timed out after {self.config.timeout}s",
                    is_error=True,
                )
            except Exception as e:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error: {type(e).__name__}: {e}",
                    is_error=True,
                )


class MCPState:
    """Manages connections to multiple MCP servers."""

    def __init__(self, configs: list[MCPClientConfig]) -> None:
        self._configs = configs
        self._clients: dict[str, MCPClient] = {}
        self._tool_to_server: dict[str, str] = {}
        self._tools_cache: list[MCPTool] | None = None

    async def initialize(self) -> None:
        for config in self._configs:
            client = MCPClient(config)
            try:
                await client.connect()
                self._clients[config.name] = client
            except Exception:
                logger.error("Skipping server %s (connection failed)", config.name)

        # Build tool routing table
        for name, client in self._clients.items():
            if not client.is_connected:
                continue
            for tool in await client.list_tools():
                self._tool_to_server[tool.name] = name

        logger.info("MCPState: %d servers, %d tools", len(self._clients), len(self._tool_to_server))

    async def get_all_tools(self) -> list[MCPTool]:
        if self._tools_cache is not None:
            return self._tools_cache
        tools = []
        for client in self._clients.values():
            if client.is_connected:
                tools.extend(await client.list_tools())
        self._tools_cache = tools
        return tools

    async def execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        async def _exec(tc: ToolCall) -> ToolResult:
            server = self._tool_to_server.get(tc.name)
            if server is None:
                return ToolResult(tc.id, tc.name, f"Error: unknown tool '{tc.name}'", True)
            return await self._clients[server].call_tool(tc)

        return await asyncio.gather(*[_exec(tc) for tc in tool_calls])

    async def shutdown(self) -> None:
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()
        self._tool_to_server.clear()
        self._tools_cache = None


# ============================================================================
# Inline Qwen tool call parser (extracted from slime.rollout.tool_parser.qwen)
# ============================================================================

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_CALL_TRUNCATED = re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL)


@dataclass
class ParseResult:
    tool_calls: list[ToolCall]
    thinking: str | None = None
    content: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


def _parse_tool_call_json(json_str: str) -> ToolCall | None:
    json_str = json_str.strip()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", json_str, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    name = None
    arguments = {}
    if "name" in data:
        name = data["name"]
        arguments = data.get("arguments", data.get("parameters", {}))
    elif "function" in data:
        name = data["function"].get("name")
        arguments = data["function"].get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
    elif "tool" in data:
        name = data["tool"]
        arguments = data.get("args", data.get("arguments", {}))

    if not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}

    return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)


def parse_qwen(text: str) -> ParseResult:
    thinking_matches = THINK_PATTERN.findall(text)
    thinking = "\n".join(m.strip() for m in thinking_matches) if thinking_matches else None

    tool_calls = []
    for m in TOOL_CALL_PATTERN.findall(text):
        tc = _parse_tool_call_json(m.strip())
        if tc:
            tool_calls.append(tc)

    if not tool_calls:
        trunc = TOOL_CALL_TRUNCATED.search(text)
        if trunc:
            tc = _parse_tool_call_json(trunc.group(1).strip())
            if tc:
                tool_calls.append(tc)

    content = THINK_PATTERN.sub("", text)
    content = TOOL_CALL_PATTERN.sub("", content)
    content = TOOL_CALL_TRUNCATED.sub("", content).strip()

    return ParseResult(tool_calls=tool_calls, thinking=thinking, content=content)


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    lines = ["# Tools", "", "You have access to the following tools:"]
    for tool in tools:
        func = tool.get("function", tool)
        lines += ["", f"## {func.get('name', '')}", func.get("description", ""), "", "Parameters:", "```json", json.dumps(func.get("parameters", {}), indent=2), "```"]
    lines += [
        "",
        "# Tool Use Format",
        "",
        "To use a tool, output your thinking in <think> tags, then the tool call in <tool_call> tags:",
        "",
        "<think>",
        "Your reasoning about which tool to use and why...",
        "</think>",
        "",
        "<tool_call>",
        '{"name": "tool_name", "arguments": {"param1": "value1"}}',
        "</tool_call>",
        "",
        "You can make multiple tool calls in one response. After receiving tool results, continue reasoning or provide your final answer.",
    ]
    return "\n".join(lines)


STOP_SEQUENCES = ["</tool_call>"]

# ============================================================================
# Grading (same as simpleqa_verified_benchmark.py)
# ============================================================================

JUDGE_MODEL = "gpt-4.1-2025-04-14"

GRADER_TEMPLATE = """
  Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
  First, I will give examples of each grade, and then you will grade a new example.


  The following are examples of CORRECT predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia Obama and Sasha Obama
  Predicted answer 1: sasha and malia obama
  Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
  Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
  Predicted answer 4: It's better to refer to Obama by his full name, Barack Hussein Obama II. He has two daughters: Malia and Sasha.
  ```
  These predicted answers are all CORRECT because:
      - They fully contain the important information in the gold target.
      - They do not contain any information that contradicts the gold target.
      - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
      - Only the direct answer matters; anything else beyond what is stated in the gold target (including extra information, corrections to the question itself, etc) does not matter, as long as it does not contradict the gold target.
      - Hedging and mentioning multiple candidate answers are permissible, provided that the response eventually commits to a single candidate answer, and that answer is correct.


  The following are examples of INCORRECT predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia and Sasha
  Predicted answer 1: Malia.
  Predicted answer 2: Malia, Sasha, and Susan.
  Predicted answer 3: Barack Obama does not have any children.
  Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
  Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
  Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
  ```
  These predicted answers are all INCORRECT because:
      - A factual statement in the answer contradicts the gold target.
      - A hedged statement (e.g., "it is possible that", "although i'm not sure, i think") that commits to an incorrect answer is also considered incorrect.


  The following are examples of NOT_ATTEMPTED predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia and Sasha
  Predicted answer 1: I don't know.
  Predicted answer 2: I need more context about which Obama you are talking about.
  Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
  Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
  Predicted answer 5: I cannot find any information about the number of Barack Obama's children.
  Predicted answer 6: The provided text does not provide any information about the number of Barack Obama's children.
  Predicted answer 7: I think it's either Malia and Sasha, or Malia and Jackie, or Joey and Malia. I'm not sure which one.
  ```
  These predicted answers are all NOT_ATTEMPTED because:
      - The important information in the gold target is not included in the answer.
      - No statements in the answer contradict the gold target.
      - Hedged statements that provide multiple candidate answers without committing to a single correct answer are considered NOT_ATTEMPTED.


  Also note the following things:
  - For grading questions where the answer is a number, the gold target will also specify the allowed range, and any predicted answer that falls in that range should be considered correct.
  - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
  - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
  - Do not punish for typos in people's name if it's clearly the same name.


  Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
  ```
  Question: {question}
  Gold target: {target}
  Predicted answer: {predicted_answer}
  ```

  Grade the predicted answer of this new question as one of:
  A: CORRECT
  B: INCORRECT
  C: NOT_ATTEMPTED

  Just return the letters "A", "B", or "C", with no text around it.
""".strip()

CHOICE_STRINGS = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}

# ============================================================================
# Default system prompt
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

{tools_section}

When you need to use a tool, wrap your reasoning in <think> tags and the tool call in <tool_call> tags.
After receiving tool results, you can call more tools or provide your final answer.

Remember:
- Think step by step before using tools
- You can make multiple tool calls in sequence
- Provide a clear final answer when you're done
"""

# ============================================================================
# Clients (initialized in main)
# ============================================================================

judge_client: OpenAI | None = None
test_client: OpenAI | None = None
TEST_MODEL: str | None = None
test_extra_headers: dict = {}


def init_clients(
    judge_api_key: str,
    judge_base_url: str | None,
    test_api_key: str,
    test_base_url: str | None,
    test_model: str,
    extra_headers: dict | None = None,
):
    global judge_client, test_client, TEST_MODEL, test_extra_headers
    TEST_MODEL = test_model
    test_extra_headers = extra_headers or {}
    judge_client = OpenAI(
        api_key=judge_api_key,
        **({"base_url": judge_base_url} if judge_base_url else {}),
    )
    test_client = OpenAI(
        api_key=test_api_key,
        **({"base_url": test_base_url} if test_base_url else {}),
    )


# ============================================================================
# LLM helpers
# ============================================================================


def generate_judge(prompt: str) -> str:
    return judge_client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}]).choices[0].message.content


def generate_test_chat(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    stop: list[str] | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": TEST_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        kwargs["stop"] = stop
    if test_extra_headers:
        kwargs["extra_headers"] = test_extra_headers
    return test_client.chat.completions.create(**kwargs).choices[0].message.content or ""


def grade_answer_with_llm(question: str, target: str, predicted_answer: str) -> str:
    prompt = GRADER_TEMPLATE.format(question=question, target=target, predicted_answer=predicted_answer)
    resp = generate_judge(prompt)
    m = re.search(r"(A|B|C)", resp)
    if m:
        return m.group(0)
    upper = resp.upper()
    if "CORRECT" in upper and "INCORRECT" not in upper:
        return "A"
    if "INCORRECT" in upper:
        return "B"
    if "NOT_ATTEMPTED" in upper:
        return "C"
    return "C"


# ============================================================================
# Dataset loading
# ============================================================================

KAGGLEHUB_HANDLE = "deepmind/simpleqa-verified"
SIMPLEQA_CSV = "simpleqa_verified.csv"
_dataset_path = None


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def _load_contaminated(jsonl_path: str) -> set[str]:
    out = set()
    with open(jsonl_path) as f:
        for line in f:
            for msg in json.loads(line).get("messages", []):
                if msg.get("role") == "user":
                    out.add(_normalize(msg["content"]))
    return out


def load_dataset(num_examples: int | None = None, decontaminate_path: str | None = None) -> pd.DataFrame:
    global _dataset_path
    if _dataset_path is None:
        _dataset_path = Path(kagglehub.dataset_download(KAGGLEHUB_HANDLE)) / SIMPLEQA_CSV
        print(f"Dataset: {_dataset_path}")

    df = pd.read_csv(_dataset_path)
    if decontaminate_path:
        contam = _load_contaminated(decontaminate_path)
        before = len(df)
        df = df[~df["problem"].apply(_normalize).isin(contam)]
        print(f"Decontamination: {before} -> {len(df)} ({before - len(df)} removed)")

    examples = [row.to_dict() for _, row in df.iterrows()]
    if num_examples and num_examples > 0:
        rng = random.Random(0)
        k = min(num_examples, len(examples))
        examples = [examples[i] for i in sorted(rng.sample(range(len(examples)), k=k))]
    return pd.DataFrame(examples)


# ============================================================================
# MCP transport probe
# ============================================================================


async def _probe_transport(url: str, timeout: float = 10.0) -> MCPTransport:
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client

    for transport_type, client_fn in [
        (MCPTransport.STREAMABLE_HTTP, lambda: streamablehttp_client(url, timeout=timeout)),
        (MCPTransport.SSE, lambda: sse_client(url)),
    ]:
        try:
            async with asyncio.timeout(timeout):
                async with client_fn() as transport:
                    read, write = transport[0], transport[1]
                    from mcp import ClientSession as CS

                    async with CS(read, write) as session:
                        await session.initialize()
                        logger.info("Detected %s for %s", transport_type.value, url)
                        return transport_type
        except Exception as e:
            logger.debug("%s failed for %s: %s", transport_type.value, url, e)

    logger.warning("Defaulting to streamable_http for %s", url)
    return MCPTransport.STREAMABLE_HTTP


def _build_configs(urls: list[str], transports: dict[str, MCPTransport]) -> list[MCPClientConfig]:
    configs = []
    for i, url in enumerate(urls):
        name = f"MCP-{url.split('://')[1].split('/')[0]}" if "://" in url else f"Server{i + 1}"
        t = transports.get(url, MCPTransport.SSE if url.endswith("/sse") else MCPTransport.STREAMABLE_HTTP)
        configs.append(MCPClientConfig(name=name, transport=t, url=url, concurrency_limit=16, timeout=30.0))
    return configs


# ============================================================================
# MCP Agent Loop
# ============================================================================


async def mcp_agent_loop(
    question: str,
    mcp_state: MCPState,
    max_steps: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_tool_chars: int = 8000,
) -> dict:
    """Run multi-step agent loop. Returns dict with predicted_answer, num_steps, trajectory."""
    tools = await mcp_state.get_all_tools()
    tools_openai = [t.to_openai_format() for t in tools]
    tools_section = format_tools_for_prompt(tools_openai)
    system_msg = DEFAULT_SYSTEM_PROMPT.format(tools_section=tools_section)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]
    trajectory = []

    for step in range(max_steps):
        resp = generate_test_chat(messages, temperature=temperature, max_tokens=max_tokens, stop=STOP_SEQUENCES)
        if not resp:
            break

        pr = parse_qwen(resp)
        rec = {"step": step, "response": resp, "has_tool_calls": pr.has_tool_calls, "tool_results": []}
        messages.append({"role": "assistant", "content": resp})

        if not pr.has_tool_calls:
            rec["is_final"] = True
            trajectory.append(rec)
            break

        results = await mcp_state.execute_tool_calls(pr.tool_calls)
        for r in results:
            msg = r.to_message()
            c = msg.get("content", "")
            if len(c) > max_tool_chars:
                msg["content"] = c[:max_tool_chars] + f"\n\n[... truncated {max_tool_chars}/{len(c)} chars ...]"
            messages.append(msg)
            rec["tool_results"].append({"name": r.name, "is_error": r.is_error})

        trajectory.append(rec)

    # Extract final answer
    predicted = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            text = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL)
            text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
            text = re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL)
            text = text.strip()
            if text:
                predicted = text
                break

    return {"predicted_answer": predicted, "num_steps": len(trajectory), "trajectory": trajectory}


# ============================================================================
# Task runner
# ============================================================================


async def run_task(
    idx: int,
    data_row: dict,
    mcp_state: MCPState,
    sem: asyncio.Semaphore,
    max_steps: int,
    temperature: float,
    max_tokens: int,
    max_tool_chars: int,
) -> tuple[int, dict | None, Exception | None]:
    async with sem:
        try:
            q = data_row.get("problem", "")
            gold = data_row.get("answer", "")
            agent = await mcp_agent_loop(q, mcp_state, max_steps, temperature, max_tokens, max_tool_chars)
            grade = await asyncio.get_event_loop().run_in_executor(None, grade_answer_with_llm, q, gold, agent["predicted_answer"])
            return (
                idx,
                {
                    "question": q,
                    "gold_target": gold,
                    "predicted_answer": agent["predicted_answer"],
                    "grade_letter": grade,
                    "grade_str": CHOICE_STRINGS.get(grade, "UNKNOWN"),
                    "is_correct": grade == "A",
                    "is_incorrect": grade == "B",
                    "is_not_attempted": grade == "C",
                    "num_steps": agent["num_steps"],
                    "trajectory_json": json.dumps(agent["trajectory"], ensure_ascii=False),
                },
                None,
            )
        except Exception as e:
            logger.error("Task %d error: %s", idx, e, exc_info=True)
            return idx, None, e


# ============================================================================
# Metrics
# ============================================================================


def accuracy_attempted(df: pd.DataFrame) -> float:
    a = df["is_correct"].sum() + df["is_incorrect"].sum()
    return df["is_correct"].sum() / a if a > 0 else 0.0


def f1_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    mc = df["is_correct"].sum() / len(df)
    aa = accuracy_attempted(df)
    d = aa + mc
    return 2 * aa * mc / d if d > 0 else 0.0


# ============================================================================
# CLI & Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="SimpleQA Verified — MCP Agent Eval")
    p.add_argument("--test-model", required=True)
    p.add_argument("--test-base-url", default=None)
    p.add_argument("--test-api-key", default=None)
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--mcp-server-url", action="append", default=[], metavar="URL", help="MCP server URL (repeatable). Defaults to SerpAPI.")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-tool-result-chars", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--decontaminate", default=None, metavar="JSONL_PATH")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--output", default=None)
    p.add_argument("--extra-header", action="append", default=[], metavar="KEY=VALUE")
    return p.parse_args()


async def async_main():
    args = parse_args()
    judge_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "")
    test_key = args.test_api_key or os.environ.get("TEST_API_KEY", "no-key")

    extra_headers = {}
    for h in args.extra_header:
        k, _, v = h.partition("=")
        extra_headers[k] = v

    init_clients(judge_key, args.judge_base_url, test_key, args.test_base_url, args.test_model, extra_headers)

    if not args.mcp_server_url:
        args.mcp_server_url = [DEFAULT_MCP_SERVER_URL]
        print(f"Using default MCP server: {DEFAULT_MCP_SERVER_URL}")

    print(f"Test model:  {TEST_MODEL} @ {args.test_base_url or 'OpenAI'}")
    print(f"Judge model: {JUDGE_MODEL} @ {args.judge_base_url or 'OpenAI'}")
    print(f"MCP servers: {args.mcp_server_url}")
    print(f"Max steps:   {args.max_steps}  Parallel: {args.parallel}  Temp: {args.temperature}")
    print()

    # Smoke tests
    print("[Smoke] Judge...", end=" ", flush=True)
    print(f"OK: {generate_judge('Count to 3.').strip()[:80]}")
    print("[Smoke] Test...", end=" ", flush=True)
    print(f"OK: {generate_test_chat([{'role': 'user', 'content': 'Count to 3.'}], temperature=0, max_tokens=64).strip()[:80]}")
    print()

    # Probe transports
    print("Probing MCP transports...")
    transports = {}
    for url in args.mcp_server_url:
        transports[url] = await _probe_transport(url)
        print(f"  {url} -> {transports[url].value}")
    print()

    # Init MCP
    configs = _build_configs(args.mcp_server_url, transports)
    mcp_state = MCPState(configs)
    await mcp_state.initialize()
    tools = await mcp_state.get_all_tools()
    print(f"Tools ({len(tools)}):")
    for t in tools:
        print(f"  - {t.name}: {t.description[:80]}")
    print()

    # Load dataset
    df = load_dataset(args.num_examples, args.decontaminate)
    total = len(df)
    print(f"Evaluating {total} examples...\n")

    # Run
    sem = asyncio.Semaphore(args.parallel)
    tasks = [run_task(i, row.to_dict(), mcp_state, sem, args.max_steps, args.temperature, args.max_tokens, args.max_tool_result_chars) for i, (_, row) in enumerate(df.iterrows())]

    results_map = {}
    done = 0
    errs = 0
    t0 = time.time()

    for coro in asyncio.as_completed(tasks):
        idx, result, err = await coro
        done += 1
        rate = done / (time.time() - t0) if time.time() > t0 else 0
        if err:
            errs += 1
            print(f"[{done}/{total}] ERROR idx={idx}: {err}", flush=True)
        else:
            results_map[idx] = result
            print(f"[{done}/{total} {rate:.1f}/s] {result['grade_str']:15s} steps={result['num_steps']} | {result['question'][:55]}...", flush=True)

    await mcp_state.shutdown()

    results_list = [results_map[i] for i in sorted(results_map)]
    rdf = pd.DataFrame(results_list)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total:              {len(rdf)}")
    print(f"Correct:            {rdf['is_correct'].sum()}")
    print(f"Incorrect:          {rdf['is_incorrect'].sum()}")
    print(f"Not attempted:      {rdf['is_not_attempted'].sum()}")
    if errs:
        print(f"Errors:             {errs}")
    print(f"Accuracy (attempted): {accuracy_attempted(rdf):.4f}")
    print(f"F1 Score:             {f1_score(rdf):.4f}")
    print(f"Avg steps/question:   {rdf['num_steps'].mean():.2f}")
    print(f"Total time:           {time.time() - t0:.1f}s")
    print("=" * 60)

    if args.output:
        rdf.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    asyncio.run(async_main())
