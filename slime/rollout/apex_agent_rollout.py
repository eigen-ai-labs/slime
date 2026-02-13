"""APEX-Agents Zero-Shot Inference — Custom generate function.

Wraps mcp_agent_loop with:
  1. Tool filtering via FilteredMCPState (duck-types MCPState)
  2. Agent-design-specific system prompts (vanilla vs plan-then-act)

Registration:
    custom_generate_function_path: slime.rollout.apex_agent_rollout.apex_generate_with_mcp
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import TYPE_CHECKING, Any

from slime.rollout.mcp import MCPState, MCPTool, get_mcp_state
from slime.rollout.mcp_agent_rollout import (
    _build_config_from_urls,
    _finalize_sample_tokens,
    _probe_transport,
    build_system_message,
    mcp_agent_loop,
)
from slime.rollout.tool_filter import filter_tools_for_task
from slime.rollout.tool_parser import get_parser
from slime.utils.misc import load_function
from slime.utils.types import Sample

if TYPE_CHECKING:
    from slime.rollout.mcp.protocols import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

VANILLA_TEMPLATE = """\
You are an expert AI assistant specializing in {domain} tasks.

{world_context}

{tools_section}

When you need to use a tool, wrap your reasoning in <think> tags and the tool call in <tool_call> tags.
After receiving tool results, you can call more tools or provide your final answer.

Remember:
- Think step by step before using tools
- You can make multiple tool calls in sequence
- Provide a clear, thorough final answer when you're done
"""

PLAN_THEN_ACT_TEMPLATE = """\
You are an expert AI assistant specializing in {domain} tasks.
You MUST follow a strict two-phase approach for every task.

{world_context}

{tools_section}

## Phase 1: PLAN
Before using ANY tools, write a detailed plan inside <plan> tags:
- What information you need to gather
- Which tools you will use and in what order
- What analysis you will perform
- How you will structure your final answer

## Phase 2: ACT
After your plan, execute it step by step:
- Use tools as planned, wrapping reasoning in <think> tags and tool calls in <tool_call> tags
- Adjust your plan if tool results reveal new information
- After all tool calls, provide your final answer

Remember:
- ALWAYS plan first, then act
- You can make multiple tool calls in sequence
- Provide a clear, thorough final answer when you're done
"""

TEMPLATES: dict[str, str] = {
    "vanilla": VANILLA_TEMPLATE,
    "plan_then_act": PLAN_THEN_ACT_TEMPLATE,
}


# ---------------------------------------------------------------------------
# FilteredMCPState — duck-types MCPState with pre-filtered tools
# ---------------------------------------------------------------------------

class FilteredMCPState:
    """Wrapper around MCPState that returns a pre-filtered tool list.

    This lets ``mcp_agent_loop`` (which calls ``get_all_tools()`` at its start)
    receive only the tools relevant to the current task, without modifying that
    function.  Tool *execution* is delegated to the real state so routing works.
    """

    def __init__(self, real_state: MCPState, filtered_tools: list[MCPTool]) -> None:
        self._real = real_state
        self._tools = filtered_tools

    # -- tool discovery (filtered) ------------------------------------------

    async def get_all_tools(self, **kwargs: Any) -> list[MCPTool]:
        """Return the pre-filtered tool subset."""
        return self._tools

    # -- tool execution (delegated) -----------------------------------------

    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        **kwargs: Any,
    ) -> list[ToolResult]:
        """Route execution to the real MCPState."""
        return await self._real.execute_tool_calls(tool_calls, **kwargs)

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        return await self._real.execute_tool_call(tool_call)

    # -- passthrough properties ---------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._real.is_initialized

    async def initialize(self) -> None:
        await self._real.initialize()

    def get_tools_openai_format(self) -> list[dict[str, Any]]:
        return [t.to_openai_format() for t in self._tools]


# ---------------------------------------------------------------------------
# Helper: build the per-task system prompt template
# ---------------------------------------------------------------------------

def _build_apex_template(metadata: dict[str, Any]) -> str:
    """Select and format the system prompt template for this task.

    The template still contains ``{tools_section}`` — it will be filled by
    ``build_system_message()`` inside ``mcp_agent_loop``.
    """
    agent_design = metadata.get("agent_design", "vanilla")
    domain = metadata.get("domain", "professional services")
    world_description = metadata.get("world_description", "")

    template = TEMPLATES.get(agent_design, TEMPLATES["vanilla"])

    world_context = ""
    if world_description:
        world_context = f"Context: {world_description}"

    # Pre-fill domain and world_context, leave {tools_section} for build_system_message
    return template.format(
        domain=domain,
        world_context=world_context,
        tools_section="{tools_section}",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def apex_generate_with_mcp(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """APEX-specific custom generate function for zero-shot eval.

    Follows the same signature as ``generate_with_mcp`` so it can be used as
    ``custom_generate_function_path`` in eval_config.yaml.

    Flow:
        1. Read expected_output / agent_design from sample metadata
        2. Init MCP state singleton (same pattern as generate_with_mcp)
        3. Filter tools via filter_tools_for_task → wrap in FilteredMCPState
        4. Select system prompt template based on agent_design
        5. Call mcp_agent_loop with filtered state + template
        6. Finalize tokens and return
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    expected_output = metadata.get("expected_output", "message_in_console")
    agent_design = metadata.get("agent_design", "vanilla")

    # ----- 1. Get or initialise MCP state (singleton) ----------------------
    mcp_config_fn = None

    if hasattr(args, "mcp_server_config_path") and args.mcp_server_config_path:
        mcp_config_fn = load_function(args.mcp_server_config_path)
    elif hasattr(args, "mcp_server_url") and args.mcp_server_url:
        urls = args.mcp_server_url if isinstance(args.mcp_server_url, list) else [args.mcp_server_url]

        if not hasattr(_build_config_from_urls, "_transport_cache"):
            _build_config_from_urls._transport_cache = {}

        for url in urls:
            if url not in _build_config_from_urls._transport_cache:
                _build_config_from_urls._transport_cache[url] = await _probe_transport(url)

        detected = _build_config_from_urls._transport_cache

        def mcp_config_fn():
            return _build_config_from_urls(urls, detected)

    mcp_state = get_mcp_state(config_fn=mcp_config_fn)

    if not mcp_state.is_initialized:
        await mcp_state.initialize()

    # ----- 2. Filter tools for this task -----------------------------------
    all_tools = await mcp_state.get_all_tools()
    all_tools_openai = [t.to_openai_format() for t in all_tools]

    filtered_tools_openai = filter_tools_for_task(all_tools_openai, expected_output)

    # Map back to MCPTool objects for FilteredMCPState
    filtered_names = {
        t.get("function", {}).get("name") or t.get("name")
        for t in filtered_tools_openai
    }
    filtered_mcp_tools = [t for t in all_tools if t.name in filtered_names]

    filtered_state = FilteredMCPState(mcp_state, filtered_mcp_tools)

    logger.info(
        "APEX task %s [%s]: %d/%d tools after filtering for '%s'",
        metadata.get("task_id", "?"),
        agent_design,
        len(filtered_mcp_tools),
        len(all_tools),
        expected_output,
    )

    # ----- 3. Build system prompt template ---------------------------------
    system_template = _build_apex_template(metadata)

    # ----- 4. Get parser and sampling params --------------------------------
    parser_type = getattr(args, "mcp_tool_parser", "qwen")
    parser = get_parser(parser_type)

    max_steps = getattr(args, "mcp_max_steps", 10)

    chat_sampling_params = {
        "temperature": sampling_params.get("temperature", getattr(args, "rollout_temperature", 0.0)),
        "top_p": sampling_params.get("top_p", getattr(args, "rollout_top_p", 1.0)),
        "max_tokens": sampling_params.get("max_new_tokens", getattr(args, "rollout_max_response_len", 8192)),
    }

    stop_seqs = parser.get_stop_sequences()
    if stop_seqs:
        chat_sampling_params["stop"] = stop_seqs

    # ----- 5. Run agent loop -----------------------------------------------
    samples = await mcp_agent_loop(
        args=args,
        initial_sample=sample,
        mcp_state=filtered_state,
        parser=parser,
        sampling_params=chat_sampling_params,
        max_steps=max_steps,
        system_prompt_template=system_template,
    )

    if not samples:
        sample.status = Sample.Status.FAILED
        sample.response = "Error: Agent loop produced no samples"
        sample.reward = 0.0
        _finalize_sample_tokens(args, sample)
        return sample if evaluation else [sample]

    # ----- 6. Finalize tokens -----------------------------------------------
    for s in samples:
        _finalize_sample_tokens(args, s)

    if evaluation:
        return samples[-1]

    # For training-style returns (shouldn't hit in eval-only, but keep consistent)
    final_sample = samples[-1]
    if final_sample.reward is not None:
        for s in samples:
            s.reward = final_sample.reward

    return samples
