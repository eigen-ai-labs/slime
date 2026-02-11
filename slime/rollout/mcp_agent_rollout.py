"""
MCP Agent Rollout - Single-turn multi-step tool calling with MCP servers.

This module implements an agent rollout workflow that:
1. Connects to MCP servers to discover available tools
2. Runs a multi-step agent loop where the LLM can call tools
3. Returns a list of Samples for each step (for per-step loss calculation)

The workflow is designed to integrate with slime's training pipeline via
the custom_generate_function_path mechanism.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import threading
from argparse import Namespace
from typing import TYPE_CHECKING, Any

from slime.rollout.mcp import MCPClientConfig, MCPState, MCPTransport, get_mcp_state
from slime.rollout.tool_parser import get_parser
from slime.utils.http_utils import post
from slime.utils.misc import load_function
from slime.utils.types import Sample

if TYPE_CHECKING:
    from slime.rollout.tool_parser.base import ToolCallParser

logger = logging.getLogger(__name__)

# Default system prompt template
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

{tools_section}

When you need to use a tool, wrap your reasoning in <think> tags and the tool call in <tool_call> tags.
After receiving tool results, you can call more tools or provide your final answer.

Remember:
- Think step by step before using tools
- You can make multiple tool calls in sequence
- Provide a clear final answer when you're done
"""


def build_system_message(
    tools_openai: list[dict[str, Any]],
    parser: ToolCallParser,
    custom_template: str | None = None,
) -> str:
    """Build the system message with tool descriptions.

    Args:
        tools_openai: List of tools in OpenAI format
        parser: Tool call parser instance
        custom_template: Optional custom system prompt template

    Returns:
        Formatted system message string
    """
    tools_section = parser.format_tools_for_prompt(tools_openai)

    if custom_template:
        return custom_template.format(tools_section=tools_section)

    return DEFAULT_SYSTEM_PROMPT.format(tools_section=tools_section)


async def generate_single_step(
    args: Namespace,
    messages: list[dict[str, Any]],
    sampling_params: dict[str, Any],
) -> str:
    """Generate a single LLM response.

    Args:
        args: Training arguments
        messages: Chat messages for the model
        sampling_params: Sampling parameters

    Returns:
        Generated text response
    """
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"

    payload = {
        "model": "default",  # SGLang uses the loaded model
        "messages": messages,
        **sampling_params,
    }

    response = await post(url, payload)

    if "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0]["message"]["content"]

    logger.error("Unexpected response format: %s", response)
    return ""


async def mcp_agent_loop(
    args: Namespace,
    initial_sample: Sample,
    mcp_state: MCPState,
    parser: ToolCallParser,
    sampling_params: dict[str, Any],
    max_steps: int = 5,
    system_prompt_template: str | None = None,
) -> list[Sample]:
    """Run the multi-step agent loop.

    This function:
    1. Creates a Sample for each step
    2. Generates LLM responses and parses tool calls
    3. Executes tool calls via MCP
    4. Accumulates the conversation history

    Args:
        args: Training arguments
        initial_sample: The starting sample with user query
        mcp_state: MCP state manager with connected servers
        parser: Tool call parser
        sampling_params: Sampling parameters for generation
        max_steps: Maximum number of agent steps
        system_prompt_template: Optional custom system prompt

    Returns:
        List of Samples, one for each step (including final answer)
    """
    # Get available tools
    tools = await mcp_state.get_all_tools()
    tools_openai = [t.to_openai_format() for t in tools]

    # Build system message
    system_message = build_system_message(tools_openai, parser, system_prompt_template)

    # Initialize messages
    # If prompt is already a list (chat format), use it directly
    if isinstance(initial_sample.prompt, list):
        # Insert system message at the beginning if not present
        messages = copy.deepcopy(initial_sample.prompt)
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_message})
        else:
            # Append tool info to existing system message
            messages[0]["content"] = messages[0]["content"] + "\n\n" + system_message
    else:
        # String prompt - convert to chat format
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": initial_sample.prompt},
        ]

    samples: list[Sample] = []

    for step in range(max_steps):
        logger.debug("Agent step %d/%d", step + 1, max_steps)

        # Generate response
        step_response = await generate_single_step(args, messages, sampling_params)

        if not step_response:
            logger.warning("Empty response at step %d", step)
            break

        # Parse for tool calls
        parse_result = parser.parse(step_response)

        # Create sample for this step
        step_sample = Sample(
            group_index=initial_sample.group_index,
            index=initial_sample.index,
            prompt=copy.deepcopy(messages),
            response=step_response,
            label=initial_sample.label,
            reward=0.0,
            status=Sample.Status.PENDING,
            metadata={
                **initial_sample.metadata,
                "step": step,
                "has_tool_calls": parse_result.has_tool_calls,
                "thinking": parse_result.thinking,
            },
        )

        # Add assistant message to history
        messages.append({"role": "assistant", "content": step_response})

        if not parse_result.has_tool_calls:
            # No tool calls - this is the final answer
            step_sample.status = Sample.Status.COMPLETED
            samples.append(step_sample)
            logger.debug("Agent completed with final answer at step %d", step + 1)
            break

        # Execute tool calls
        tool_results = await mcp_state.execute_tool_calls(
            parse_result.tool_calls,
            parallel=True,
        )

        # Add tool results to messages (with truncation to avoid context overflow)
        max_tool_result_chars = getattr(args, "mcp_max_tool_result_chars", 8000)
        for result in tool_results:
            tool_msg = result.to_message()
            # Truncate long tool results to prevent context overflow
            if len(tool_msg.get("content", "")) > max_tool_result_chars:
                truncated_content = tool_msg["content"][:max_tool_result_chars]
                truncated_content += f"\n\n[... truncated, showing first {max_tool_result_chars} chars of {len(tool_msg['content'])} total ...]"
                tool_msg["content"] = truncated_content
                logger.debug("Truncated tool result from %d to %d chars", len(result.content), max_tool_result_chars)
            messages.append(tool_msg)

        step_sample.metadata["tool_results"] = [{"name": r.name, "content": r.content, "is_error": r.is_error} for r in tool_results]

        samples.append(step_sample)

        # Check if any tool had an error
        if any(r.is_error for r in tool_results):
            logger.warning("Tool execution error at step %d", step)
            # Continue anyway - model might handle the error

    # If we hit max_steps without a final answer, mark last sample as truncated
    if samples and samples[-1].status == Sample.Status.PENDING:
        samples[-1].status = Sample.Status.TRUNCATED

    # Let the pipeline compute reward for the final step (e.g. LLM judge).
    if samples:
        samples[-1].reward = None

    return samples


async def _probe_transport(url: str, timeout: float = 10.0) -> MCPTransport:
    """Probe MCP server to detect transport type (Streamable HTTP or SSE)."""
    from mcp import ClientSession
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
                    session = ClientSession(read, write)
                    async with session:
                        await session.initialize()
                        logger.info(f"Detected {transport_type.value} transport for {url}")
                        return transport_type
        except Exception as e:
            logger.debug(f"{transport_type.value} failed for {url}: {e}")

    logger.warning(f"Could not detect transport for {url}, defaulting to Streamable HTTP")
    return MCPTransport.STREAMABLE_HTTP


def _build_config_from_urls(urls: list[str], detected_transports: dict[str, MCPTransport] | None = None) -> list[MCPClientConfig]:
    """Build MCP configs from a list of URLs."""
    configs = []
    for i, url in enumerate(urls):
        name = f"MCP-{url.split('://')[1].split('/')[0]}" if "://" in url else f"Server{i + 1}"

        if detected_transports and url in detected_transports:
            transport = detected_transports[url]
        elif url.endswith("/sse"):
            transport = MCPTransport.SSE
        else:
            transport = MCPTransport.STREAMABLE_HTTP

        configs.append(MCPClientConfig(name=name, transport=transport, url=url, concurrency_limit=16, timeout=30.0))
    return configs


def _get_tokenizer(args: Namespace):
    """Get or create a cached tokenizer instance."""
    from slime.utils.processing_utils import load_tokenizer

    if not hasattr(_get_tokenizer, "_cache"):
        _get_tokenizer._cache = {}

    cache_key = args.hf_checkpoint
    if cache_key not in _get_tokenizer._cache:
        _get_tokenizer._cache[cache_key] = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return _get_tokenizer._cache[cache_key]


def _finalize_sample_tokens(args: Namespace, sample: Sample) -> None:
    """Tokenize sample prompt and response, setting tokens and response_length.

    This function ensures the sample has proper token information for training.
    It tokenizes the prompt and full conversation to compute response_length.

    Args:
        args: Training arguments
        sample: Sample to finalize (modified in place)
    """
    tokenizer = _get_tokenizer(args)

    # Build the full text: prompt + response
    if isinstance(sample.prompt, list):
        # Chat format prompt - apply chat template
        prompt_text = tokenizer.apply_chat_template(
            sample.prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # String prompt
        prompt_text = sample.prompt

    full_text = prompt_text + sample.response

    # Tokenize prompt and full text
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Set sample fields
    sample.tokens = full_tokens
    sample.response_length = len(full_tokens) - len(prompt_tokens)

    # Ensure response_length is at least 1 to avoid padding errors
    if sample.response_length <= 0:
        logger.warning(
            "Sample has non-positive response_length=%d, setting to 1. prompt_tokens=%d, full_tokens=%d",
            sample.response_length,
            len(prompt_tokens),
            len(full_tokens),
        )
        sample.response_length = max(1, sample.response_length)


_jsonl_lock = threading.Lock()


def _save_mcp_rollout(args: Namespace, samples: list[Sample]) -> None:
    """Append MCP rollout samples directly to rollouts.jsonl.

    The sidecar syncs this file to S3, and the backend serves it
    to the frontend Rollout Explorer via GET /api/jobs/{id}/rollouts.
    """
    output_dir = getattr(args, "save", None)
    if not output_dir:
        return

    rollout_dir = os.path.join(output_dir, "rollout_data")
    os.makedirs(rollout_dir, exist_ok=True)

    rollout_id = getattr(args, "_mcp_rollout_iteration", 0)

    lines: list[str] = []
    for s in samples:
        tool_calls = [
            {
                "tool_name": tr.get("name", "unknown"),
                "tool_input": "",
                "tool_output": tr.get("content", ""),
            }
            for tr in s.metadata.get("tool_results", [])
        ]
        record = {
            "rollout_id": rollout_id,
            "prompt": s.prompt,
            "response": s.response,
            "reward": s.reward if s.reward is not None else 0.0,
            "status": s.status.name if hasattr(s.status, "name") else str(s.status),
            "metadata": {
                "step": s.metadata.get("step", 0),
                "has_tool_calls": s.metadata.get("has_tool_calls", False),
                "tool_calls": tool_calls,
            },
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    jsonl_path = os.path.join(rollout_dir, "rollouts.jsonl")
    with _jsonl_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()


async def generate_with_mcp(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Custom generate function for MCP agent rollout.

    This function is designed to be used as `custom_generate_function_path`
    in the slime training configuration.

    Args:
        args: Training arguments
        sample: Input sample
        sampling_params: Sampling parameters
        evaluation: Whether this is evaluation mode

    Returns:
        Single Sample (for evaluation) or list of Samples (for training)
    """
    # Get or initialize MCP state
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

    # Ensure MCP is initialized
    if not mcp_state.is_initialized:
        await mcp_state.initialize()

    # Get parser
    parser_type = getattr(args, "mcp_tool_parser", "qwen")
    parser = get_parser(parser_type)

    # Get max steps
    max_steps = getattr(args, "mcp_max_steps", 5)

    # Get custom system prompt template
    system_template = getattr(args, "mcp_system_prompt_template", None)

    # Prepare sampling params for chat completions
    chat_sampling_params = {
        "temperature": sampling_params.get("temperature", args.rollout_temperature),
        "top_p": sampling_params.get("top_p", args.rollout_top_p),
        "max_tokens": sampling_params.get("max_new_tokens", args.rollout_max_response_len),
    }

    # Add stop sequences from parser
    stop_seqs = parser.get_stop_sequences()
    if stop_seqs:
        chat_sampling_params["stop"] = stop_seqs

    # Run agent loop
    samples = await mcp_agent_loop(
        args=args,
        initial_sample=sample,
        mcp_state=mcp_state,
        parser=parser,
        sampling_params=chat_sampling_params,
        max_steps=max_steps,
        system_prompt_template=system_template,
    )

    if not samples:
        # No samples generated - return error sample
        sample.status = Sample.Status.FAILED
        sample.response = "Error: Agent loop produced no samples"
        sample.reward = 0.0
        _finalize_sample_tokens(args, sample)
        return [sample]

    # Finalize all samples with proper token information
    for s in samples:
        _finalize_sample_tokens(args, s)

    if evaluation:
        # For evaluation, return only the final sample
        final_sample = samples[-1]
        return final_sample

    # Leave reward=None for the upper layer (sglang_rollout.generate_and_rm)
    # to compute via custom_rm / async_rm / batched_async_rm.
    # Only broadcast if reward was already set during generation.
    final_sample = samples[-1]
    if final_sample.reward is not None:
        for s in samples:
            s.reward = final_sample.reward

    # Save rollout data for the watcher → JSONL → S3 → frontend
    if getattr(args, "mcp_save_rollouts", False):
        _save_mcp_rollout(args, samples)

    # Log multi-step agent completion with sampling
    # Print once per rollout based on batch_size * n_samples_per_prompt
    if not hasattr(args, "_mcp_print_counter"):
        args._mcp_print_counter = 0

    args._mcp_print_counter += 1

    # Calculate print interval: total samples per rollout
    batch_size = getattr(args, "rollout_batch_size", 32)
    n_samples = getattr(args, "n_samples_per_prompt", 1)
    samples_per_rollout = batch_size * n_samples
    # Print once per rollout (first sample of each rollout)
    should_print = (args._mcp_print_counter % samples_per_rollout) == 1

    if should_print:
        # Extract user prompt from the original sample
        prompt_text = ""
        if hasattr(sample, "prompt") and sample.prompt:
            if isinstance(sample.prompt, list):
                for msg in sample.prompt:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt_text = msg.get("content", "")[:200]
                        break
            else:
                prompt_text = str(sample.prompt)[:200]

        # Build step-by-step log
        steps_log = [f"  Prompt: {prompt_text}"]
        for i, s in enumerate(samples, 1):
            steps_log.append(f"  Step {i}: {s.response}")
            # Add tool call results if any
            tool_results = s.metadata.get("tool_results", [])
            for tr in tool_results:
                status = "error" if tr.get("is_error") else "success"
                steps_log.append(f"    Tool: {tr.get('name', 'unknown')} -> {status}")

        final_reward = samples[-1].reward if samples[-1].reward else 0.0
        separator = "=" * 60
        logger.info(
            "\n%s\nMCP agent completed: %d steps, reward=%.3f\n%s\n%s",
            separator,
            len(samples),
            final_reward,
            "\n".join(steps_log),
            separator,
        )

    return samples


def generate_mcp_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> Any:
    """Main rollout function for MCP agent.

    This function serves as the entry point for the rollout workflow.
    It integrates with slime's training loop via `rollout_function_path`.

    Args:
        args: Training arguments
        rollout_id: Current rollout iteration
        data_source: Data source for samples
        evaluation: Whether this is evaluation mode

    Returns:
        RolloutFnTrainOutput or RolloutFnEvalOutput
    """
    # This function delegates to the standard sglang_rollout
    # but uses generate_with_mcp as the custom generate function
    from slime.rollout.sglang_rollout import generate_rollout

    # Set the custom generate function path to use our MCP generate
    original_custom_path = args.custom_generate_function_path
    args.custom_generate_function_path = "slime.rollout.mcp_agent_rollout.generate_with_mcp"

    # Pass rollout iteration number so _save_mcp_rollout can group by iteration
    args._mcp_rollout_iteration = rollout_id

    try:
        return generate_rollout(args, rollout_id, data_source, evaluation)
    finally:
        # Restore original setting
        args.custom_generate_function_path = original_custom_path


# Async cleanup helper
async def cleanup_mcp():
    """Clean up MCP connections."""
    from slime.rollout.mcp import reset_mcp_state

    await reset_mcp_state()
