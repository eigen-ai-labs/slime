#!/usr/bin/env python3
"""Convert tool-calling chat transcripts into slime SFT training data.

Input JSONL format (per line, simplified):
{
  "tools": [...],  # optional tool schema (list/dict or JSON string)
  "messages": [
      {"role": "system" | "user" | "assistant" | "tool_call" | "tool_response", ...},
      ...
  ],
  ...
}

Output schema (one row per input sample) compatible with Qwen chat_template:
{
  "tools": [...],  # preserved from the input and normalized
  "messages": [
      {"role": "system" | "user" | "assistant" | "tool", "content": "...", "step_loss_mask": 0/1, ...},
      ...
  ]
}

- Assistant tool calls are emitted via the `tool_calls` field:
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}], ...}
- Tool responses use role "tool".
- `step_loss_mask=1` is set on assistant/tool-call turns that should be supervised; all other roles default to 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input JSONL file containing tool transcripts.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination parquet path for slime SFT data.",
    )
    parser.add_argument(
        "--output-format",
        choices=("parquet", "jsonl"),
        help=(
            "Explicitly set output format. If omitted, the format is inferred "
            "from the output file extension (defaults to parquet)."
        ),
    )
    parser.add_argument(
        "--system-prefix",
        default="",
        help=(
            "Optional text prepended to the first system message. If no system "
            "message exists, a new one is inserted at the beginning."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_tools(tools: Any) -> Any:
    if isinstance(tools, str):
        try:
            return json.loads(tools)
        except json.JSONDecodeError:
            return tools
    return tools


def _normalize_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _normalize_tool_call_entry(entry: Any) -> dict[str, Any] | None:
    data = entry
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {
                "type": "function",
                "function": {"name": "unknown", "arguments": data},
            }

    if isinstance(data, dict):
        if "type" in data and "function" in data:
            fn = data.get("function", {}) or {}
            name = fn.get("name")
            args = _normalize_arguments(fn.get("arguments"))
            if not name:
                return None
            return {"type": "function", "function": {"name": name, "arguments": args}}

        name = data.get("name")
        args = _normalize_arguments(data.get("arguments"))
        if not name:
            return None
        return {"type": "function", "function": {"name": name, "arguments": args}}

    if isinstance(data, list):
        normalized_list: list[dict[str, Any]] = []
        for item in data:
            normalized = _normalize_tool_call_entry(item)
            if normalized:
                normalized_list.append(normalized)
        # Returning None will allow the caller to extend with the flattened list.
        return {"type": "__list__", "function": {"name": "", "arguments": normalized_list}}

    return None


def parse_tool_calls(payload: Any) -> list[dict[str, Any]]:
    if payload is None or payload == "":
        return []

    data = payload
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return [
                {
                    "type": "function",
                    "function": {"name": "unknown", "arguments": data},
                }
            ]

    if isinstance(data, dict) and "tool_calls" in data:
        data = data["tool_calls"]

    if not isinstance(data, list):
        data = [data]

    normalized_list: list[dict[str, Any]] = []
    for element in data:
        normalized = _normalize_tool_call_entry(element)
        if not normalized:
            continue
        if normalized["type"] == "__list__":
            nested = normalized["function"]["arguments"]
            if isinstance(nested, list):
                normalized_list.extend(nested)
        else:
            normalized_list.append(normalized)

    return normalized_list


def convert_messages(record: dict[str, Any], system_prefix: str) -> dict[str, Any]:
    tools = normalize_tools(record.get("tools"))
    prefix_text = system_prefix.strip()
    prefix_applied = not prefix_text

    messages_out: list[dict[str, Any]] = []

    for message in record.get("messages", []):
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            msg_content = content if content is not None else ""
            if prefix_text and not prefix_applied:
                msg_content = prefix_text + ("\n" + msg_content if msg_content else "")
                prefix_applied = True
            messages_out.append(
                {"role": "system", "content": msg_content, "step_loss_mask": 0}
            )
            continue

        if role == "user":
            messages_out.append(
                {"role": "user", "content": content, "step_loss_mask": 0}
            )
            continue

        if role == "assistant":
            tool_calls_payload = message.get("tool_calls")
            tool_calls = parse_tool_calls(tool_calls_payload) if tool_calls_payload else []
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content if content is not None else "",
                "step_loss_mask": 1,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages_out.append(assistant_msg)
            continue

        if role == "tool_call":
            tool_calls_payload = message.get("tool_calls")
            if tool_calls_payload is None:
                tool_calls_payload = message.get("tool_call") or content
            tool_calls = parse_tool_calls(tool_calls_payload)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("assistant_content") or "",
                "step_loss_mask": 1,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            else:
                assistant_msg["content"] = content
            messages_out.append(assistant_msg)
            continue

        if role == "tool_response":
            messages_out.append(
                {"role": "tool", "content": content, "step_loss_mask": 0}
            )
            continue

        # Fallback: treat unknown roles as assistant context without supervision.
        messages_out.append(
            {"role": "assistant", "content": content, "step_loss_mask": 0}
        )

    if prefix_text and not prefix_applied:
        messages_out.insert(
            0, {"role": "system", "content": prefix_text, "step_loss_mask": 0}
        )

    return {"tools": tools, "messages": messages_out}


def main() -> None:
    args = parse_args()
    records = [
        convert_messages(record, system_prefix=args.system_prefix)
        for record in read_jsonl(args.input)
    ]
    if not records:
        raise SystemExit("No records were produced from the input JSONL.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_format = args.output_format
    if output_format is None:
        suffix = args.output.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            output_format = "jsonl"
        else:
            output_format = "parquet"

    if output_format == "jsonl":
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
    else:
        if pa is not None and pq is not None:
            table = pa.Table.from_pylist(records)
            pq.write_table(table, str(args.output))
        else:
            df = pd.DataFrame.from_records(records)
            df.to_parquet(args.output)


if __name__ == "__main__":
    main()
