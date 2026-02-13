"""Tool filtering for APEX-Agents benchmark.

Filters MCP tools based on task requirements (expected_output type).
For text-only tasks (88% of benchmark), removes all write tools except code_exec,
reducing tool count from 68 to 27 (~60% reduction).

Usage:
    from slime.rollout.tool_filter import filter_tools_for_task

    # tools = list of tool dicts from MCP tools/list
    filtered = filter_tools_for_task(tools, expected_output="message_in_console")
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# --- Tool classification ---

# Name prefixes that indicate WRITE operations
WRITE_PREFIXES = (
    "create_", "delete_", "edit_", "add_", "modify_",
    "send_", "reply_", "forward_", "post_", "insert_",
    "update_", "apply_",
)

# Name prefixes that indicate READ operations
READ_PREFIXES = (
    "list_", "read_", "get_", "search_",
)

# Tools that don't follow naming conventions — manually classified
MANUAL_CLASSIFICATION = {
    # Word (Documents) server — dual-purpose tools, classified as WRITE
    # because they CAN modify documents
    "page_margins": "write",
    "page_orientation": "write",
    "header_footer": "write",
    "comments": "write",
    # Code execution — always needed (computations), classified separately
    "code_exec": "code_exec",
}

# --- Server-to-tool mapping (for output-type routing) ---
# Maps expected_output types to which servers' WRITE tools to keep

# Server identification: we detect which server a tool belongs to
# based on known tool-to-server mapping
TOOL_TO_SERVER = {
    # Calendar
    "list_events": "calendar", "read_event": "calendar",
    "create_event": "calendar", "update_event": "calendar", "delete_event": "calendar",
    # Chat
    "list_channels": "chat", "get_channel_history": "chat", "get_thread_replies": "chat",
    "get_user_profile": "chat", "get_users": "chat", "post_message": "chat",
    "reply_to_thread": "chat", "add_reaction": "chat", "delete_post": "chat",
    # Code
    "code_exec": "code",
    # Documents (Word)
    "create_document": "documents", "delete_document": "documents",
    "get_document_overview": "documents", "read_document_content": "documents",
    "add_content_text_doc": "documents", "edit_content_text": "documents",
    "delete_content_text": "documents", "add_image_doc": "documents",
    "modify_image_doc": "documents", "apply_formatting": "documents",
    "page_margins": "documents", "page_orientation": "documents",
    "header_footer": "documents", "comments": "documents",
    # Filesystem
    "list_files": "filesystem", "read_image_file": "filesystem",
    # Mail
    "list_mails": "mail", "read_mail": "mail", "search_mail": "mail",
    "send_mail": "mail", "reply_mail": "mail", "reply_all_mail": "mail", "forward_mail": "mail",
    # PDFs
    "create_pdf": "pdfs", "read_pdf_pages": "pdfs", "search_pdf": "pdfs",
    "read_page_as_image": "pdfs",
    # Presentations (PowerPoint)
    "create_deck": "presentations", "delete_deck": "presentations",
    "add_slide": "presentations", "edit_slides": "presentations",
    "insert_chart": "presentations", "insert_table": "presentations",
    "add_shape": "presentations", "read_slides": "presentations",
    "read_completedeck": "presentations", "read_individualslide": "presentations",
    # Spreadsheets (Excel)
    "create_spreadsheet": "spreadsheets", "delete_spreadsheet": "spreadsheets",
    "read_tab": "spreadsheets", "read_csv": "spreadsheets",
    "list_tabs_in_spreadsheet": "spreadsheets", "add_tab": "spreadsheets",
    "delete_tab": "spreadsheets", "edit_spreadsheet": "spreadsheets",
    "delete_content_cell": "spreadsheets", "create_chart": "spreadsheets",
}

# Note: some tools like "read_image", "add_content_text", "add_image", "modify_image"
# exist on multiple servers. We handle these by classification (read vs write)
# rather than server mapping.

# Which servers' WRITE tools to keep for each output type
OUTPUT_WRITE_SERVERS: dict[str, set[str]] = {
    "message_in_console":       {"code"},
    "make_new_doc":             {"code", "documents"},
    "edit_existing_doc":        {"code", "documents"},
    "make_new_sheet":           {"code", "spreadsheets"},
    "edit_existing_sheet":      {"code", "spreadsheets"},
    "make_new_slide_deck":      {"code", "presentations"},
    "edit_existing_slide_deck": {"code", "presentations"},
}


def classify_tool(tool_name: str) -> str:
    """Classify a tool as 'read', 'write', or 'code_exec'.

    Args:
        tool_name: The tool's function name.

    Returns:
        'read', 'write', or 'code_exec'
    """
    # Check manual overrides first
    if tool_name in MANUAL_CLASSIFICATION:
        return MANUAL_CLASSIFICATION[tool_name]

    # Check prefixes
    name_lower = tool_name.lower()
    if name_lower.startswith(READ_PREFIXES):
        return "read"
    if name_lower.startswith(WRITE_PREFIXES):
        return "write"

    # Default: treat unknown as write (safer to include than exclude)
    logger.warning("Unknown tool '%s', defaulting to 'write' classification", tool_name)
    return "write"


def get_tool_server(tool_name: str) -> str | None:
    """Get the server a tool belongs to, if known.

    Falls back to None for tools not in the static mapping (e.g. tools
    with duplicate names across servers like 'read_image').
    """
    return TOOL_TO_SERVER.get(tool_name)


def filter_tools_for_task(
    tools: list[dict[str, Any]],
    expected_output: str,
) -> list[dict[str, Any]]:
    """Filter tools based on task's expected_output type.

    Keeps:
    - All READ tools (always needed)
    - code_exec (always needed for computation)
    - WRITE tools only for servers relevant to the output type

    Args:
        tools: List of tool schema dicts (must have 'name' or 'function.name' field).
        expected_output: The task's expected_output field value.

    Returns:
        Filtered list of tool schema dicts.
    """
    allowed_write_servers = OUTPUT_WRITE_SERVERS.get(expected_output)

    # If output type is unknown, return all tools (no filtering)
    if allowed_write_servers is None:
        logger.warning(
            "Unknown expected_output '%s', returning all %d tools unfiltered",
            expected_output, len(tools),
        )
        return tools

    filtered = []
    removed = []

    for tool in tools:
        # Extract tool name from schema (handle both flat and nested formats)
        name = tool.get("name") or tool.get("function", {}).get("name", "")

        classification = classify_tool(name)

        if classification == "read":
            # Always keep read tools
            filtered.append(tool)
        elif classification == "code_exec":
            # Always keep code_exec
            filtered.append(tool)
        elif classification == "write":
            # Keep write tools only if their server is in the allowed set
            server = get_tool_server(name)
            if server is not None and server in allowed_write_servers:
                filtered.append(tool)
            elif server is None:
                # Unknown server — keep if ANY write server is allowed
                # (conservative: don't accidentally remove needed tools)
                filtered.append(tool)
            else:
                removed.append(name)
        else:
            # Unknown classification, keep to be safe
            filtered.append(tool)

    if removed:
        logger.info(
            "Filtered tools for '%s': %d -> %d (removed %d: %s)",
            expected_output, len(tools), len(filtered), len(removed),
            ", ".join(removed[:10]) + ("..." if len(removed) > 10 else ""),
        )

    return filtered
