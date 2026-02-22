# Multi-session overlay for canonical-delivery images.
# Adds session-aware path resolution and session management endpoints.
#
# Build:  docker build --build-arg BASE_IMAGE=canonical-delivery:task_xxx -t canonical-ms:task_xxx .
# The same overlay works on all 30 task images (shared code, different data).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# ── Shared session modules (importable by all MCP servers) ───────────────────
COPY session_context.py    /app/tools/session_context.py
COPY session_middleware.py  /app/tools/session_middleware.py

# Symlink into each MCP server's directory so "import session_context" works
# regardless of the CWD when the server starts.
RUN for svc in calendar chat code_execution excel filesystem mail pdfs powerpoint word; do \
        ln -sf /app/tools/session_context.py   "/app/tools/mcp_servers/${svc}/mcp_servers/session_context.py" ; \
        ln -sf /app/tools/session_middleware.py "/app/tools/mcp_servers/${svc}/mcp_servers/session_middleware.py" ; \
    done

# ── Patch: filesystem_server path_utils (session-aware get_fs_root) ──────────
COPY patches/filesystem_path_utils.py \
     /app/tools/mcp_servers/filesystem/mcp_servers/filesystem_server/utils/path_utils.py

# The same path_utils.py is shared by excel, pdfs, powerpoint, word
COPY patches/filesystem_path_utils.py \
     /app/tools/mcp_servers/excel/mcp_servers/sheets_server/utils/path_utils.py
COPY patches/filesystem_path_utils.py \
     /app/tools/mcp_servers/pdfs/mcp_servers/pdf_server/utils/path_utils.py
COPY patches/filesystem_path_utils.py \
     /app/tools/mcp_servers/powerpoint/mcp_servers/slides_server/utils/path_utils.py
COPY patches/filesystem_path_utils.py \
     /app/tools/mcp_servers/word/mcp_servers/docs_server/utils/path_utils.py

# ── Patch: calendar config + path ────────────────────────────────────────────
COPY patches/calendar_config.py \
     /app/tools/mcp_servers/calendar/mcp_servers/calendar_server/utils/config.py
COPY patches/calendar_path.py \
     /app/tools/mcp_servers/calendar/mcp_servers/calendar_server/utils/path.py

# ── Patch: chat config + path ────────────────────────────────────────────────
COPY patches/chat_config.py \
     /app/tools/mcp_servers/chat/mcp_servers/chat_server/utils/config.py
COPY patches/chat_path.py \
     /app/tools/mcp_servers/chat/mcp_servers/chat_server/utils/path.py

# ── Patch: mail config + path ────────────────────────────────────────────────
COPY patches/mail_config.py \
     /app/tools/mcp_servers/mail/mcp_servers/mail_server/utils/config.py
COPY patches/mail_path.py \
     /app/tools/mcp_servers/mail/mcp_servers/mail_server/utils/path.py

# ── Patch: code-execution tools (session-aware working_dir) ──────────────────
COPY patches/code_exec_tools.py \
     /app/tools/mcp_servers/code_execution/mcp_servers/code_execution_server/tools/code_exec.py

# ── Patch: session management router (new endpoints on the runner) ───────────
RUN mkdir -p /app/tools/runner/sessions
COPY patches/session_router.py /app/tools/runner/sessions/router.py
RUN echo 'from .router import router' > /app/tools/runner/sessions/__init__.py

# ── Patch: runner main.py to include session router ──────────────────────────
# Insert import + include after the existing gateway_router line.
RUN sed -i '/^app\.include_router(gateway_router)/a\from .sessions import router as session_router\napp.include_router(session_router)' \
    /app/tools/runner/main.py

# ── Create session directories skeleton ──────────────────────────────────────
RUN mkdir -p /filesystem/sessions /.apps_data/sessions && \
    chown root:workspace /filesystem/sessions 2>/dev/null || true && \
    chmod 2770 /filesystem/sessions 2>/dev/null || true && \
    chown runner:runner /.apps_data/sessions 2>/dev/null || true && \
    chmod 0711 /.apps_data/sessions 2>/dev/null || true

# ── Inject session middleware into each MCP server's main.py ─────────────────
# Add two lines at the end of each main.py (before the if __name__ block):
#   from session_middleware import inject_session_middleware
#   inject_session_middleware(mcp)
RUN for main_py in \
        /app/tools/mcp_servers/calendar/mcp_servers/calendar_server/main.py \
        /app/tools/mcp_servers/chat/mcp_servers/chat_server/main.py \
        /app/tools/mcp_servers/code_execution/mcp_servers/code_execution_server/main.py \
        /app/tools/mcp_servers/excel/mcp_servers/sheets_server/main.py \
        /app/tools/mcp_servers/filesystem/mcp_servers/filesystem_server/main.py \
        /app/tools/mcp_servers/mail/mcp_servers/mail_server/main.py \
        /app/tools/mcp_servers/pdfs/mcp_servers/pdf_server/main.py \
        /app/tools/mcp_servers/powerpoint/mcp_servers/slides_server/main.py \
        /app/tools/mcp_servers/word/mcp_servers/docs_server/main.py \
    ; do \
        sed -i '/^if __name__/i from session_middleware import inject_session_middleware\ninject_session_middleware(mcp)\n' "$main_py" ; \
    done
