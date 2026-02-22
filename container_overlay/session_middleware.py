"""
ASGI middleware that extracts X-Session-Id from HTTP headers
and stores it in session_context.session_id_var for the request duration.

Usage in each MCP server's main.py:

    from session_middleware import inject_session_middleware
    # after creating `mcp = FastMCP(...)`
    inject_session_middleware(mcp)
"""

from session_context import session_id_var

_HEADER_NAME = b"x-session-id"


class SessionMiddleware:
    """ASGI middleware: reads X-Session-Id header → sets ContextVar."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        # Proxy attribute access (e.g. .state, .routes) to the wrapped app
        # so FastMCP's run_http_async can access app.state.path etc.
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            sid = None
            for name, value in scope.get("headers", []):
                if name == _HEADER_NAME:
                    sid = value.decode("latin-1")
                    break
            if sid:
                token = session_id_var.set(sid)
                try:
                    await self.app(scope, receive, send)
                finally:
                    session_id_var.reset(token)
                return
        await self.app(scope, receive, send)


def inject_session_middleware(mcp_server):
    """Wrap a FastMCP server's HTTP app with SessionMiddleware.

    Call this BEFORE mcp.run(). It monkey-patches mcp._http_app() so that
    when FastMCP creates the Starlette app, our middleware wraps it.

    For simplicity, we override mcp.run() to inject the middleware.
    """
    _original_run = mcp_server.run

    def patched_run(*args, **kwargs):
        # FastMCP.run(transport="http") internally calls mcp.http_app()
        # We override http_app to wrap with our middleware
        _original_http_app = mcp_server.http_app

        def patched_http_app(*a, **kw):
            app = _original_http_app(*a, **kw)
            return SessionMiddleware(app)

        mcp_server.http_app = patched_http_app
        return _original_run(*args, **kwargs)

    mcp_server.run = patched_run
