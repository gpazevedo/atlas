"""MCP server — exposes ATLAS Counsel as tools Buyer Team's orchestrator calls.

The tools are thin wrappers over the SAME `CounselService` the FastAPI app
uses, so behavior is identical across transports. This is the integration
boundary in the architecture diagram: Buyer Team's Strands orchestrator lists
`counsel.ask` / `counsel.brief` as tools and invokes them over MCP.

Run as an MCP stdio server (local dev):

    uv run python -m atlas_counsel.service.mcp_server

Run as a Streamable HTTP server (deployed):

    uv run python -m atlas_counsel.service.mcp_server --transport streamable-http
"""

from __future__ import annotations

import os

from .core import CounselService
from .tenants import DEFAULT_TENANT

# Shared single instance; the FastAPI app creates its own via build_mcp_server().
_service = CounselService()

_MCP_VALIDATION_ERROR = {"status": "error", "answer": ""}


def build_mcp_server(service: CounselService | None = None):
    """Build a FastMCP server with all tools registered.

    service: if None, uses the module-level singleton. Callers that share a
      TenantRegistry (e.g. the FastAPI app) should inject their service here.
    """
    from mcp.server.fastmcp import FastMCP

    svc = service or _service
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))
    mcp = FastMCP("atlas-counsel", host=host, port=port)

    @mcp.tool()
    def counsel_ask(tenant_id: str, question: str) -> dict:
        """Answer a policy or contract question with citations.

        tenant_id identifies the organization (e.g. 'acme', 'buyer-team').
        Returns a dict with status (answered | refused | needs_input), the
        answer text, citations (span ids), and a thread_id. If status is
        needs_input, call counsel_resume with that thread_id."""
        if not question or not question.strip():
            return {"status": "error", "answer": "question must not be empty"}
        if len(question) > 2000:
            return {"status": "error", "answer": "question too long"}
        try:
            return svc.ask(question, tenant_id=tenant_id).model_dump()
        except ValueError as exc:
            return {"status": "error", "answer": str(exc)}

    @mcp.tool()
    def counsel_resume(tenant_id: str, thread_id: str, action: str,
                       guidance: str = "") -> dict:
        """Resume a paused counsel run that hit the human-gate.

        action is 'steer' (proceed, optionally guided by `guidance`, e.g. a
        document id) or 'decline' (refuse safely). tenant_id must match the
        tenant used in the original counsel_ask call."""
        if action not in ("steer", "decline"):
            return {"status": "error", "answer": "action must be 'steer' or 'decline'"}
        try:
            return svc.resume(
                thread_id, action, guidance=guidance or None, tenant_id=tenant_id,
            ).model_dump()
        except ValueError as exc:
            return {"status": "error", "answer": str(exc)}

    @mcp.tool()
    def counsel_health() -> dict:
        """Deep health check: verifies graph, checkpointer, and retriever."""
        return svc.deep_health()

    @mcp.tool()
    def counsel_brief(tenant_id: str, vendor: str) -> dict:
        """Generate a negotiation pre-brief grounded in the vendor's contract
        and any prior negotiation logs."""
        question = (
            f"Summarize the key contract terms and negotiation precedent for "
            f"{vendor}: service levels, payment terms, liability, and any prior "
            f"negotiation outcomes."
        )
        try:
            return svc.ask(question, tenant_id=tenant_id).model_dump()
        except ValueError as exc:
            return {"status": "error", "answer": str(exc)}

    return mcp


def main() -> None:
    import sys
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "stdio"
    mcp = build_mcp_server()
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
