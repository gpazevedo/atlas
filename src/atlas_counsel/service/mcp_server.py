"""MCP server — exposes ATLAS Counsel as tools Buyer Team's orchestrator calls.

The tools are thin wrappers over the SAME `CounselService` the FastAPI app
uses, so behavior is identical across transports. This is the integration
boundary in the architecture diagram: Buyer Team's Strands orchestrator lists
`counsel.ask` / `counsel.brief` as tools and invokes them over MCP.

Run as an MCP stdio server:

    python -m atlas_counsel.service.mcp_server

Buyer Team (or any MCP client) then connects and sees three tools.
"""

from __future__ import annotations

from .core import CounselService

# A single service instance backs all tools (shares one checkpointer, so a
# thread_id returned by `ask` is resumable by `resume`).
_service = CounselService()


def _build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("atlas-counsel")

    @mcp.tool()
    def counsel_ask(question: str) -> dict:
        """Answer a procurement-policy or contract question with citations.

        Returns a dict with status (answered | refused | needs_input), the
        answer text, citations (span ids), and a thread_id. If status is
        needs_input, call counsel_resume with that thread_id."""
        return _service.ask(question).model_dump()

    @mcp.tool()
    def counsel_resume(thread_id: str, action: str, guidance: str = "") -> dict:
        """Resume a paused counsel run that hit the human-gate.

        action is 'steer' (proceed, optionally guided by `guidance`, e.g. a
        document id) or 'decline' (refuse safely)."""
        return _service.resume(
            thread_id, action, guidance=guidance or None
        ).model_dump()

    @mcp.tool()
    def counsel_brief(vendor: str) -> dict:
        """Generate a negotiation pre-brief grounded in the vendor's contract
        and any prior negotiation logs."""
        question = (
            f"Summarize the key contract terms and negotiation precedent for "
            f"{vendor}: service levels, payment terms, liability, and any prior "
            f"negotiation outcomes."
        )
        return _service.ask(question).model_dump()

    return mcp


def main() -> None:
    _build_server().run()


if __name__ == "__main__":
    main()
