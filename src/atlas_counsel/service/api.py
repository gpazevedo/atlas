"""FastAPI runtime around the CounselService.

Endpoints (all thin wrappers over the service — no logic here):

  POST /ask            {question}             -> AskResult
  POST /resume         {thread_id, action,..} -> AskResult
  GET  /health                                -> {status}
  WS   /ws/ask         stream node-by-node progress, then the result

The WebSocket demonstrates the "real-time / streaming AI experiences" the JD
asks for: it streams the graph's state updates as each node completes, then a
terminal frame with the final result (or a needs_input frame at the gate).
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .core import AskResult, AskStatus, CounselService


class AskRequest(BaseModel):
    question: str
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    action: str  # "steer" | "decline"
    guidance: str | None = None


def create_app(service: CounselService | None = None) -> FastAPI:
    service = service or CounselService()
    app = FastAPI(title="ATLAS Counsel", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResult)
    def ask(req: AskRequest) -> AskResult:
        return service.ask(req.question, thread_id=req.thread_id)

    @app.post("/resume", response_model=AskResult)
    def resume(req: ResumeRequest) -> AskResult:
        return service.resume(req.thread_id, req.action, guidance=req.guidance)

    @app.websocket("/ws/ask")
    async def ws_ask(ws: WebSocket) -> None:
        await ws.accept()
        try:
            req = await ws.receive_json()
            question = req["question"]
            thread_id = req.get("thread_id")
            # Stream the graph node-by-node, then send the terminal frame.
            async for frame in service.astream(question, thread_id=thread_id):
                await ws.send_json(frame)
        except WebSocketDisconnect:
            return

    return app


# Convenience for `uvicorn atlas_counsel.service.api:app`
app = create_app()
