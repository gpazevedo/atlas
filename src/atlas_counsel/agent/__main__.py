"""Run the ATLAS Counsel agent interactively (offline providers).

    python -m atlas_counsel.agent --q "who approves a \$60,000 purchase?"

At a human-gate the run pauses; pass --decline or --steer DOC to resume.
"""
from __future__ import annotations

import argparse
import warnings

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ..chunking import chunk_corpus
from ..corpus import build_corpus
from ..embeddings import HashingEmbedder
from ..retrieval import InMemoryHybridRetriever
from . import build_counsel_graph


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="question")
    ap.add_argument("--decline", action="store_true", help="auto-decline at gate")
    ap.add_argument("--steer", default=None, help="doc hint to steer toward at gate")
    args = ap.parse_args()

    r = InMemoryHybridRetriever(HashingEmbedder())
    r.index(chunk_corpus(build_corpus()))
    graph = build_counsel_graph(r, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "cli"}}

    out = graph.invoke({"question": args.q}, cfg)
    if "__interrupt__" in out:
        reason = out["__interrupt__"][0].value["reason"]
        print(f"[human-gate] paused: {reason}")
        if args.steer:
            out = graph.invoke(Command(resume={"action": "steer", "guidance": args.steer}), cfg)
        else:
            out = graph.invoke(Command(resume={"action": "decline"}), cfg)

    ans = out["answer"]
    print(f"\nrefused={ans.refused}  attempts={ans.attempts}  escalated={ans.escalated}")
    print("citations:", ans.citations)
    print("\n" + ans.text)


if __name__ == "__main__":
    main()
