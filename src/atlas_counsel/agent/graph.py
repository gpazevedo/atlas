"""Graph assembly.

Wires the nodes and conditional edges into a compiled LangGraph StateGraph.
The checkpointer is injected so dev uses in-memory and prod uses Sqlite/Postgres
without changing the topology.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .llm import LLMProvider, TemplateLLM
from .nodes import (
    make_nodes,
    route_after_human,
    route_after_validate,
    route_after_verify,
)
from .state import CounselState
from ..decompose import HeuristicDecomposer, QueryDecomposer
from ..memory.store import MemoryStore
from ..retrieval import Retriever


def build_counsel_graph(
    retriever: Retriever,
    *,
    llm: LLMProvider | None = None,
    decomposer: QueryDecomposer | None = None,
    checkpointer=None,
    top_k: int = 5,
    hitl_enabled: bool = True,
    memory_store: MemoryStore | None = None,
):
    """Compile the ATLAS Counsel agent graph.

    checkpointer: any LangGraph checkpointer. If None, the graph still compiles
    but interrupt/resume needs one — callers that use the human-gate must pass
    e.g. MemorySaver() (dev) or a Sqlite/Postgres saver (prod).

    hitl_enabled: when False, the agent refuses instead of pausing at the
    human-gate. Use for unattended / automated deployments.

    memory_store: optional multi-tier memory store. When provided, the graph
    gains load_memory (before plan) and save_memory (after finalize) nodes,
    enabling cross-session recall.
    """
    llm = llm or TemplateLLM()
    decomposer = decomposer if decomposer is not None else HeuristicDecomposer()
    nodes = make_nodes(retriever, llm, decomposer, top_k=top_k,
                       hitl_enabled=hitl_enabled, memory_store=memory_store)

    g = StateGraph(CounselState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    if memory_store is not None:
        g.add_edge(START, "load_memory")
        g.add_edge("load_memory", "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "validate")

        g.add_conditional_edges("validate", route_after_validate,
                                {"synthesize": "synthesize", "gap_analyze": "gap_analyze",
                                 "human_gate": "human_gate", "finalize": "finalize"})
        g.add_edge("gap_analyze", "retrieve")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", route_after_verify,
                                {"finalize": "finalize", "synthesize": "synthesize",
                                 "human_gate": "human_gate"})
        g.add_conditional_edges("human_gate", route_after_human,
                                {"synthesize": "synthesize", "finalize": "finalize"})
        g.add_edge("finalize", "save_memory")
        g.add_edge("save_memory", END)
    else:
        g.add_edge(START, "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "validate")

        g.add_conditional_edges("validate", route_after_validate,
                                {"synthesize": "synthesize", "gap_analyze": "gap_analyze",
                                 "human_gate": "human_gate", "finalize": "finalize"})
        g.add_edge("gap_analyze", "retrieve")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", route_after_verify,
                                {"finalize": "finalize", "synthesize": "synthesize",
                                 "human_gate": "human_gate"})
        g.add_conditional_edges("human_gate", route_after_human,
                                {"synthesize": "synthesize", "finalize": "finalize"})
        g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
