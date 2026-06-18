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
from ..retrieval import Retriever


def build_counsel_graph(
    retriever: Retriever,
    *,
    llm: LLMProvider | None = None,
    decomposer: QueryDecomposer | None = None,
    checkpointer=None,
    top_k: int = 5,
):
    """Compile the ATLAS Counsel agent graph.

    checkpointer: any LangGraph checkpointer. If None, the graph still compiles
    but interrupt/resume needs one — callers that use the human-gate must pass
    e.g. MemorySaver() (dev) or a Sqlite/Postgres saver (prod).
    """
    llm = llm or TemplateLLM()
    decomposer = decomposer if decomposer is not None else HeuristicDecomposer()
    nodes = make_nodes(retriever, llm, decomposer, top_k=top_k)

    g = StateGraph(CounselState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "validate")

    g.add_conditional_edges("validate", route_after_validate,
                            {"synthesize": "synthesize", "human_gate": "human_gate"})
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", route_after_verify,
                            {"finalize": "finalize", "synthesize": "synthesize",
                             "human_gate": "human_gate"})
    g.add_conditional_edges("human_gate", route_after_human,
                            {"synthesize": "synthesize", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
