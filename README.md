# ATLAS Counsel

Citation-grounded agentic RAG over a synthetic procurement corpus. A
complementary knowledge service to the Buyer Team platform: a multi-agent
(LangGraph) RAG pipeline with hybrid retrieval, reranking, citation grounding,
and a measured eval harness.

This first slice ships the **corpus generator + golden eval set** plus a
**hybrid retrieval layer** (dense + sparse, RRF fusion) over it.

## Quickstart

```bash
pip install -e ".[dev]"
atlas-corpus                          # generate the corpus
pytest                                # 14 tests, no external services

# hybrid retrieval, offline (in-memory, no Qdrant):
python -m atlas_counsel.ingest --dry-run

# hybrid retrieval against real Qdrant:
docker compose up -d
pip install -e ".[qdrant]"
python -m atlas_counsel.ingest --url http://localhost:6333
pytest tests/test_qdrant_integration.py -v
```

## Retrieval design

- **Hybrid, native.** One collection per chunk holds a dense vector and a
  sparse (lexical) vector. Queries fuse both channels with Reciprocal Rank
  Fusion (RRF). Sparse rescues exact tokens — `$25,000`, `99.5%` — that a
  semantic dense model smears together. `tests/test_retrieval.py` proves a
  real fusion win on a query where the channels disagree.
- **Vector-space safety by construction.** Each embedding provider declares a
  `space_id`; the collection name is derived from it (`counsel_bge-m3`,
  `counsel_titan-v2`). A local-dev index and a Bedrock-prod index are
  physically separate collections — you cannot query one space against the
  other by accident.
- **Provider abstraction.** `EmbeddingProvider` is a Protocol yielding dense +
  sparse per text. `HashingEmbedder` is a deterministic offline stand-in for
  CI; real bge-m3 (dev) and Titan (prod) providers implement the same
  interface. Dev/prod is a config swap, not a code branch.
- **Citations survive retrieval.** Chunks map 1:1 to spans and carry the
  `span_id` through Qdrant payloads, so a result always cites `POL-001#S1`.

## What the corpus contains

- **8 documents / 24 citable spans**: procurement policies, vendor MSAs, a
  negotiation log. Each span has a stable id (e.g. `POL-001#S1`) that the
  golden set and, later, the retriever cite directly.
- **8 golden Q/A items** spanning three answer types:
  - `grounded` — answer lives in specific span(s)
  - `multi_hop` — requires combining spans across documents
  - `unanswerable` — answer is **absent by design**; the agent must refuse

### Planted hard cases (each tagged, so eval can slice on them)

| Trap | Where | Tests |
| ------ | ------- | ------- |
| Contradiction | AcmeCloud 99.9% vs NorthLink 99.5% uptime | cross-document reasoning |
| Threshold precision | POL-001 ($50k) vs near-duplicate POL-002 ($25k) | reranker precision |
| Unanswerable | supplier-gift question (Q-006) | refuse-if-ungrounded |
| Anti-splitting | $120k split question (Q-007) | policy reasoning |

## Reproducibility

The corpus is fully seeded and template-driven — no LLM calls, no network.
Same seed in, byte-identical corpus out. That is what makes the downstream
eval numbers trustworthy.

## Evaluation harness

Measured before any agent exists, so every later change is regression-checked.

```bash
python -m atlas_counsel.eval        # aggregate + per-tag breakdown
python -m atlas_counsel.eval --ab   # A/B two embedding configs
```

- **Retrieval metrics** (no LLM, exact): hit@k, context recall, context
  precision (AP-style), MRR — scored against the golden set's known
  `supporting_span_ids`.
- **Answer metrics** behind an `LLMJudge` protocol: a deterministic
  `HeuristicJudge` runs in CI; inject an LLM-backed judge locally for
  faithfulness / answer-relevancy at full fidelity.
- **Refuse-if-ungrounded** is a first-class scored dimension. Refusal is
  decided by *lexical grounding overlap*, not retrieval score — RRF fusion
  scores are rank-based and carry no absolute relevance signal, so a score
  threshold cannot tell answerable from unanswerable.
- **Per-tag slicing** reports each planted hard case (threshold-precision,
  contradiction, unanswerable…) separately, proving the hard cases are
  handled — not just the easy ones.
- **A/B table** across two retrievers (two embedding providers) is the
  harness's native output — the artifact the provider abstraction exists for.
- **Langfuse** export is optional (`pip install -e ".[langfuse]"`, set
  `LANGFUSE_*` env); a silent no-op otherwise so CI never depends on it.
- A **regression gate** (`tests/test_eval.py`) locks in current aggregate
  quality so future changes fail loudly instead of degrading silently.

## Advanced retrieval: rerank + query decomposition

Two optional stages compose around the base hybrid retriever via
`RetrievalPipeline`, each independently toggleable so the eval harness can
attribute any metric change to a specific stage (ablation):

```text
query --(decompose?)--> sub-queries --retrieve+merge--> --(rerank?)--> top_k
```

- **Reranking.** `Reranker` protocol. `TokenInteractionReranker` is a
  deterministic offline proxy for CI (query-doc token interaction + a
  proportional phrase-adjacency bonus, scored without ever consulting golden
  spans). `CrossEncoderReranker` wraps bge-reranker-v2-m3 for production
  (`pip install -e ".[rerank]"`).
- **Query decomposition.** Conditional by design — a query is split only when
  it names >= 2 known entities AND carries a comparison cue, so simple queries
  pass through untouched (tested). Multi-hop questions retrieve per-entity and
  merge, recovering the starved side.

### Honest ablation

```bash
python -m atlas_counsel.ablation
```

On the current small corpus, first-stage hybrid nearly saturates retrieval, so
the **offline lexical proxies show no net gain** (rerank slightly lowers
precision; decomposition is net-neutral). This is reported, not hidden. The
stages ship as correct, unit-tested infrastructure; the production
cross-encoder and an LLM decomposer are the implementations expected to win,
and this same harness is how you confirm it locally. The proxies are
deliberately NOT tuned toward the golden spans to manufacture an improvement.

## The agent: LangGraph StateGraph

The orchestration layer. A compiled `StateGraph` with conditional routing, a
checkpointed human-gate, and a bounded verify/retry loop. Replaces the eval
stub answerer and the PR3 lexical-refusal proxy with real graph logic.

```bash
python -m atlas_counsel.agent --q "who approves a $60,000 purchase?"
python -m atlas_counsel.agent --q "policy on supplier gifts?" --decline
```

Flow:

```
plan -> retrieve -> validate
validate --grounded--> synthesize        --insufficient--> human_gate
synthesize -> verify
verify --pass--> finalize -> END
       --unfaithful & attempts<MAX--> synthesize   (bounded retry)
       --unfaithful & exhausted--> human_gate
human_gate --steer--> synthesize         --decline--> finalize(refuse)
```

- **Structured outputs / citation grounding.** `synthesize` emits a Pydantic
  `DraftAnswer` whose every `Claim` carries a `span_id`; `verify` rejects any
  claim citing a span that wasn't retrieved. A hallucinated citation is caught,
  not shipped — proven by `test_hallucination_is_bounded_and_escalates`.
- **Human-assisted decisions.** `human_gate` uses LangGraph `interrupt()`; the
  run pauses, the caller resumes with `Command(resume=...)` to steer or
  decline. State survives the pause via the checkpointer.
- **Bounded loops.** The verify→synthesize retry is capped at `MAX_ATTEMPTS`,
  then escalates — no unbounded LLM spinning.
- **Provider abstraction.** `LLMProvider` protocol with an offline
  `TemplateLLM` for CI (cites only retrieved spans, never fabricates) and real
  Ollama/Bedrock injected locally. Checkpointer is injected too: MemorySaver
  (dev) or Sqlite/Postgres (prod).
- **Measured, not asserted.** Run over the golden set the agent matches the
  stub: correct refusal on the unanswerable item, 7/7 answerable not wrongly
  refused. The one citation miss (Q-003) is the known PR4 retrieval weakness a
  real cross-encoder addresses — left visible, not hidden.

### Buyer Team integration (next)

The compiled graph is exposed as an MCP tool (`counsel.ask`, `counsel.brief`)
so Buyer Team's Strands orchestrator calls it as one tool among its own. That
PR adds the MCP server + FastAPI/WebSocket runtime around this graph.

## Runtime: FastAPI + MCP

The graph exposed over two transports, both thin wrappers over one
`CounselService` — so HTTP and MCP behave identically (proven by
`test_mcp_and_http_agree`).

```bash
uv sync --extra service
uv run uvicorn atlas_counsel.service.api:app    # HTTP on :8000
uv run python -m atlas_counsel.service.mcp_server      # MCP stdio server
```

**REST**

- `POST /ask {question}` → `{status, thread_id, answer, citations[]}`
- `POST /resume {thread_id, action, guidance?}` → same shape
- `WS /ws/ask` → streams `{event:"node", node}` per step, then a terminal
  `result` / `needs_input` frame (the JD's streaming requirement)

**Interrupt across stateless calls.** The agent pauses at the human-gate via
LangGraph `interrupt()`, but HTTP/MCP are request/response. So a paused run
returns `status="needs_input"` plus a `thread_id`; a second `/resume` call
continues it. The checkpointer carries state across those two independent
calls — the core integration design.

**MCP tools** (what Buyer Team calls): `counsel_ask`, `counsel_resume`,
`counsel_brief`. See `buyer-team-mcp.example.json` for the client entry. The
Strands orchestrator lists these alongside its own tools and invokes them over
MCP — this is the integration boundary, with neither system absorbing the other.

## Layout

```text
src/atlas_counsel/corpus/
  models.py      # Pydantic schema — the citation contract
  generator.py   # deterministic content + planted hard cases
  writer.py      # serialize to markdown + JSONL
tests/           # determinism, integrity, hard-case guards
data/            # generated (gitignored)
```

## Evaluation harness

Measured before any agent exists, so every later change is regression-checked.

```bash
python -m atlas_counsel.eval        # aggregate + per-tag breakdown
python -m atlas_counsel.eval --ab   # A/B two embedding configs
```

- **Retrieval metrics** (no LLM, exact): hit@k, context recall, context
  precision (AP-style), MRR — scored against the golden set's known
  `supporting_span_ids`.
- **Answer metrics** behind an `LLMJudge` protocol: a deterministic
  `HeuristicJudge` runs in CI; inject an LLM-backed judge locally for
  faithfulness / answer-relevancy at full fidelity.
- **Refuse-if-ungrounded** is a first-class scored dimension. Refusal is
  decided by *lexical grounding overlap*, not retrieval score — RRF fusion
  scores are rank-based and carry no absolute relevance signal, so a score
  threshold cannot tell answerable from unanswerable.
- **Per-tag slicing** reports each planted hard case (threshold-precision,
  contradiction, unanswerable…) separately, proving the hard cases are
  handled — not just the easy ones.
- **A/B table** across two retrievers (two embedding providers) is the
  harness's native output — the artifact the provider abstraction exists for.
- **Langfuse** export is optional (`pip install -e ".[langfuse]"`, set
  `LANGFUSE_*` env); a silent no-op otherwise so CI never depends on it.
- A **regression gate** (`tests/test_eval.py`) locks in current aggregate
  quality so future changes fail loudly instead of degrading silently.

## Advanced retrieval: rerank + query decomposition

Two optional stages compose around the base hybrid retriever via
`RetrievalPipeline`, each independently toggleable so the eval harness can
attribute any metric change to a specific stage (ablation):

```text
query --(decompose?)--> sub-queries --retrieve+merge--> --(rerank?)--> top_k
```

- **Reranking.** `Reranker` protocol. `TokenInteractionReranker` is a
  deterministic offline proxy for CI (query-doc token interaction + a
  proportional phrase-adjacency bonus, scored without ever consulting golden
  spans). `CrossEncoderReranker` wraps bge-reranker-v2-m3 for production
  (`pip install -e ".[rerank]"`).
- **Query decomposition.** Conditional by design — a query is split only when
  it names >= 2 known entities AND carries a comparison cue, so simple queries
  pass through untouched (tested). Multi-hop questions retrieve per-entity and
  merge, recovering the starved side.

### Honest ablation

```bash
python -m atlas_counsel.ablation
```

On the current small corpus, first-stage hybrid nearly saturates retrieval, so
the **offline lexical proxies show no net gain** (rerank slightly lowers
precision; decomposition is net-neutral). This is reported, not hidden. The
stages ship as correct, unit-tested infrastructure; the production
cross-encoder and an LLM decomposer are the implementations expected to win,
and this same harness is how you confirm it locally. The proxies are
deliberately NOT tuned toward the golden spans to manufacture an improvement.

## The agent: LangGraph StateGraph

The orchestration layer. A compiled `StateGraph` with conditional routing, a
checkpointed human-gate, and a bounded verify/retry loop. Replaces the eval
stub answerer and the PR3 lexical-refusal proxy with real graph logic.

```bash
python -m atlas_counsel.agent --q "who approves a $60,000 purchase?"
python -m atlas_counsel.agent --q "policy on supplier gifts?" --decline
```

Flow:

```text
plan -> retrieve -> validate
validate --grounded--> synthesize        --insufficient--> human_gate
synthesize -> verify
verify --pass--> finalize -> END
       --unfaithful & attempts<MAX--> synthesize   (bounded retry)
       --unfaithful & exhausted--> human_gate
human_gate --steer--> synthesize         --decline--> finalize(refuse)
```

- **Structured outputs / citation grounding.** `synthesize` emits a Pydantic
  `DraftAnswer` whose every `Claim` carries a `span_id`; `verify` rejects any
  claim citing a span that wasn't retrieved. A hallucinated citation is caught,
  not shipped — proven by `test_hallucination_is_bounded_and_escalates`.
- **Human-assisted decisions.** `human_gate` uses LangGraph `interrupt()`; the
  run pauses, the caller resumes with `Command(resume=...)` to steer or
  decline. State survives the pause via the checkpointer.
- **Bounded loops.** The verify→synthesize retry is capped at `MAX_ATTEMPTS`,
  then escalates — no unbounded LLM spinning.
- **Provider abstraction.** `LLMProvider` protocol with an offline
  `TemplateLLM` for CI (cites only retrieved spans, never fabricates) and real
  Ollama/Bedrock injected locally. Checkpointer is injected too: MemorySaver
  (dev) or Sqlite/Postgres (prod).
- **Measured, not asserted.** Run over the golden set the agent matches the
  stub: correct refusal on the unanswerable item, 7/7 answerable not wrongly
  refused. The one citation miss (Q-003) is the known PR4 retrieval weakness a
  real cross-encoder addresses — left visible, not hidden.

### Buyer Team integration (next)

The compiled graph is exposed as an MCP tool (`counsel.ask`, `counsel.brief`)
so Buyer Team's Strands orchestrator calls it as one tool among its own. That
PR adds the MCP server + FastAPI/WebSocket runtime around this graph.

## Runtime: FastAPI + MCP

The graph exposed over two transports, both thin wrappers over one
`CounselService` — so HTTP and MCP behave identically (proven by
`test_mcp_and_http_agree`).

```bash
uv sync --extra service
uv run uvicorn atlas_counsel.service.api:app    # HTTP on :8000
uv run python -m atlas_counsel.service.mcp_server      # MCP stdio server
```

**REST**

- `POST /ask {question}` → `{status, thread_id, answer, citations[]}`
- `POST /resume {thread_id, action, guidance?}` → same shape
- `WS /ws/ask` → streams `{event:"node", node}` per step, then a terminal
  `result` / `needs_input` frame (the JD's streaming requirement)

**Interrupt across stateless calls.** The agent pauses at the human-gate via
LangGraph `interrupt()`, but HTTP/MCP are request/response. So a paused run
returns `status="needs_input"` plus a `thread_id`; a second `/resume` call
continues it. The checkpointer carries state across those two independent
calls — the core integration design.

**MCP tools** (what Buyer Team calls): `counsel_ask`, `counsel_resume`,
`counsel_brief`. See `buyer-team-mcp.example.json` for the client entry. The
Strands orchestrator lists these alongside its own tools and invokes them over
MCP — this is the integration boundary, with neither system absorbing the other.

## Layout

```text
src/atlas_counsel/
  corpus/          # synthetic corpus generator + golden set (PR1)
  chunking.py      # span -> chunk, preserving citation ids
  embeddings.py    # EmbeddingProvider protocol + offline HashingEmbedder
  retrieval.py     # Retriever protocol, RRF fusion, in-memory hybrid
  qdrant_store.py  # production Qdrant hybrid retriever (named vectors + RRF)
  ingest.py        # CLI: build -> chunk -> index
  eval/            # metrics, judge protocol, runner, A/B report, Langfuse hook
  rerank.py        # Reranker protocol, offline proxy + cross-encoder
  decompose.py     # conditional query decomposition
  pipeline.py      # composed decompose->retrieve->merge->rerank
  ablation.py      # CLI: measure each stage's effect
  agent/           # LangGraph StateGraph: state, schemas, nodes, graph, llm
  service/         # CounselService + FastAPI api + MCP server (one logic, two transports)
tests/             # corpus + retrieval + (skipped) Qdrant integration
docker-compose.yml # local Qdrant
```
