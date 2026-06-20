# ATLAS Counsel

**Citation-grounded agentic RAG over a synthetic procurement corpus.** A LangGraph
agent that answers procurement-policy and contract questions, grounds every claim
in a retrievable source span, refuses when the corpus doesn't cover the question,
and pauses for a human when it isn't sure — exposed to a separate platform
([Buyer Team](#buyer-team-integration)) as a single MCP tool.

It is built to be *measured*: an evaluation harness existed before the agent did,
so every change is regression-checked rather than asserted. The whole thing runs
offline and reproducibly in CI (`114` tests, no external services), then swaps to
real models and Qdrant in production via config, not code changes.

```bash
uv sync --extra dev
atlas-corpus                  # generate the corpus
uv run pytest                 # 113 tests offline (Qdrant integration test skips)
uv run python -m atlas_counsel.agent --q "who approves a $60,000 purchase?"
```

---

## What it does, in one pass

```text
question
   │
   ▼
 plan ─▶ retrieve ─▶ validate ──grounded──▶ synthesize ─▶ verify ──pass──▶ finalize
              ▲          │                       ▲            │
              │          │ insufficient          │ unfaithful │ (bounded retry)
              │          ▼                        └────────────┘
              └── gap_analyze (re-retrieve,            │ exhausted
                  bounded ×2)                          ▼
                                              human-gate (interrupt / resume)
                                                 steer ─▶ synthesize
                                                 decline ─▶ refuse
```

Three things make this more than a retrieval demo:

- **Citations are enforced by the type system, not by prompt politeness.** Every
  `Claim` is a Pydantic model that *cannot exist* without a `span_id`, and a
  `verify` node rejects any claim citing a span that wasn't retrieved. A
  hallucinated citation is caught before it ships — proven by a test, not hoped for.
- **Refusal is a first-class outcome.** An out-of-corpus question (e.g. "what's
  our supplier-gift policy?") is refused rather than answered with a confident
  guess. The decision is made on *lexical grounding overlap*, not retrieval score,
  because RRF fusion scores are rank-based and carry no absolute relevance signal.
- **Hard cases are planted and tagged**, so the eval can prove they're handled —
  a contradiction between two contracts, two near-duplicate policies that differ
  by a single dollar threshold, and an anti-splitting policy-reasoning trap.

---

## Architecture

![ATLAS Counsel Architecture](images/atlas-counsel-architecture.svg)

![LangGraph Implementation](images/atlas-counsel-langgraph.svg)

---

## The corpus and golden set

The corpus is **fully seeded and template-driven — no LLM calls, no network**.
Same seed in, byte-identical corpus out. That determinism is what makes the
downstream eval numbers trustworthy and keeps the repo public-safe.

- **8 documents / 24 citable spans**: procurement policies, vendor MSAs, a
  negotiation log. Each span carries a stable id (e.g. `POL-001#S1`) that the
  golden set and the retriever cite directly — the citation contract holds end
  to end.
- **8 golden Q/A items** across three answer types: `grounded` (answer lives in
  specific spans), `multi_hop` (combine spans across documents), and
  `unanswerable` (absent by design — the agent must refuse).

### Planted hard cases

| Trap                | Where                                           | What it tests            |
| ------------------- | ----------------------------------------------- | ------------------------ |
| Contradiction       | AcmeCloud 99.9% vs NorthLink 99.5% uptime       | cross-document reasoning |
| Threshold precision | POL-001 ($50k) vs near-duplicate POL-002 ($25k) | reranker precision       |
| Unanswerable        | supplier-gift question (Q-006)                  | refuse-if-ungrounded     |
| Anti-splitting      | $120k split question (Q-007)                    | policy reasoning         |

---

## Retrieval design

- **Hybrid, native.** One collection per chunk holds a dense vector and a sparse
  (lexical) vector; queries fuse both channels with Reciprocal Rank Fusion (RRF).
  Sparse rescues exact tokens — `$25,000`, `99.5%` — that a semantic dense model
  smears together. `tests/test_retrieval.py` proves a real fusion win on a query
  where the two channels disagree.
- **Vector-space safety by construction.** Each embedding provider declares a
  `space_id`, and the collection name is derived from it (`counsel_bge-m3`,
  `counsel_titan-v2`). A local-dev index and a Bedrock-prod index are physically
  separate collections — you cannot query one space against the other by accident.
- **Provider abstraction.** `EmbeddingProvider` is a Protocol yielding dense +
  sparse per text. `HashingEmbedder` is a deterministic offline stand-in for CI;
  real bge-m3 (dev) and Titan (prod) providers implement the same interface.
  Dev/prod is a config swap, not a code branch.

### Advanced stages: rerank + query decomposition

Two optional stages compose around the base retriever via `RetrievalPipeline`,
each independently toggleable so the eval harness can attribute any metric change
to a specific stage:

```text
query --(decompose?)--> sub-queries --retrieve + merge--> --(rerank?)--> top_k
```

- **Reranking** — `TokenInteractionReranker` is a deterministic offline proxy for
  CI (token interaction + a phrase-adjacency bonus, scored without ever consulting
  golden spans). `CrossEncoderReranker` wraps bge-reranker-v2-m3 for production.
- **Query decomposition** — conditional by design: a query is split only when it
  names ≥ 2 known entities *and* carries a comparison cue, so simple queries pass
  through untouched. Multi-hop questions retrieve per-entity and merge, recovering
  the starved side.

#### Honest ablation

```bash
uv run python -m atlas_counsel.ablation
```

On this small corpus, first-stage hybrid nearly saturates retrieval, so the
offline lexical proxies show **no net gain** (rerank slightly lowers precision;
decomposition is net-neutral). That's reported, not hidden. The stages ship as
correct, unit-tested infrastructure; the production cross-encoder and an LLM
decomposer are the implementations expected to win, and this same harness is how
you confirm it locally. The proxies are deliberately *not* tuned toward the golden
spans to manufacture an improvement.

---

## The agent: LangGraph StateGraph

A compiled `StateGraph` with conditional routing, gap-aware iterative retrieval,
a checkpointed human-gate, and a bounded verify/retry loop.

```bash
uv run python -m atlas_counsel.agent --q "who approves a $60,000 purchase?"
uv run python -m atlas_counsel.agent --q "policy on supplier gifts?" --decline
```

- **Structured outputs / citation grounding.** `synthesize` emits a Pydantic
  `DraftAnswer` whose every `Claim` carries a `span_id`; `verify` rejects any claim
  citing a span that wasn't retrieved — proven by
  `test_hallucination_is_bounded_and_escalates`.
- **Gap-aware iterative retrieval.** When `validate` finds insufficient context,
  `gap_analyze` extracts missing-topic tokens and re-retrieves (bounded at
  `MAX_GAP_ITERATIONS = 2`). Retrieved chunks accumulate across iterations so no
  evidence is discarded; only when the gap loop is exhausted does the agent
  escalate to the human gate.
- **Human-assisted decisions.** `human_gate` uses LangGraph `interrupt()`; the run
  pauses and the caller resumes with `Command(resume=...)` to steer or decline.
  State survives the pause via the checkpointer.
- **Bounded loops.** The verify→synthesize retry is capped at `MAX_ATTEMPTS`, then
  escalates — no unbounded LLM spinning.
- **Provider abstraction.** An `LLMProvider` protocol with an offline `TemplateLLM`
  for CI (cites only retrieved spans, never fabricates) and real Ollama/Bedrock
  injected locally. The checkpointer is injected too: `SqliteSaver` (default) or
  `MemorySaver` (dev CLI).
- **Measured, not asserted.** Over the golden set the agent refuses correctly on
  the unanswerable item and does not wrongly refuse any of the 7 answerable ones.
  The single citation miss (Q-003) is a known first-stage retrieval weakness a real
  cross-encoder addresses — left visible, not hidden.

---

## Multi-tier memory

The agent remembers across sessions through three memory tiers, each with a
distinct access pattern:

| Tier       | What it stores                              | Write trigger          | Retrieval                        |
|------------|---------------------------------------------|------------------------|----------------------------------|
| Semantic   | Facts as NL strings                         | After every answer     | Embedding similarity on question |
| Episodic   | One rolling summary per thread              | After every answer     | Embedding similarity on question |
| Procedural | Learned prompt fragments + `when_to_use` cue | After every answer     | JIT similarity on `when_to_use`  |

Memory is loaded before plan (`load_memory` node) and persisted after finalize
(`save_memory` node). The `memory_context` string — relevant facts, past thread
summaries, and matching skills — is injected into the synthesis prompt so the
LLM has continuity across conversations.

```text
START → load_memory → plan → retrieve → validate → ... → finalize → save_memory → END
```

Two storage backends implement the same `MemoryStore` protocol:

- **`InMemoryMemoryStore`** — dict-based, uses the existing `EmbeddingProvider`
  for similarity. Deterministic, dependency-free, used in CI.
- **`SqliteMemoryStore`** — SQLite with JSON-serialized embeddings, per-tenant
  database at `data/{tenant_id}/memory.db`. Cosine similarity at query time.

**Per-tenant isolation.** Each tenant's memories are physically separate — two
tenants never see each other's facts, episode summaries, or skills.

**Offline reflection.** `TemplateLLM.reflect()` extracts semantic facts from
answer sentences heuristically and builds a short episodic summary. Real LLM
providers override this with structured-output prompting for richer extraction.

```bash
uv run pytest tests/test_memory.py -v   # 22 tests covering all three tiers
```

---

## Evaluation harness

Measured *before* the agent existed, so every later change is regression-checked.

```bash
uv run python -m atlas_counsel.eval        # aggregate + per-tag breakdown
uv run python -m atlas_counsel.eval --ab   # A/B two embedding configs
uv run python -m atlas_counsel.eval --meta # judge-bias report
```

- **Retrieval metrics** (no LLM, exact): hit@k, context recall, AP-style context
  precision, MRR — scored against the golden set's known `supporting_span_ids`.
- **Answer metrics** behind an `LLMJudge` protocol: a deterministic `HeuristicJudge`
  runs in CI; inject an LLM-backed judge locally for faithfulness / answer-relevancy
  at full fidelity.
- **Refuse-if-ungrounded** is a scored dimension in its own right (lexical grounding
  overlap, for the reason given above).
- **Per-tag slicing** reports each planted hard case separately, proving the hard
  ones are handled, not just the easy ones.
- **A/B table** across two embedding providers is the harness's native output — the
  artifact the provider abstraction exists to produce.
- **Meta-evaluation** (`--meta`) detects judge bias without an external LLM:
  perturbation-based fluency and confidence-formatting tests, plus Spearman-rank
  retrieval-judge correlation. `HeuristicJudge` is the ~zero-delta baseline; a real
  LLM judge would reveal actual bias.
- A **regression gate** (`tests/test_eval.py`) locks in current aggregate quality so
  future changes fail loudly instead of degrading silently.
- **Langfuse** export is optional (`uv sync --extra langfuse`, set `LANGFUSE_*`); a
  silent no-op otherwise, so CI never depends on it.

---

## Runtime: FastAPI + MCP

The graph is exposed over two transports, both thin wrappers over one
`CounselService` — so HTTP and MCP behave identically (proven by
`test_mcp_and_http_agree`).

```bash
uv sync --extra service
uv run uvicorn atlas_counsel.service.api:app        # HTTP on :8000
uv run python -m atlas_counsel.service.mcp_server   # MCP stdio server
```

**REST**

- `POST /ask {tenant_id, question}` → `{status, thread_id, answer, citations[]}`
- `POST /resume {tenant_id, thread_id, action, guidance?}` → same shape
- `WS /ws/ask` → streams `{event:"node", node}` per step, then a terminal
  `result` / `needs_input` frame

**Interrupt across stateless calls.** The agent pauses at the human-gate via
LangGraph `interrupt()`, but HTTP/MCP are request/response. So a paused run returns
`status="needs_input"` plus a `thread_id`; a second `/resume` call continues it. The
checkpointer carries state across those two otherwise-independent calls — the core
integration design.

**Multi-tenancy.** Each tenant gets its own `SqliteSaver` (isolated checkpoints)
while sharing the read-only retriever. Tenant ids are validated and path-safe.

### Buyer Team integration

The compiled graph is exposed as four MCP tools — `counsel_ask`, `counsel_resume`,
`counsel_brief`, `counsel_health` — so Buyer Team's Strands orchestrator calls it as
one tool among its own. All accept a `tenant_id`. Deployed remotely over Streamable
HTTP or locally over stdio; see `buyer-team-mcp.example.json` for both client entries.

---

## Observability: OpenTelemetry

A silent no-op when the OTEL SDK isn't installed or `OTEL_EXPORTER_OTLP_ENDPOINT`
isn't set — CI never depends on it. When configured, it exports OTLP traces via a
`BatchSpanProcessor`.

| Variable                      | Purpose                 | Default         |
| ----------------------------- | ----------------------- | --------------- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | (disabled)      |
| `OTEL_SERVICE_NAME`           | Service name in traces  | `atlas-counsel` |

Traced: every HTTP request (FastAPI auto-instrumentation), the
`counsel.ask` / `counsel.resume` / `counsel.astream` spans (with `tenant_id`,
`question_len`, `thread_id`, action), and `tenant.create` graph-compilation time.

```bash
uv sync --extra otel
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  uv run uvicorn atlas_counsel.service.api:app
```

---

## Running the full suite

```bash
# offline (113 tests; the Qdrant integration test skips without a server)
uv sync --extra dev
uv run pytest

# full suite (114 tests) — needs the service + qdrant extras
uv sync --extra dev --extra service --extra qdrant
uv run pytest

# hybrid retrieval against a real Qdrant
docker compose up
uv run python -m atlas_counsel.ingest --url http://localhost:6333
uv run pytest tests/test_qdrant_integration.py -v
```

---

## Deployment

AWS ECS Fargate with Qdrant Cloud and per-tenant SQLite checkpoints on EFS.

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # set your qdrant_url
terraform init && terraform apply
# CI/CD then deploys on push to main (build → ECR → ECS).
```

The deployed service exposes both HTTP REST and MCP on the same port: `GET /health`,
`POST /ask`, `POST /resume`, `GET /mcp` (MCP Streamable HTTP), and `WS /ws/ask`. All
endpoints take a `tenant_id` (defaults to `"default"` for single-tenant use).

---

## Layout

```text
src/atlas_counsel/
  _tokenize.py       # shared tokenizer (reranker, judge, answerer)
  telemetry.py       # OpenTelemetry setup (silent no-op when off)
  corpus/            # synthetic corpus generator + golden set
  chunking.py        # span -> chunk, preserving citation ids
  embeddings.py      # EmbeddingProvider protocol + HashingEmbedder
  retrieval.py       # Retriever protocol, RRF fusion, in-memory hybrid
  qdrant_store.py    # production Qdrant hybrid retriever (retry + timeout)
  ingest.py          # CLI: build -> chunk -> index
  eval/              # metrics, judge protocol, runner, A/B, meta-eval, Langfuse
  rerank.py          # Reranker protocol + offline proxy + cross-encoder
  decompose.py       # conditional query decomposition
  pipeline.py        # composed decompose -> retrieve -> merge -> rerank
  ablation.py        # CLI: measure each stage's effect
  agent/             # LangGraph StateGraph: state, nodes (incl. memory), graph, llm
  memory/            # multi-tier memory: semantic, episodic, procedural
    store.py         # MemoryStore protocol + InMemory + Sqlite backends
  service/
    api.py           # FastAPI REST + WebSocket + mounted MCP
    core.py          # CounselService (transport-agnostic)
    mcp_server.py    # MCP tools + remote transport
    tenants.py       # TenantRegistry (per-tenant SqliteSaver)
tests/
infra/               # Terraform: ECS, ALB, EFS, ECR
.github/workflows/   # CI (test) + CD (deploy)
```

---

## License

GNU AGPL v3 — see [LICENSE](LICENSE).
