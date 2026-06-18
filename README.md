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
|------|-------|-------|
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
- **Langfuse** export is optional (`uv sync --extra langfuse`, set
  `LANGFUSE_*` env); a silent no-op otherwise so CI never depends on it.
- A **regression gate** (`tests/test_eval.py`) locks in current aggregate
  quality so future changes fail loudly instead of degrading silently.

## Layout

```
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
- **Langfuse** export is optional (`uv sync --extra langfuse`, set
  `LANGFUSE_*` env); a silent no-op otherwise so CI never depends on it.
- A **regression gate** (`tests/test_eval.py`) locks in current aggregate
  quality so future changes fail loudly instead of degrading silently.

## Layout

```
src/atlas_counsel/
  corpus/          # synthetic corpus generator + golden set (PR1)
  chunking.py      # span -> chunk, preserving citation ids
  embeddings.py    # EmbeddingProvider protocol + offline HashingEmbedder
  retrieval.py     # Retriever protocol, RRF fusion, in-memory hybrid
  qdrant_store.py  # production Qdrant hybrid retriever (named vectors + RRF)
  ingest.py        # CLI: build -> chunk -> index
  eval/            # metrics, judge protocol, runner, A/B report, Langfuse hook
tests/             # corpus + retrieval + (skipped) Qdrant integration
docker-compose.yml # local Qdrant
```
