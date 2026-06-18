# ATLAS Counsel

Citation-grounded agentic RAG over a synthetic procurement corpus. A
complementary knowledge service to the Buyer Team platform: a multi-agent
(LangGraph) RAG pipeline with hybrid retrieval, reranking, citation grounding,
and a measured eval harness.

This first slice ships the **corpus generator + golden eval set** — the
foundation everything else is measured against.

## Quickstart

```bash
pip install -e ".[dev]"
atlas-corpus            # writes data/corpus/*.md, manifest.json, data/eval/golden.jsonl
pytest                  # determinism, integrity, and hard-case guards
```

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

## Layout

```
src/atlas_counsel/corpus/
  models.py      # Pydantic schema — the citation contract
  generator.py   # deterministic content + planted hard cases
  writer.py      # serialize to markdown + JSONL
tests/           # determinism, integrity, hard-case guards
data/            # generated (gitignored)
```
