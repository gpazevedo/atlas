"""Serialize a validated Corpus to disk.

Layout:
  data/corpus/<DOC_ID>.md   — one markdown file per document (front-matter +
                              spans with explicit anchors), for ingest + humans
  data/corpus/manifest.json — machine index of every span id (citation source)
  data/eval/golden.jsonl    — one GoldenItem per line for the eval harness
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Corpus


def _doc_to_markdown(doc) -> str:
    fm = [
        "---",
        f"doc_id: {doc.doc_id}",
        f"category: {doc.category.value}",
        f"title: {doc.title}",
    ]
    if doc.vendor:
        fm.append(f"vendor: {doc.vendor}")
    if doc.tags:
        fm.append(f"tags: [{', '.join(doc.tags)}]")
    fm.append("---\n")

    body = [f"# {doc.title}\n"]
    for s in doc.spans:
        # Explicit anchor comment so the span id is recoverable from the md.
        body.append(f"<!-- span:{s.span_id} -->")
        if s.heading:
            body.append(f"## {s.heading}")
        body.append(s.text + "\n")
    return "\n".join(fm) + "\n".join(body)


def write_corpus(corpus: Corpus, root: Path) -> dict[str, int]:
    corpus_dir = root / "data" / "corpus"
    eval_dir = root / "data" / "eval"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"spans": [], "documents": []}
    for doc in corpus.documents:
        (corpus_dir / f"{doc.doc_id}.md").write_text(
            _doc_to_markdown(doc), encoding="utf-8"
        )
        manifest["documents"].append(
            {"doc_id": doc.doc_id, "category": doc.category.value,
             "title": doc.title, "vendor": doc.vendor, "tags": doc.tags}
        )
        for s in doc.spans:
            manifest["spans"].append(
                {"span_id": s.span_id, "doc_id": doc.doc_id,
                 "heading": s.heading, "text": s.text}
            )

    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (eval_dir / "golden.jsonl").open("w", encoding="utf-8") as fh:
        for item in corpus.golden:
            fh.write(item.model_dump_json() + "\n")

    return {
        "documents": len(corpus.documents),
        "spans": len(manifest["spans"]),
        "golden_items": len(corpus.golden),
    }
