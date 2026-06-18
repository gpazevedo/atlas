"""CLI: python -m atlas_counsel.corpus [--seed N] [--root PATH]"""
import argparse
from pathlib import Path

from . import build_corpus, write_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the ATLAS Counsel corpus.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args()

    corpus = build_corpus(seed=args.seed)
    stats = write_corpus(corpus, args.root)
    print(f"Corpus written under {args.root / 'data'}")
    for k, v in stats.items():
        print(f"  {k:14} {v}")


if __name__ == "__main__":
    main()
