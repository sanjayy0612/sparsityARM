"""Materialize the preregistered WikiText-2 EXP-003 screening corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def select_records(texts, tokenizer, count: int, min_tokens: int) -> list[dict]:
    selected = []
    for row, text in enumerate(texts):
        text = text.as_py() if hasattr(text, "as_py") else text
        if not isinstance(text, str) or not text.strip() or text.lstrip().startswith("="):
            continue
        token_count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if token_count < min_tokens:
            continue
        selected.append({"id": f"wikitext2-test-row-{row}", "source_row": row,
                         "untruncated_token_count": token_count, "text": text})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"found only {len(selected)} qualifying records")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--min-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing corpus")
    table = pq.read_table(args.parquet, columns=["text"])
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True,
                                              trust_remote_code=False)
    records = select_records(table["text"], tokenizer, args.count, args.min_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                   for record in records))
    print(f"Wrote {len(records)} records from rows "
          f"{records[0]['source_row']}..{records[-1]['source_row']} to {args.output}")


if __name__ == "__main__":
    main()
