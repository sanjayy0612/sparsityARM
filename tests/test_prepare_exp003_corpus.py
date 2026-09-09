import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/prepare_exp003_corpus.py"
    spec = importlib.util.spec_from_file_location("prepare_exp003_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


def test_selection_is_ordered_and_excludes_headings_and_short_rows():
    module = _module()
    texts = ["", " = heading = ", "too short", "one two three four", "a b c d e"]
    records = module.select_records(texts, Tokenizer(), count=2, min_tokens=4)
    assert [record["source_row"] for record in records] == [3, 4]
    assert [record["id"] for record in records] == ["wikitext2-test-row-3", "wikitext2-test-row-4"]
