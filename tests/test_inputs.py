import hashlib

import pytest

from research_tools import inputs


def make_root(tmp_path):
    root = tmp_path / "clone"
    (root / "research" / "data").mkdir(parents=True)
    (root / "research" / "data" / "corpus.jsonl").write_text("abc")
    return root


DIGEST = hashlib.sha256(b"abc").hexdigest()


def test_resolve_rebases_foreign_absolute_path(tmp_path):
    root = make_root(tmp_path)
    foreign = "/Users/someone/elsewhere/arm-sparse/research/data/corpus.jsonl"
    assert inputs.resolve(foreign, root) == root / "research/data/corpus.jsonl"


def test_resolve_keeps_existing_and_unrebasable_paths(tmp_path):
    root = make_root(tmp_path)
    existing = root / "research/data/corpus.jsonl"
    assert inputs.resolve(existing, root) == existing
    missing = "/Users/x/.cache/huggingface/model.safetensors"
    assert str(inputs.resolve(missing, root)) == missing


def test_check_sha_verifies_present_file_via_rebasing(tmp_path):
    root = make_root(tmp_path)
    foreign = "/Users/x/arm-sparse/research/data/corpus.jsonl"
    assert inputs.check_sha(foreign, DIGEST, "corpus", root) is True


def test_check_sha_skips_missing_with_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(inputs.STRICT_ENV, raising=False)
    root = make_root(tmp_path)
    assert inputs.check_sha("/Users/x/arm-sparse/research/data/gone.gguf", DIGEST, "model", root) is False
    out = capsys.readouterr().out
    assert out.startswith(inputs.SKIP_PREFIX) and "gone.gguf" in out and DIGEST in out


def test_strict_mode_fails_on_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(inputs.STRICT_ENV, "1")
    root = make_root(tmp_path)
    with pytest.raises(inputs.MissingInput):
        inputs.check_sha("/Users/x/arm-sparse/research/data/gone.gguf", DIGEST, "model", root)


@pytest.mark.parametrize("strict", ["0", "1"])
def test_hash_mismatch_always_fails(tmp_path, monkeypatch, strict):
    monkeypatch.setenv(inputs.STRICT_ENV, strict)
    root = make_root(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        inputs.check_sha(root / "research/data/corpus.jsonl", "0" * 64, "corpus", root)
