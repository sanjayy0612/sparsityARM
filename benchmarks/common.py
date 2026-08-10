import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from armsparse.utils.hardware import system_info
from armsparse.utils.reproducibility import set_seed


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model ID")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "results")
    return parser


def load_cpu_model(model_name: str, threads: int | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    return model, tokenizer


def write_result(record: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "machine": system_info(), **record}
    name = f"{record['mode']}_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.json"
    path = directory / name
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
