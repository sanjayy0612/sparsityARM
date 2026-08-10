"""CPU-only dense generation benchmark."""
from pathlib import Path
import json, time
from common import base_parser, load_cpu_model, write_result


def main() -> None:
    args = base_parser(__doc__).parse_args()
    import torch
    prompts = json.loads((Path(__file__).with_name("prompts.json")).read_text())
    model, tokenizer = load_cpu_model(args.model, args.threads)
    prompt = prompts[0]
    inputs = tokenizer(prompt["prompt"], return_tensors="pt")
    for _ in range(args.warmup):
        with torch.inference_mode(): model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    elapsed = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        with torch.inference_mode(): output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        elapsed.append(time.perf_counter() - start)
    generated = output.shape[-1] - inputs.input_ids.shape[-1]
    record = {"mode":"dense", "model":args.model, "prompt_id":prompt["id"], "prompt_tokens":inputs.input_ids.shape[-1],
              "generated_tokens":generated, "repeats":args.repeats, "warmup":args.warmup, "threads":args.threads,
              "median_generation_ms":sorted(elapsed)[len(elapsed)//2]*1000,
              "tokens_per_second": generated / (sum(elapsed)/len(elapsed)), "decoding":"greedy", "dtype":"float32"}
    path = write_result(record, args.output_dir)
    print(json.dumps(record, indent=2)); print(f"Saved {path}")

if __name__ == "__main__": main()
