# EXP-002: real-model activation mask coverage

Question: does natural FFN neuron order preserve useful block sparsity when
magnitude-selected neurons are expanded to their containing blocks?

Primary model: official meta-llama/Llama-3.2-1B, pinned revision
4e20de362430cd3b72f300e6b0f18e50e7166e08. CPU only, BF16 stored weights and model
activations, one thread, no autocast, training or weight changes. This precision
choice keeps the initial capture practical on an 8 GiB M2. Saved activations
are converted losslessly from BF16 to FP32 for offline analysis. This is not
the FP32 performance protocol of EXP-001.

Capture all 16 layers from eight fixed, locally authored prompts covering code,
prose, arithmetic, dialogue and structured data. Use causal teacher-forced
prefill, one prompt at a time, at most 64 tokens, without KV cache or generation.
Analyze token positions 8 onward (zero-based) to exclude the shortest prefixes.
Save exact token IDs, prompt text, MLP inputs and down-projection inputs
(SiLU(gate_proj(x))*up_proj(x)). The down-projection pre-hook observes actual
model activations; it must not modify them. Validate selected observed values
by independently recomputing the FFN activation from saved MLP inputs.

Rank neurons by absolute post-SwiGLU activation at 20/30/40% requested sparsity.
This is an activation-magnitude reference, not proof of optimal importance or
a deployable predictor. Resolve ties using a fixed randomized permutation to
avoid favoring contiguous indices when BF16 values tie. Save masks themselves.

For B=8/16/32/64, measure per layer/prompt/token:

- Realized neuron sparsity and block sparsity after expansion preserving every
  selected neuron.
- Expansion coverage under three fixed random neuron permutations of the same
  masks, as a spatial-locality control.
- Selecting the top blocks by summed absolute activation at the same nominal
  sparsity budget: realized sparsity and retained activation L1 mass, compared
  with neuron selection. These masks may discard originally selected neurons.

Retained activation mass is a surrogate, not language quality or FFN output
error. Report the full grid and layer variation. Tokens and layers are
correlated; this pilot corpus is not a representative quality evaluation.
No selector latency or inference speed is claimed. Capture elapsed time is
operational metadata only, with no warmups or repetitions for performance.

Save revision and model-file hashes, environment, source snapshots, git commit
and dirty state, seeds, token IDs, raw activations/masks/metrics and aggregation
method. Fail capture on nonfinite values or hook/recomputation disagreement.
Use a memory check after loading; do not proceed if resident memory exceeds
5.5 GiB. Record peak RSS and system swap when the OS exposes it; otherwise
record swap as null rather than failing the capture. Do not silently substitute a
different model, precision, prompt set or capture mode.

Success means a verified capture and reproducible coverage analysis, regardless
of whether block coverage is favorable. A positive result supports later replay;
a negative result restricts the current expansion strategy, not all structured
sparsity approaches. Major changes to the research direction still need human
review.

## Execution

Install the model extra with `pip install -e '.[dev,model]'`. Download only the
pinned official checkpoint/config/tokenizer files using Hugging Face, then:

```bash
.venv/bin/python benchmarks/capture_exp002.py --snapshot <pinned-snapshot-directory> --output research/results/EXP-002/<new-run-directory>
.venv/bin/python research_tools/analyze_exp002.py research/results/EXP-002/<new-run-directory>
.venv/bin/python research_tools/validate_exp002.py research/results/EXP-002/<new-run-directory>
```

Both capture and analysis refuse to overwrite existing output directories.
Checkpoint access credentials remain in the local Hugging Face configuration;
they are never copied into experiment artifacts.

Expected local artifact volume is roughly 0.3 GiB for uncompressed activation
captures plus compressed mask/metric outputs; verify at least 2 GiB free before
starting. Model checkpoint storage is separate.
