# AGENTS.md

Single-file educational repo for an Unsloth QLoRA SFT tutorial (YouTube companion).

## Layout
- `finetune_example.py` — the only executable source. Self-contained demo that builds a toy chat dataset, attaches LoRA adapters to `unsloth/Llama-3.2-1B-Instruct`, trains with `SFTTrainer`, runs a sample inference, and saves the adapter to `llama32_1b_unsloth_lora/`.

## Running
Requires CUDA or Apple MPS; the script exits on CPU-only hosts.

```bash
pip install -U unsloth datasets trl transformers accelerate bitsandbytes
python finetune_example.py
```

Note: the module's docstring says `python unsloth_finetune_tutorial.py`, but the actual file is `finetune_example.py` — run the latter.

## Device-specific behavior (important)
The code branches on `torch.backends.mps.is_available()`:
- **MPS (Apple Silicon):** uses raw `transformers` + `peft` (not `FastLanguageModel`), `load_in_4bit=False`, `adamw_torch`, `bf16=False`. Model selection uses `unsloth/Llama-3.2-1B-Instruct`.
- **CUDA:** uses `unsloth.FastLanguageModel` with 4-bit quantization, `adamw_8bit`, `bf16=True`, gradient checkpointing. Model uses `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`.

Don't "fix" the MPS path to use `FastLanguageModel` — Unsloth's 4-bit path is CUDA-only by design.

## Conventions
- No tests, no CI, no lint/typecheck config. Verify edits by running the script end-to-end on the target device.
- Training writes to `outputs/` (checkpoints) and `llama32_1b_unsloth_lora/` (final adapter + tokenizer). Both are gitignored artifacts.
- The toy dataset is hardcoded in `build_toy_dataset()`; swapping in real data means replacing those pairs only.