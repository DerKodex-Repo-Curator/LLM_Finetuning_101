# LLM_Finetuning_101

Companion code for the YouTube tutorial on fine-tuning a small LLM with **Unsloth + QLoRA + SFT**.

The single script `finetune_example.py` builds a toy chat dataset, attaches LoRA adapters to `unsloth/Llama-3.2-1B-Instruct`, trains with `SFTTrainer`, runs a sample inference, and saves the adapter to `llama32_1b_unsloth_lora/`.

## Requirements

- A CUDA GPU **or** Apple Silicon (MPS). The script exits on CPU-only hosts.
- Python 3.10+

## Setup

```bash
pip install -U unsloth datasets trl transformers accelerate bitsandbytes
```

## Run

```bash
python finetune_example.py
```

## What it does

1. Builds a tiny chat-formatted dataset in `build_toy_dataset()` (replace these pairs with your own data later).
2. Loads `Llama-3.2-1B-Instruct` and attaches LoRA adapters (r=16) on the attention and MLP projections.
3. Trains with `SFTTrainer` for 30 demo steps (increase `max_steps` or switch to epochs for real data).
4. Streams a sample generation from the fine-tuned adapter.
5. Saves the LoRA adapter + tokenizer to `llama32_1b_unsloth_lora/`. Checkpoints go to `outputs/`.

## Device-specific behavior

The script branches on `torch.backends.mps.is_available()`:

| Path | Model | Quantization | Optimizer | Precision |
|------|-------|--------------|-----------|-----------|
| **MPS (Apple Silicon)** | `unsloth/Llama-3.2-1B-Instruct` | none (`load_in_4bit=False`) | `adamw_torch` | fp32 |
| **CUDA** | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | 4-bit | `adamw_8bit` | bf16 |

On MPS the script uses raw `transformers` + `peft` instead of `unsloth.FastLanguageModel`, since Unsloth's 4-bit path is CUDA-only by design.

## Swapping in real data

Replace the `pairs` list in `build_toy_dataset()` (`finetune_example.py:25`) with your own `(question, answer)` tuples, or load a Hugging Face dataset and map it into the same `{"messages": [...]}` format.

## Outputs (gitignored)

- `outputs/` — training checkpoints
- `llama32_1b_unsloth_lora/` — final LoRA adapter + tokenizer

## License

See [LICENSE](LICENSE).