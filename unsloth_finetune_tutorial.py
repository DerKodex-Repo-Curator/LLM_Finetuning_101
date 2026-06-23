"""
Unsloth QLoRA SFT tutorial.

Install:
    pip install -U unsloth datasets trl transformers accelerate bitsandbytes

Run:
    python unsloth_finetune_tutorial.py
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from rich import print

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct" if torch.backends.mps.is_available() else "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
OUTPUT_DIR = "outputs"
ADAPTER_DIR = "llama32_1b_unsloth_lora"
MAX_SEQ_LENGTH = 1024


def build_toy_dataset() -> Dataset:
    """Replace these rows with your own JSON/CSV data later."""
    pairs = [
        ("What is Unsloth?", "Unsloth is a fast fine-tuning library for training open LLMs with less VRAM."),
        ("Explain LoRA in one sentence.", "LoRA trains small adapter weights instead of updating every model parameter."),
        ("Explain QLoRA in one sentence.", "QLoRA combines LoRA with 4-bit quantization to reduce memory use."),
        ("When should I fine-tune instead of using RAG?", "Fine-tune when you want behavior, style, or task performance to change."),
        ("What does max_seq_length control?", "It controls the maximum tokens in one training example."),
        ("What is gradient accumulation?", "It simulates a larger batch by accumulating gradients over small steps."),
        ("What should I save after LoRA training?", "Save the LoRA adapter and tokenizer so you can reload them later."),
        ("Give me a YouTube intro for this tutorial.", "Today we are fine-tuning a small language model with Unsloth and QLoRA."),
    ]
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "You are a concise AI coding tutor."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            for q, a in pairs
        ]
    )


def load_model():
    """Load a 4-bit model and attach trainable LoRA adapters."""
    if torch.backends.mps.is_available():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="mps",
        )

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=not torch.backends.mps.is_available(),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


def format_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Convert chat messages into the model's chat-template text."""

    def to_text(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in batch["messages"]]
        return {"text": texts}

    return dataset.map(to_text, batched=True, remove_columns=["messages"])


def train(model, tokenizer, dataset: Dataset):
    """Short demo fine-tune. Increase max_steps or use epochs for real data."""
    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_torch" if torch.backends.mps.is_available() else "adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        bf16=False if torch.backends.mps.is_available() else True,
        fp16=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()


def ask_model(model, tokenizer, prompt: str):
    """Run a quick test after training."""
    if torch.cuda.is_available():
        FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": "You are a concise AI coding tutor."},
        {"role": "user", "content": prompt},
    ]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        streamer=streamer,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )


def main():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        raise SystemExit("This demo expects an NVIDIA CUDA GPU or Apple GPU (MPS).")
    raw_dataset = build_toy_dataset()
    model, tokenizer = load_model()
    train_dataset = format_dataset(raw_dataset, tokenizer)
    print("\n--- Sample formatted training row ---")
    print(train_dataset[0]["text"][:500])
    print("\n--- Training ---")
    train(model, tokenizer, train_dataset)
    print("\n--- Test the fine-tuned adapter ---")
    ask_model(model, tokenizer, "Explain QLoRA like I am new to fine-tuning.")
    print(f"\n--- Saving LoRA adapter to {ADAPTER_DIR} ---")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)


if __name__ == "__main__":
    main()
