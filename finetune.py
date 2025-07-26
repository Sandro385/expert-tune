"""
finetune.py
---------------

This script performs LoRA fine‑tuning of a large language model using
the Unsloth `FastLanguageModel` utilities.  In the updated
architecture, chat history is stored in a SQLite database via
`auth_db.py`.  When this script runs, it first extracts the chat
history for the current user and domain, writes it to a JSONL
dataset and then performs LoRA fine‑tuning.  The resulting adapter
is saved in the `lora_model` directory.

Set the environment variables `CURRENT_USER` and `CURRENT_DOMAIN`
before invoking this script so that it can retrieve the appropriate
conversation from the database.
"""

import os
import json
from typing import List, Dict

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from trl import SFTTrainer

from auth_db import load_history

def build_dataset_from_history(history: List[Dict[str, str]], domain: str, output_file: str) -> None:
    """Create a JSONL dataset from the user's chat history.

    This helper function iterates over the chat history in pairs of
    messages (user → assistant) and writes them to a JSONL file
    suitable for training.  Each line contains a JSON object with
    `prompt` and `completion` keys.

    Args:
        history: A list of messages returned by `load_history`, each
                 with keys `role` and `content`.
        domain: The domain context (e.g. "იურისტი") used to augment
                the prompt.
        output_file: Path to the JSONL file to create.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                prompt = f"შექმენი {domain} პასუხი.\n{history[i]['content']}"
                completion = history[i + 1]["content"]
                record = {"prompt": prompt, "completion": completion}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Entry point for the fine‑tuning script."""
    # Retrieve the user and domain from environment variables.  If
    # these variables are not set, raise an informative error.
    user = os.getenv("CURRENT_USER")
    domain = os.getenv("CURRENT_DOMAIN")
    if not user or not domain:
        raise RuntimeError(
            "CURRENT_USER and CURRENT_DOMAIN environment variables must be set"
        )
    # Load chat history from the database
    history = load_history(user, domain)
    if not history:
        raise RuntimeError("No chat history found for the specified user and domain")
    # Build the training dataset
    dataset_path = "dataset.jsonl"
    build_dataset_from_history(history, domain, dataset_path)

    # Maximum sequence length for the model
    max_seq_length = 2048
    # Load a 4‑bit quantised Meta‑Llama model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    # Prepare the model for LoRA fine‑tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    # Load the dataset from the generated JSONL
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # Get chat template and formatting function
    tokenizer = FastLanguageModel.get_chat_template(tokenizer)
    def formatting_prompts_func(examples):
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                tokenize=False,
            )
            texts.append(text)
        return {"text": texts}
    # Apply formatting to the dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    # Configure the SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=dict(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )
    # Train and save the LoRA‑adapted model
    trainer.train()
    trainer.save_model("lora_model")


if __name__ == "__main__":
    main()