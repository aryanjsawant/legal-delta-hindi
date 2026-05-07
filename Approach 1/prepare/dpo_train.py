import os
import re
import torch
from typing import List, Literal, Optional
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from unsloth import PatchDPOTrainer, FastLanguageModel
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

# 1. INITIAL CONFIGURATION
HF_TOKEN = os.getenv("HF_TOKEN")
max_seq_length = 4096 
dtype = None 
load_in_4bit = True 

# Patch the DPO Trainer
PatchDPOTrainer()

# 2. LOAD MODEL & TOKENIZER
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/zephyr-sft-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF_TOKEN,
)

# 3. UTILS & DATASET FUNCTIONS
def apply_chat_template(example, tokenizer, task="sft", assistant_prefix="<|assistant|>\n"):
    def _strip_prefix(s, pattern):
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task == "dpo":
        prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
        if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["chosen"][0])
        
        chosen_messages = example["chosen"][1:]
        rejected_messages = example["rejected"][1:]
        
        example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
        example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
    return example

def create_dpo_pairs(example):
    user_prompt_parts = example['text']['facts-and-arguments']
    judge_opinion_parts = example['text']['judge-opinion']
    label = example['label'] 

    user_prompt_str = " ".join(user_prompt_parts)
    judge_opinion_str = " ".join(judge_opinion_parts)

    generic_granted_response = "Based on the provided arguments and the court's review, the bail application is hereby granted."
    generic_denied_response = "After considering all arguments and evidence, the bail application is hereby denied."

    prompt_messages = [{"role": "user", "content": user_prompt_str}]

    if label == 1:
        chosen_response, rejected_response = judge_opinion_str, generic_denied_response
    elif label == 0:
        chosen_response, rejected_response = generic_granted_response, judge_opinion_str
    else:
        return None

    example["chosen"] = prompt_messages + [{"role": "assistant", "content": chosen_response}]
    example["rejected"] = prompt_messages + [{"role": "assistant", "content": rejected_response}]
    return example

# 4. DATA PREPARATION
il_tur_dataset = load_dataset("Exploration-Lab/IL-TUR", "bail", token=HF_TOKEN)
raw_datasets = DatasetDict({
    "train": il_tur_dataset["train_all"],
    "test": il_tur_dataset["test_all"]
})

raw_datasets = raw_datasets.map(create_dpo_pairs, remove_columns=raw_datasets["train"].column_names)

raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"},
    num_proc = 4,
    remove_columns = ["chosen", "rejected"],
)

for split in raw_datasets.keys():
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

# 5. TRAINING
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 5e-6,
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
        report_to = "none",
        save_strategy = "steps",
        save_steps = 1000,
        save_total_limit = 2,
    ),
    beta = 0.1,
    train_dataset = raw_datasets["train"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)

dpo_trainer.train()
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")