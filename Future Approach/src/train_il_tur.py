import os
import argparse

# Parse HPC shard arguments
parser = argparse.ArgumentParser()
parser.add_argument("--shard_id", type=int, default=1, help="Slurm array task ID")
parser.add_argument("--num_shards", type=int, default=30, help="Total number of shards")
args, _ = parser.parse_known_args()

# Rely on SLURM for GPU allocation; removing strict CUDA_VISIBLE_DEVICES to prevent single-GPU pinning on multi-GPU nodes
os.environ["UNSLOTH_COMPILE_OVERWRITE"] = "0"
os.environ["UNSLOTH_CACHE_DIR"] = "/home/tanmoyhazra/RLnew/scripts/unsloth_compiled_cache"

import torch._dynamo
import torch
import json
import re
import math
from datetime import datetime
import time
import numpy as np

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

experiment_name = f"llama3_1_8B_GRPO_IL_TUR_shard{args.shard_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"{experiment_name}.log")

def log_print(message):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

# ========== MODEL CONFIGURATION ==========
# For A100/H100 HPC execution, maintaining a high max seq length safely
max_seq_length = 4096 
lora_rank = 32

log_print("Loading Unsloth Model (Meta-Llama-3.1-8B-Instruct)...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True, # Safety default for OOM prevention
        fast_inference=False,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )
except Exception as e:
    log_print(f"CRITICAL ERROR loading model: {e}")
    raise e

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
)

# ========== HINDI SYSTEM PROMPT ==========
SYSTEM_PROMPT = """आप एक भारतीय कानूनी विशेषज्ञ हैं। 
उपयोगकर्ता एक जमानत आवेदन (Bail Application) के कुछ तथ्य (Facts) प्रस्तुत करेगा। 
आपको तथ्यों का गहराई से विश्लेषण करना है और निर्णय लेना है कि जमानत मंजूर (Granted) होगी या नामंजूर (Denied)।

कृपया निम्नलिखित प्रारूप का पालन करते हुए उत्तर दें:
<reasoning>
यहां तथ्यों का विस्तृत विश्लेषण करें, अपने विचार और तार्किक कारण समझाएं। (हिंदी में)
</reasoning>
<answer>
[Bail]Granted<eoa> या [Bail]Denied<eoa>
</answer>"""

# ========== DATA PREPARATION ==========
def extract_facts(text_str):
    try:
        data = json.loads(text_str)
        facts = "\n".join(data.get("facts-and-arguments", []))
        return facts
    except:
        return str(text_str)

def get_dataset() -> Dataset:
    log_print("Downloading IL-TUR dataset from HuggingFace...")
    try:
        # Fallback to no task_name if "bail" fails since HF structure varies
        dataset = load_dataset("Exploration-Lab/IL-TUR", "bail", revision="script", token=True)
    except ValueError:
        log_print("Failed to load 'bail' config, loading default config instead...")
        dataset = load_dataset("Exploration-Lab/IL-TUR", revision="script", token=True)
        
    ds = dataset["train"]
         
    def format_row(x):
        facts = extract_facts(x.get('text', ''))
        label_str = str(x.get('label', ''))
        
        # Binary 1 or 0
        is_granted = "1" in label_str
        answer_text = "[Bail]Granted<eoa>" if is_granted else "[Bail]Denied<eoa>"
        
        prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"[QUERY_ID:{x.get('id', 'N/A')}]\nकृपया निम्नलिखित तथ्यों के आधार पर जमानत की भविष्यवाणी करें:\n\nतथ्य:\n{facts}"}
        ]
        
        return {
            'prompt': prompt,
            'answer': answer_text,
            'id': str(x.get('id', ''))
        }

    formatted_ds = ds.map(format_row, desc="Formatting dataset")
    return formatted_ds

try:
    log_print("Preparing datasets...")
    full_dataset = get_dataset()
    if args.num_shards > 1:
        log_print(f"Sharding dataset: shard {args.shard_id} of {args.num_shards}")
        # HF dataset shards are 0-indexed, so args.shard_id - 1
        full_dataset = full_dataset.shard(num_shards=args.num_shards, index=args.shard_id - 1)
    train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']
    log_print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
except Exception as e:
    log_print(f"CRITICAL ERROR loading datasets: {e}")
    raise e

# ========== REWARD FUNCTIONS (Bullet-Proofed for HPC) ==========

def count_xml_reward(text):
    count = 0.0
    if text.count("<reasoning>\n") == 1: count += 0.125
    if text.count("\n</reasoning>\n") == 1: count += 0.125
    if text.count("<answer>\n") == 1: count += 0.125
    if text.count("\n</answer>") == 1: count += 0.125
    if re.search(r'\[Bail\]', text): count += 0.125
    if text.count("<eoa>") == 1:
        count += 0.125
        penalty = len(text.split("<eoa>\n</answer>")[-1]) * 0.01
        count -= min(penalty, 0.375)
    try:
        parts = text.split("<reasoning>\n")
        reasoning_content = parts[1].split("\n</reasoning>\n")[0]
        answer_parts = text.split("<answer>\n")
        answer_content = answer_parts[1].split("\n</answer>")[0]
        if reasoning_content and answer_content and "[Bail]" in answer_content and "<eoa>" in answer_content:
            count += 0.25
    except:
        pass
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Safety-wrapped XML structure reward calculator"""
    scores = []
    for completion in completions:
        try:
            scores.append(count_xml_reward(completion[0]["content"]))
        except Exception:
            scores.append(0.0) # Fail safe
    return scores

def bail_accuracy_score(response, current_answer):
    pred_match = re.search(r'\[Bail\](.*?)<eoa>', response, re.DOTALL)
    predicted = pred_match.group(1).strip() if pred_match else ""
    
    corr_match = re.search(r'\[Bail\](.*?)<eoa>', current_answer, re.DOTALL)
    correct = corr_match.group(1).strip() if corr_match else current_answer
    
    if not predicted:
        return 0.0
    elif predicted == correct:
        return 2.0  # Max reward for matching properly
    elif predicted in ["Granted", "Denied"]:
        return 0.5  # Output correctly formatted but incorrect prediction
    else:
        return -1.0 # Penalize hallucinated label wrappers

def calculate_answer_diversity_bonus(answer_token_info, baseline_answer_info, base_score):
    """The core 'Delta' Logic from LegalDelta Framework"""
    if not answer_token_info or 'avg_logit' not in answer_token_info:
        return base_score
    if not baseline_answer_info or 'avg_logit' not in baseline_answer_info:
        return base_score
    
    reasoning_logit = answer_token_info['avg_logit']
    direct_logit = baseline_answer_info['avg_logit']
    
    # Information gain metric
    info_gain = reasoning_logit - direct_logit
    info_gain_factor = torch.sigmoid(torch.tensor(info_gain/5)).item()
    return base_score * info_gain_factor

def enhanced_bail_reward_func(completions, answer, answer_token_info=None, baseline_answer_info=None, **kwargs) -> list[float]:
    """Bail Accuracy combined with LegalDelta Information Gain Multipliers"""
    enhanced_scores = []
    for i, completion in enumerate(completions):
        try:
            response = completion[0]["content"]
            current_answer = answer[i] if i < len(answer) else answer[0]
            
            # 1. Base Accuracy Score
            base_score = bail_accuracy_score(response, current_answer)
            
            # 2. Extract Token Info for Delta Gain (If unsloth/TRL passes it)
            current_ans_info = answer_token_info[i] if answer_token_info and i < len(answer_token_info) else None
            current_base_info = baseline_answer_info[i] if baseline_answer_info and i < len(baseline_answer_info) else None
            
            # 3. Apply Multiplier
            if current_ans_info and current_base_info:
                enhanced_score = calculate_answer_diversity_bonus(current_ans_info, current_base_info, base_score)
            else:
                enhanced_score = base_score
                
            enhanced_scores.append(enhanced_score)
        except Exception:
            enhanced_scores.append(0.0) # Fail safe to avoid crashing training loop
            
    return enhanced_scores

# ========== TRAINING CONFIG ==========
training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=4, # Safe sizing to avoid HPC VRAM spikes 
    per_device_eval_batch_size=2,
    max_prompt_length=2048, 
    max_completion_length=512, 
    num_train_epochs=1, 
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_strategy="steps",
    log_level="info",
    max_grad_norm=0.1,
    report_to="tensorboard",
    output_dir=os.path.join("outputs", experiment_name),
    local_rank=-1,
)

# ========== TRAINER INSTANTIATION ==========
log_print("Initializing Trainer...")
try:
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[xmlcount_reward_func, enhanced_bail_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
except Exception as e:
    log_print(f"CRITICAL ERROR initializing Trainer: {e}")
    raise e

log_print("Starting Pilot Training on IL-TUR...")
try:
    trainer.train()
    log_print("Training Complete! Saving final model...")
    trainer.save_model(os.path.join("outputs", experiment_name, "final_model"))
except Exception as e:
    log_print(f"CRITICAL ERROR during training: {e}")
    raise e
