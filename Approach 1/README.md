# Approach 1: GRPO with Information Gain (Hindi Bail Prediction)

## Overview

This directory contains our faithful adaptation of the **LegalΔ** paper. We took the original GRPO training pipeline (designed for Chinese legal data) and re-engineered it for **Indian bail prediction in Hindi**.

This approach uses Group Relative Policy Optimization (GRPO) and a unique Information Gain (Delta) reward mechanism. By comparing model confidence between responses *with* reasoning and *without* reasoning, it trains the model to generate meaningful Chain-of-Thought (CoT) processes before concluding whether bail is Granted or Denied.

### What We Changed From the Original LegalDelta Framework

| Component | Original (Chinese) | Our Adaptation (Hindi) |
|---|---|---|
| **Model** | `Qwen-2.5-14B-Instruct` | `Meta-Llama-3.1-8B-Instruct` (better Hindi tokenization) |
| **Dataset** | Custom Chinese legal JSON | `Exploration-Lab/IL-TUR` (HuggingFace, bail subset) |
| **System Prompt** | Chinese | Hindi (हिंदी) — full prompt in Devanagari |
| **Answer Format** | `[法条]`, `[罪名]`, `[刑期]` etc. | `[Bail]Granted<eoa>` / `[Bail]Denied<eoa>` |
| **Reward Functions** | Chinese statute regex parser | `bail_accuracy_score()` — simple binary match |
| **Quantization** | Full precision | 4-bit (`load_in_4bit=True`) for VRAM efficiency |

## Repository Structure

- `src/train_il_tur.py`: Our main monolithic script containing the Hindi system prompt, IL-TUR dataset loading logic, XML Format Rewards, Bail Accuracy Rewards, and the Information Gain (Delta) Reward formula.
- `requirements.txt` / `req.txt`: Pip requirements for this specific environment.

## How to Set Up and Run

### Prerequisites
- **Hardware**: GPU with ≥24 GB VRAM (tested on NVIDIA H100). This setup requires significant compute due to the GRPO generation process.
- **Software**: Python 3.11+, CUDA 12.x
- **HuggingFace Token**: Required for accessing the IL-TUR dataset.

### Setup
```bash
# Create conda environment
conda create -n grpo python=3.11
conda activate grpo

# Install dependencies
pip install -r req.txt

# Set HuggingFace token
export HF_TOKEN="your_hf_token_here"
```

### Run Training
**Direct execution (single GPU):**
```bash
python src/train_il_tur.py
```

### Merge LoRA Weights
After training, to merge the trained LoRA adapters back into the base model:
```bash
python src/merge_lora.py \
    --model_name_or_path outputs/<experiment_name>/final_model \
    --base_model_path unsloth/Meta-Llama-3.1-8B-Instruct \
    --save_path outputs/merged_model
```