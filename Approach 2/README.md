# Approach 2: DPO on NyayaAnumana Legal Dataset

## Overview

This directory contains our approach using **Direct Preference Optimization (DPO)** to align a Large Language Model with Indian judicial decisions. 

Instead of a complex reward model (like GRPO), DPO simplifies the Reinforcement Learning pipeline by learning directly from preference pairs (chosen vs. rejected responses). It forces the model to increase the probability of the correct/preferred answer while decreasing the probability of the rejected one.

## How It Works

1. **Dataset**: We utilized the **NyayaAnumana** dataset — a corpus of Indian court case descriptions featuring ternary classification labels (Accepted / Rejected / Multi-label).
2. **Preference Pair Construction**: For each case, we transformed the raw classification dataset into `(prompt, chosen, rejected)` triples:
   - *Example chosen*: `"Classification: Accepted\nRationale: The case outcome indicates acceptance."`
   - *Example rejected*: `"Classification: Rejected\nRationale: The case outcome indicates rejection."`
3. **Training**: We used `Qwen2.5-3B-Instruct` with 4-bit quantization and LoRA parameter-efficient fine-tuning via the Unsloth library.

## Results

This approach was highly successful. As documented in our experiments:
- **Validation Loss (0.1278)** was lower than **Training Loss (0.1345)**, showing **zero overfitting**.
- The log probability gap between the correct legal answer and the wrong one grew over time, proving the model became more confident in its legal reasoning.
- The Logits/chosen and Logits/rejected metrics were very close, suggesting the model wasn't making "wild" guesses.

## How to Set Up and Run

This approach is self-contained within a Google Colab notebook for easy execution.

### Option A — Google Colab (Recommended)
1. Open `dpo_unsloth_nyananuman_legal.ipynb` in Google Colab.
2. Set your runtime hardware accelerator to **GPU** (a free T4 is sufficient).
3. Run all cells sequentially. The notebook will automatically handle downloading the NyayaAnumana dataset from Google Drive.

### Option B — Local Environment
1. Ensure you have a GPU with ≥16 GB VRAM.
2. Install dependencies:
   ```bash
   pip install unsloth "trl[all]"
   pip install --no-deps peft accelerate bitsandbytes
   pip install datasets sentencepiece gdown
   ```
3. Run the notebook using Jupyter locally, or convert it to a python script:
   ```bash
   jupyter nbconvert --to script dpo_unsloth_nyananuman_legal.ipynb
   python dpo_unsloth_nyananuman_legal.py
   ```
