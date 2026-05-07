# Preference-Based Training for Legal Bail Prediction

> **Course Project — Reinforcement Learning**  
> **Team:** Aryan Sawant (U23AI042), Jaya Bhati (U23AI043), Kuruv Patel (U23AI051)  
> **Topic:** Adapting preference-based alignment and reinforcement learning (including the [LegalΔ framework](https://arxiv.org/abs/2508.12281)) for **Indian legal datasets**.

---

## 📌 Project Overview

This project explores different Reinforcement Learning from Human Feedback (RLHF) and preference alignment techniques to improve the legal reasoning and outcome prediction of Large Language Models (LLMs) in the Indian legal context.

Based on our presentation and codebase, we explored the following three approaches:

### 1. DPO on IL-TUR Dataset (Initial Experiment)
- **Objective:** Use Direct Preference Optimization (DPO) to align the `zephyr-sft-bnb-4bit` model with judicial decisions on the IL-TUR bail dataset.
- **Method:** We constructed preference pairs where the "Chosen" response was the actual judge's opinion and the "Rejected" response provided the opposite, incorrect legal outcome.
- **Results:** The model aggressively favored the chosen response, rapidly achieving perfect accuracy. However, this led to vanishing loss (no gradient for backpropagation), indicating the model was either overfitting to the exact labels or the setup was too constrained.

### 2. DPO on NyayaAnumana Dataset (Approach 2)
- **Objective:** Use DPO to align the `Qwen2.5-3B-Instruct-bnb-4bit` model with judicial decisions on the NyayaAnumana dataset.
- **Method:** We transformed the raw classification dataset into `(prompt, chosen, rejected)` preference triples and applied LoRA for parameter-efficient fine-tuning on a Google Colab T4.
- **Results:** Highly successful. Validation Loss (0.1278) was lower than Training Loss (0.1345), showing **zero overfitting**. The log probability gap between correct and incorrect answers grew, proving the model became more confident in its legal reasoning.

### 3. GRPO on IL-TUR Dataset (Approach 1 / Future Approach)
- **Objective:** Fine-tune `Meta-Llama-3.1-8B-Instruct` to predict bail outcomes using Group Relative Policy Optimization (GRPO), faithfully adapting the **LegalΔ** paper's methodology for Hindi.
- **Method:** Generates multiple outputs for the same prompt and rewards the best ones relative to others using rule-based reward functions, eliminating the need for a massive Critic model.
- **Key Innovations:**
  - **Native Hindi System Prompt** for better bilingual legal reasoning.
  - **XML Format Reward:** Enforces strict adherence to `<reasoning>` and `<answer>` tags.
  - **Bail Accuracy Score:** +2.0 for correct, -1.0 for hallucinations.
  - **Information Gain Multiplier:** Rewards the model if the Chain-of-Thought reasoning logically increases the confidence of the correct answer (Delta reward).
- **Execution:** Due to compute constraints, this is marked as a "Future Approach" requiring High-Performance Computing (HPC/SLURM) access to run.

---

## 📂 Repository Structure

```text
RL/
├── README.MD                          ← You are here
├── Approach 1/                        ← GRPO on IL-TUR (LegalDelta Adaptation in Hindi)
│   ├── src/train_il_tur.py            ← Main GRPO training script
│   └── README.md                      ← Details on how to run Approach 1
├── Approach 2/                        ← DPO on NyayaAnumana (Preference Alignment)
│   ├── dpo_unsloth_nyananuman_legal.ipynb  ← Complete Colab notebook for Approach 2
│   └── README.md                      ← Details on how to run Approach 2
└── Future Approach/                   ← HPC Execution Workspace for GRPO
    ├── run_grpo.sh                    ← SLURM batch script for HPC
    └── README.md                      
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **PyTorch** | Deep learning framework |
| **Unsloth** | 2× faster LoRA fine-tuning |
| **TRL** | DPO and GRPO trainers |
| **Transformers / PEFT** | Model loading & LoRA adapter management |
| **BitsAndBytes** | 4-bit quantization |

## 📚 References
- **Presentation:** `Preference-Based Training for Legal Bail Prediction.pptx`
- **Paper:** [LegalΔ: Enhancing Legal Reasoning in LLMs via RL with CoT Guided Information Gain](https://arxiv.org/abs/2508.12281)
- **Datasets:** IL-TUR Dataset, NyayaAnumana Legal Corpus