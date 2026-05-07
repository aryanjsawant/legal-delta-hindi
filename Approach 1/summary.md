# Project Summary: LegalDelta Indian Jurisdiction Adaptation (IL-TUR)

## Objective
To adapt the "LegalDelta" specific Reinforcement Learning (GRPO) framework natively established on Chinese legal data toward analyzing the Indian Legal Framework, specifically focusing on the `IL-TUR` dataset.

## Key Accomplishments

### 1. Dataset Realignment
The initial architecture parsed highly varied and multi-defendant charges against Chinese statutory citations. Upon establishing the `Exploration-Lab/IL-TUR` as the baseline dataset, we streamlined the goal toward **Bail Prediction**. Instead of messy extraction pipelines, the target was cleaned up into predicting straightforward `[Bail]Granted<eoa>` and `[Bail]Denied<eoa>` metrics from the `label` definitions.

### 2. Overcoming Language Constraints
The IL-TUR dataset processes case facts entirely in **Hindi**. 
*   **Model Pivot:** The original `train.py` mapped to `Qwen-2.5-14B` (best for CN/EN texts). We upgraded the deployment strategy to rely on **`unsloth/Meta-Llama-3.1-8B-Instruct`**, ensuring superior Hindi tokenization and lower VRAM usage suited perfectly to processing Indian regional language data.
*   **Prompt Restructuring:** Rather than feeding Hindi logic into an English reasoning system prompt—leading to translation bottlenecks—we rewrote `SYSTEM_PROMPT` completely in Hindi. This reinforces the RL model to "think" (inside the `<reasoning>` tags) naturally in its contextual language.

### 3. Creating a Decentralized HPC Monolith
HPC environments with high pending queues are prone to catastrophic failures if modular scripts break referencing dependencies halfway. We successfully decoupled the architecture from `data/`, `scripts/`, and convoluted `reward.py` imports. 

A brand new script `src/train_il_tur.py` was built containing:
*   Native HuggingFace `load_dataset()` mappings embedded within mapping iterators.
*   The `bail_accuracy_reward_func` substituting the bloated 300+ line Chinese statute regex parser.
*   The core **Information Gain (Diversity Bonus)** which defined the *LegalDelta* paper, mathematically adjusting base reward scores depending on reasoning logic vs direct answering.
*   Critical `try/except` safeguards across the Unsloth load initialization, GRPO Trainer generation, and dataset download protocols ensuring zero risk to queue disruptions.
