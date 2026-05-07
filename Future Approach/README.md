# Future Approach: GRPO on IL-TUR (HPC Execution Workspace)

## Overview

This directory acts as the workspace for executing our GRPO (Group Relative Policy Optimization) adaptation for Hindi legal reasoning on an HPC (High-Performance Computing) cluster.

Because GRPO requires the model to generate multiple output candidates simultaneously to compute relative advantages (and calculate Information Gain/Delta rewards), it is highly VRAM-intensive. While DPO (Approach 2) can easily run on a free Google Colab T4, our GRPO implementation using `Meta-Llama-3.1-8B-Instruct` requires a more robust GPU environment (e.g., NVIDIA H100 or multi-GPU A100 setups).

This folder contains the SLURM batch execution scripts and logs mapping to the code inside the `Approach 1` directory.

## Execution via SLURM

To run the GRPO training job on an HPC cluster equipped with SLURM workload manager:

1. Ensure the training script and dataset are ready (refer to `Approach 1`).
2. Submit the job using the provided batch script:
   ```bash
   sbatch run_grpo.sh
   ```
3. Monitor your job's progress by tailing the output and error logs:
   ```bash
   tail -f logs_<job_id>.out
   tail -f logs_<job_id>.err
   ```