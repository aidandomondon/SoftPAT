#!/bin/bash
#
# Run Soft Prompt Adversarial Training (SoftPAT)
#
# Usage:
#   ./run_soft_pat.sh [options]
#
# Examples:
#   ./run_soft_pat.sh --model_path /path/to/vicuna-7b
#   ./run_soft_pat.sh --n_iterations 200 --alpha 0.3
#

# Default device
DEVICE=${CUDA_VISIBLE_DEVICES:-0}

# Run the training script
python soft_prompt_adversarial_training.py \
    --model_path "lmsys/vicuna-7b-v1.5" \
    --device "cuda:${DEVICE}" \
    --conv_template "vicuna" \
    --harmful_data_path "data/advbench/harmful_behaviors.csv" \
    --benign_data_path "data/benign/benign_test.csv" \
    --n_train_harmful 25 \
    --n_test_harmful 25 \
    --n_train_benign 25 \
    --defense_prompt_length 20 \
    --attack_prompt_length 20 \
    --n_iterations 100 \
    --alpha 0.5 \
    --lr_defense 0.01 \
    --lr_attack 0.01 \
    --batch_size 4 \
    --attack_freq 1 \
    --defense_freq 1 \
    --eval_freq 10 \
    --max_new_tokens 64 \
    --output_dir "results/soft_pat" \
    --seed 42 \
    "$@"
