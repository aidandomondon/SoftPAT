# Soft Prompt Adversarial Training (SoftPAT)

Minimal implementation of alternating soft prompt optimization for LLM safety research.

## Overview

This system implements adversarial training between two soft prompts:
- **Defensive prompt**: Encourages refusal of harmful requests while maintaining helpfulness
- **Offensive prompt**: Attempts to elicit harmful responses

## Usage

### Quick Start
```bash
python run_experiment.py
```

### Custom Parameters
```bash
python soft_prompt_optimization.py \
    --harmful_path data/advbench/harmful_behaviors.csv \
    --benign_path data/benign/benign_test.csv \
    --iterations 20 \
    --train_samples 50 \
    --test_samples 25 \
    --alpha 0.5 \
    --prompt_length 10
```

## Parameters

- `iterations`: Number of alternating optimization steps
- `train_samples`: Number of samples for training each prompt
- `test_samples`: Number of samples for evaluation (no overlap with training)
- `alpha`: Balance parameter for defensive loss (0=only harmful, 1=only benign)
- `prompt_length`: Length of soft prompt embeddings

## Output

The system reports Attack Success Rate (ASR) - the fraction of harmful requests that receive compliant responses rather than refusals.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```