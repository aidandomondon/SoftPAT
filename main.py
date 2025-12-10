"""
main.py
Entrypoint for Soft Prompt Adversarial Training (SoftPAT)
"""
import os
import gc
import json
import argparse
import random
from datetime import datetime
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template
from softpat.data import load_harmful_data, load_benign_data
from softpat.soft_prompt import SoftPromptManager
from softpat.training import defense_step, attack_step
from softpat.evaluation import evaluate_asr

DEFAULT_CONFIG = {
    "model_path": "lmsys/vicuna-7b-v1.5",
    "device": "cuda:0",
    "harmful_data_path": "data/advbench/harmful_behaviors.csv",
    "benign_data_path": "data/benign/benign_test.csv",
    "n_train_harmful": 25,
    "n_test_harmful": 25,
    "n_train_benign": 25,
    "data_offset": 0,
    "defense_prompt_length": 20,
    "attack_prompt_length": 20,
    "n_iterations": 100,
    "alpha": 0.5,
    "lr_defense": 0.001,  # Lowered for stability
    "lr_attack": 0.001,   # Lowered for stability
    "batch_size": 4,
    "attack_freq": 1,
    "defense_freq": 1,
    "eval_freq": 10,
    "max_new_tokens": 64,
    "output_dir": "results/soft_pat",
    "seed": 42,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Soft Prompt Adversarial Training")
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG['model_path'])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG['device'])
    parser.add_argument("--conv_template", type=str, default="vicuna")
    parser.add_argument("--harmful_data_path", type=str, default=DEFAULT_CONFIG['harmful_data_path'])
    parser.add_argument("--benign_data_path", type=str, default=DEFAULT_CONFIG['benign_data_path'])
    parser.add_argument("--n_train_harmful", type=int, default=DEFAULT_CONFIG['n_train_harmful'])
    parser.add_argument("--n_test_harmful", type=int, default=DEFAULT_CONFIG['n_test_harmful'])
    parser.add_argument("--n_train_benign", type=int, default=DEFAULT_CONFIG['n_train_benign'])
    parser.add_argument("--data_offset", type=int, default=DEFAULT_CONFIG['data_offset'])
    parser.add_argument("--defense_prompt_length", type=int, default=DEFAULT_CONFIG['defense_prompt_length'])
    parser.add_argument("--attack_prompt_length", type=int, default=DEFAULT_CONFIG['attack_prompt_length'])
    parser.add_argument("--n_iterations", type=int, default=DEFAULT_CONFIG['n_iterations'])
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG['alpha'])
    parser.add_argument("--lr_defense", type=float, default=DEFAULT_CONFIG['lr_defense'])
    parser.add_argument("--lr_attack", type=float, default=DEFAULT_CONFIG['lr_attack'])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument("--attack_freq", type=int, default=DEFAULT_CONFIG['attack_freq'])
    parser.add_argument("--defense_freq", type=int, default=DEFAULT_CONFIG['defense_freq'])
    parser.add_argument("--eval_freq", type=int, default=DEFAULT_CONFIG['eval_freq'])
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_CONFIG['max_new_tokens'])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG['output_dir'])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG['seed'])
    return parser.parse_args()

def main():
    args = parse_args()
    config = vars(args)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("=" * 60)
    print("Soft Prompt Adversarial Training (SoftPAT)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'], trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'], torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(config['device'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    conv_template = get_conversation_template(config.get('conv_template', 'vicuna'))
    print("\nLoading data...")
    train_harmful_goals, train_harmful_targets, test_harmful_goals, test_harmful_targets = load_harmful_data(
        config['harmful_data_path'], config['n_train_harmful'], config['n_test_harmful'], config['data_offset']
    )
    train_benign_queries, train_benign_answers = load_benign_data(
        config['benign_data_path'], config['n_train_benign'], config['data_offset']
    )
    print(f"  Harmful train: {len(train_harmful_goals)}")
    print(f"  Harmful test: {len(test_harmful_goals)}")
    print(f"  Benign train: {len(train_benign_queries)}")
    print("\nInitializing soft prompts...")
    soft_prompt_manager = SoftPromptManager(
        model=model,
        tokenizer=tokenizer,
        defense_length=config['defense_prompt_length'],
        attack_length=config['attack_prompt_length'],
        device=config['device']
    )
    soft_prompt_manager.setup_optimizers(
        lr_defense=config['lr_defense'],
        lr_attack=config['lr_attack']
    )
    print(f"  Defense prompt shape: {soft_prompt_manager.defense_prompt.shape}")
    print(f"  Attack prompt shape: {soft_prompt_manager.attack_prompt.shape}")
    training_log = {
        'iterations': [],
        'defense_losses': [],
        'attack_losses': [],
        'asr_with_attack': [],
        'asr_without_attack': []
    }
    print("\nInitial evaluation...")
    initial_asr, initial_results = evaluate_asr(
        model=model,
        soft_prompt_manager=soft_prompt_manager,
        tokenizer=tokenizer,
        test_goals=test_harmful_goals,
        test_targets=test_harmful_targets,
        max_new_tokens=config['max_new_tokens'],
        include_attack=True
    )
    print(f"  Initial ASR (with attack): {initial_asr:.2%}")
    print(f"\nStarting training for {config['n_iterations']} iterations...")
    from tqdm import tqdm
    for iteration in tqdm(range(config['n_iterations']), desc="Training"):
        if iteration % config['defense_freq'] == 0:
            defense_losses = defense_step(
                model=model,
                soft_prompt_manager=soft_prompt_manager,
                tokenizer=tokenizer,
                harmful_goals=train_harmful_goals,
                harmful_targets=train_harmful_targets,
                benign_queries=train_benign_queries,
                benign_answers=train_benign_answers,
                alpha=config['alpha'],
                conv_template=conv_template,
                batch_size=config['batch_size']
            )
        else:
            defense_losses = None
        if iteration % config['attack_freq'] == 0:
            attack_losses = attack_step(
                model=model,
                soft_prompt_manager=soft_prompt_manager,
                tokenizer=tokenizer,
                harmful_goals=train_harmful_goals,
                harmful_targets=train_harmful_targets,
                conv_template=conv_template,
                batch_size=config['batch_size']
            )
        else:
            attack_losses = None
        training_log['iterations'].append(iteration)
        training_log['defense_losses'].append(defense_losses)
        training_log['attack_losses'].append(attack_losses)
        if (iteration + 1) % config['eval_freq'] == 0:
            print(f"\n--- Iteration {iteration + 1} ---")
            if defense_losses:
                print(f"  Defense - Harmful: {defense_losses.get('harmful', 'N/A'):.4f}, "
                      f"Benign: {defense_losses.get('benign', 'N/A'):.4f}, "
                      f"Total: {defense_losses.get('total', 'N/A'):.4f}")
            if attack_losses:
                print(f"  Attack: {attack_losses.get('attack', 'N/A'):.4f}")
            asr_with_attack, _ = evaluate_asr(
                model=model,
                soft_prompt_manager=soft_prompt_manager,
                tokenizer=tokenizer,
                test_goals=test_harmful_goals,
                test_targets=test_harmful_targets,
                max_new_tokens=config['max_new_tokens'],
                include_attack=True
            )
            asr_without_attack, _ = evaluate_asr(
                model=model,
                soft_prompt_manager=soft_prompt_manager,
                tokenizer=tokenizer,
                test_goals=test_harmful_goals,
                test_targets=test_harmful_targets,
                max_new_tokens=config['max_new_tokens'],
                include_attack=False
            )
            print(f"  ASR (with attack): {asr_with_attack:.2%}")
            print(f"  ASR (without attack): {asr_without_attack:.2%}")
            training_log['asr_with_attack'].append((iteration + 1, asr_with_attack))
            training_log['asr_without_attack'].append((iteration + 1, asr_without_attack))
            checkpoint = {
                'iteration': iteration + 1,
                'defense_prompt': soft_prompt_manager.defense_prompt.detach().cpu(),
                'attack_prompt': soft_prompt_manager.attack_prompt.detach().cpu(),
                'asr_with_attack': asr_with_attack,
                'asr_without_attack': asr_without_attack
            }
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_iter{iteration+1}.pt'))
        if iteration % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_asr_with_attack, final_results_with = evaluate_asr(
        model=model,
        soft_prompt_manager=soft_prompt_manager,
        tokenizer=tokenizer,
        test_goals=test_harmful_goals,
        test_targets=test_harmful_targets,
        max_new_tokens=config['max_new_tokens'],
        include_attack=True
    )
    final_asr_without_attack, final_results_without = evaluate_asr(
        model=model,
        soft_prompt_manager=soft_prompt_manager,
        tokenizer=tokenizer,
        test_goals=test_harmful_goals,
        test_targets=test_harmful_targets,
        max_new_tokens=config['max_new_tokens'],
        include_attack=False
    )
    print(f"\nFinal ASR (with attack prompt): {final_asr_with_attack:.2%}")
    print(f"Final ASR (without attack prompt): {final_asr_without_attack:.2%}")
    final_results = {
        'config': config,
        'initial_asr': initial_asr,
        'final_asr_with_attack': final_asr_with_attack,
        'final_asr_without_attack': final_asr_without_attack,
        'training_log': training_log,
        'final_results_with_attack': final_results_with,
        'final_results_without_attack': final_results_without
    }
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    final_checkpoint = {
        'defense_prompt': soft_prompt_manager.defense_prompt.detach().cpu(),
        'attack_prompt': soft_prompt_manager.attack_prompt.detach().cpu()
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'final_soft_prompts.pt'))
    print(f"\nResults saved to: {output_dir}")
    return final_results

if __name__ == "__main__":
    main()
