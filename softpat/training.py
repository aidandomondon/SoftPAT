"""
training.py
Loss and training step functions for Soft Prompt Adversarial Training (SoftPAT)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from .soft_prompt import SoftPromptManager
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_loss(
    model: AutoModelForCausalLM,
    input_embeds: torch.Tensor,
    target_text: str,
    tokenizer: AutoTokenizer,
    soft_prompt_manager: SoftPromptManager,
    positions: dict,
    loss_type: str = "harmful"
) -> torch.Tensor:
    target_tokens = tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    target_ids = target_tokens.input_ids.to(input_embeds.device)
    embed_layer = soft_prompt_manager._get_embedding_layer()
    target_embeds = embed_layer(target_ids)
    full_embeds = torch.cat([input_embeds, target_embeds], dim=1)
    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits
    print(f"logits shape: {logits.shape}")
    print(f"input_embeds shape: {input_embeds.shape}")
    print(f"target_embeds shape: {target_embeds.shape}")
    print(f"full_embeds: {full_embeds}")
    input_len = input_embeds.shape[1]
    target_len = target_ids.shape[1]
    shift_logits = logits[:, input_len-1:input_len+target_len-1, :]
    shift_labels = target_ids
    # Defensive checks for NaN/Inf
    if torch.isnan(shift_logits).any() or torch.isinf(shift_logits).any():
        print("[DEBUG] NaN or Inf detected in shift_logits")
        print("shift_logits:", shift_logits)
    if torch.isnan(shift_labels.float()).any() or torch.isinf(shift_labels.float()).any():
        print("[DEBUG] NaN or Inf detected in shift_labels")
        print("shift_labels:", shift_labels)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1)
    )
    if torch.isnan(loss) or torch.isinf(loss):
        print("[DEBUG] NaN or Inf detected in loss value!")
        print("input_embeds shape:", input_embeds.shape)
        print("target_embeds shape:", target_embeds.shape)
        print("shift_logits stats: min", shift_logits.min().item(), "max", shift_logits.max().item())
        print("shift_labels:", shift_labels)
    return loss

def defense_loss(
    model: AutoModelForCausalLM,
    soft_prompt_manager: SoftPromptManager,
    tokenizer: AutoTokenizer,
    harmful_goal: str,
    harmful_target: str,
    benign_query: str,
    benign_answer: str,
    alpha: float,
    conv_template
) -> Tuple[torch.Tensor, dict]:
    losses = {}
    harmful_embeds = soft_prompt_manager.get_input_embeds(
        harmful_goal,
        include_defense=True,
        include_attack=True
    )
    harmful_positions = soft_prompt_manager.get_prompt_positions(
        harmful_goal,
        include_defense=True,
        include_attack=True
    )
    loss_harmful = compute_loss(
        model=model,
        input_embeds=harmful_embeds,
        target_text=harmful_target,
        tokenizer=tokenizer,
        soft_prompt_manager=soft_prompt_manager,
        positions=harmful_positions,
        loss_type="harmful"
    )
    losses['harmful'] = loss_harmful.item()
    benign_embeds = soft_prompt_manager.get_input_embeds(
        benign_query,
        include_defense=True,
        include_attack=False
    )
    benign_positions = soft_prompt_manager.get_prompt_positions(
        benign_query,
        include_defense=True,
        include_attack=False
    )
    loss_benign = compute_loss(
        model=model,
        input_embeds=benign_embeds,
        target_text=benign_answer,
        tokenizer=tokenizer,
        soft_prompt_manager=soft_prompt_manager,
        positions=benign_positions,
        loss_type="benign"
    )
    losses['benign'] = loss_benign.item()
    total_loss = (1 - alpha) * (-loss_harmful) + alpha * loss_benign
    losses['total'] = total_loss.item()
    return total_loss, losses

def attack_loss(
    model: AutoModelForCausalLM,
    soft_prompt_manager: SoftPromptManager,
    tokenizer: AutoTokenizer,
    harmful_goal: str,
    harmful_target: str,
    conv_template
) -> Tuple[torch.Tensor, dict]:
    losses = {}
    harmful_embeds = soft_prompt_manager.get_input_embeds(
        harmful_goal,
        include_defense=True,
        include_attack=True
    )
    harmful_positions = soft_prompt_manager.get_prompt_positions(
        harmful_goal,
        include_defense=True,
        include_attack=True
    )
    loss = compute_loss(
        model=model,
        input_embeds=harmful_embeds,
        target_text=harmful_target,
        tokenizer=tokenizer,
        soft_prompt_manager=soft_prompt_manager,
        positions=harmful_positions,
        loss_type="harmful"
    )
    losses['attack'] = loss.item()
    return loss, losses

def defense_step(
    model: AutoModelForCausalLM,
    soft_prompt_manager: SoftPromptManager,
    tokenizer: AutoTokenizer,
    harmful_goals: List[str],
    harmful_targets: List[str],
    benign_queries: List[str],
    benign_answers: List[str],
    alpha: float,
    conv_template,
    batch_size: int = 4
) -> dict:
    import random
    model.eval()
    n_harmful = len(harmful_goals)
    n_benign = len(benign_queries)
    batch_indices_harmful = random.sample(range(n_harmful), min(batch_size, n_harmful))
    batch_indices_benign = random.sample(range(n_benign), min(batch_size, n_benign))
    total_loss = 0
    all_losses = {'harmful': [], 'benign': [], 'total': []}
    soft_prompt_manager.defense_optimizer.zero_grad()
    for i, (h_idx, b_idx) in enumerate(zip(batch_indices_harmful, batch_indices_benign)):
        loss, losses = defense_loss(
            model=model,
            soft_prompt_manager=soft_prompt_manager,
            tokenizer=tokenizer,
            harmful_goal=harmful_goals[h_idx],
            harmful_target=harmful_targets[h_idx],
            benign_query=benign_queries[b_idx],
            benign_answer=benign_answers[b_idx],
            alpha=alpha,
            conv_template=conv_template
        )
        total_loss += loss
        for k, v in losses.items():
            all_losses[k].append(v)
    total_loss = total_loss / len(batch_indices_harmful)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([soft_prompt_manager.defense_prompt], max_norm=1.0)
    soft_prompt_manager.defense_optimizer.step()
    # Step the defense scheduler if present
    if hasattr(soft_prompt_manager, 'step_defense_scheduler'):
        soft_prompt_manager.step_defense_scheduler()
    avg_losses = {k: np.mean(v) for k, v in all_losses.items()}
    return avg_losses

def attack_step(
    model: AutoModelForCausalLM,
    soft_prompt_manager: SoftPromptManager,
    tokenizer: AutoTokenizer,
    harmful_goals: List[str],
    harmful_targets: List[str],
    conv_template,
    batch_size: int = 4
) -> dict:
    import random
    model.eval()
    n_harmful = len(harmful_goals)
    batch_indices = random.sample(range(n_harmful), min(batch_size, n_harmful))
    total_loss = 0
    all_losses = {'attack': []}
    soft_prompt_manager.attack_optimizer.zero_grad()
    for idx in batch_indices:
        loss, losses = attack_loss(
            model=model,
            soft_prompt_manager=soft_prompt_manager,
            tokenizer=tokenizer,
            harmful_goal=harmful_goals[idx],
            harmful_target=harmful_targets[idx],
            conv_template=conv_template
        )
        total_loss += loss
        all_losses['attack'].append(losses['attack'])
    total_loss = total_loss / len(batch_indices)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([soft_prompt_manager.attack_prompt], max_norm=1.0)
    soft_prompt_manager.attack_optimizer.step()
    # Step the attack scheduler if present
    if hasattr(soft_prompt_manager, 'step_attack_scheduler'):
        soft_prompt_manager.step_attack_scheduler()
    avg_losses = {k: np.mean(v) for k, v in all_losses.items()}
    return avg_losses
