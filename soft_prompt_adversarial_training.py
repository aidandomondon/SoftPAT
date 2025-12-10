
"""
This file has been refactored for modularity.
Please use main.py as the entrypoint for Soft Prompt Adversarial Training (SoftPAT).
All core logic has been moved to the softpat/ package.
"""
    model: AutoModelForCausalLM,
    input_embeds: torch.Tensor,
    target_text: str,
    tokenizer: AutoTokenizer,
    soft_prompt_manager: SoftPromptManager,
    positions: dict,
    loss_type: str = "harmful"  # "harmful" or "benign"
) -> torch.Tensor:
    """
    Compute cross-entropy loss for the target response.
    
    For harmful: We want to either encourage (attack) or discourage (defense) 
                 generating the harmful target.
    For benign: We want to encourage generating helpful responses.
    
    Args:
        model: The language model
        input_embeds: Combined embeddings [defense + text + attack]
        target_text: The target response to predict
        tokenizer: Tokenizer for encoding target
        soft_prompt_manager: Manager for soft prompts
        positions: Position information dict
        loss_type: Type of loss computation
        
    Returns:
        Cross-entropy loss
    """
    # Tokenize target
    target_tokens = tokenizer(
        target_text, 
        return_tensors="pt", 
        add_special_tokens=False
    )
    target_ids = target_tokens.input_ids.to(input_embeds.device)
    
    # Get embeddings for target
    embed_layer = soft_prompt_manager._get_embedding_layer()
    target_embeds = embed_layer(target_ids)
    
    # Concatenate input and target embeddings
    full_embeds = torch.cat([input_embeds, target_embeds], dim=1)
    
    # Forward pass
    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits
    
    # Compute loss only on target tokens
    # Shift logits and targets for next-token prediction
    input_len = input_embeds.shape[1]
    target_len = target_ids.shape[1]
    
    # Get logits for positions that should predict target tokens
    # logits[i] predicts token[i+1], so we need logits from input_len-1 to input_len+target_len-2
    shift_logits = logits[:, input_len-1:input_len+target_len-1, :]
    shift_labels = target_ids
    
    # Compute cross-entropy loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1)
    )
    
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
    """
    Compute defensive loss: encourage refusal on harmful + helpfulness on benign.
    
    Loss = (1 - alpha) * L_harmful + alpha * L_benign
    
    For L_harmful: We want to MAXIMIZE loss (discourage generating harmful target)
                   So we negate it in the final loss.
    For L_benign: We want to MINIMIZE loss (encourage helpful responses)
    """
    losses = {}
    
    # === Harmful Loss ===
    # Structure: [defense] [harmful_goal] [attack]
    # We want the model to NOT generate the harmful target
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
    
    # Compute loss for harmful target
    loss_harmful = compute_loss(
        model=model,
        input_embeds=harmful_embeds,
        target_text=harmful_target,
        tokenizer=tokenizer,
        soft_prompt_manager=soft_prompt_manager,
        positions=harmful_positions,
        loss_type="harmful"
    )
    
    # For defense: we want to INCREASE this loss (make it harder to generate harmful)
    # So we negate it
    losses['harmful'] = loss_harmful.item()
    
    # === Benign Loss ===
    # Structure: [defense] [benign_query] (no attack prompt)
    # We want the model to generate helpful responses
    benign_embeds = soft_prompt_manager.get_input_embeds(
        benign_query,
        include_defense=True,
        include_attack=False  # No attack prompt for benign
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
    
    # Combined defense loss
    # Minimize: -L_harmful (maximize harm loss) + L_benign (minimize benign loss)
    # Using the formula: (1-alpha)*(-L_harmful) + alpha*L_benign
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
    """
    Compute offensive loss: encourage generating harmful responses.
    
    Structure: [defense] [harmful_goal] [attack]
    We want to MINIMIZE the loss for generating harmful target.
    """
    losses = {}
    
    # Structure: [defense] [harmful_goal] [attack]
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


# =============================================================================
# Training Steps
# =============================================================================

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
    """
    Perform one defense optimization step.
    """
    model.eval()  # Model in eval mode, only optimize soft prompts
    
    # Sample a batch
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
    
    # Average the loss
    total_loss = total_loss / len(batch_indices_harmful)
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_([soft_prompt_manager.defense_prompt], max_norm=1.0)
    
    # Update defense prompt
    soft_prompt_manager.defense_optimizer.step()
    
    # Return average losses
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
    """
    Perform one attack optimization step.
    """
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
    
    # Average the loss
    total_loss = total_loss / len(batch_indices)
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_([soft_prompt_manager.attack_prompt], max_norm=1.0)
    
    # Update attack prompt
    soft_prompt_manager.attack_optimizer.step()
    
    avg_losses = {k: np.mean(v) for k, v in all_losses.items()}
    return avg_losses


# =============================================================================
# Evaluation
# =============================================================================

def generate_response(
    model: AutoModelForCausalLM,
    soft_prompt_manager: SoftPromptManager,
    tokenizer: AutoTokenizer,
    goal: str,
    include_defense: bool = True,
    include_attack: bool = True,
    max_new_tokens: int = 64
) -> str:
    """Generate a response using the soft prompts."""
    model.eval()
    
    with torch.no_grad():
        # Get input embeddings
        input_embeds = soft_prompt_manager.get_input_embeds(
            goal,
            include_defense=include_defense,
            include_attack=include_attack
        )
        
        # Generate using embeddings
        # Note: Most models require input_ids for generation, so we need a workaround
        # We'll use the model's forward pass and sample tokens manually
        
        generated_tokens = []
        current_embeds = input_embeds
        
        for _ in range(max_new_tokens):
            outputs = model(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Sample next token (greedy for evaluation)
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            
            # Get embedding of next token and append
            embed_layer = soft_prompt_manager._get_embedding_layer()
            next_embed = embed_layer(next_token.unsqueeze(0))
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        
        # Decode generated tokens
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
    return response


def is_jailbroken(response: str, prefixes: List[str] = REFUSAL_PREFIXES) -> bool:
    """Check if response indicates successful jailbreak (no refusal)."""
    response_lower = response.lower()
    for prefix in prefixes:
        if prefix.lower() in response_lower:
            return False
    return True


def evaluate_asr(
    model: AutoModelForCausalLM,
    soft_prompt_manager: SoftPromptManager,
    tokenizer: AutoTokenizer,
    test_goals: List[str],
    test_targets: List[str],
    max_new_tokens: int = 64,
    include_attack: bool = True
) -> Tuple[float, List[dict]]:
    """
    Evaluate Attack Success Rate (ASR) on test set.
    
    ASR = (# successful jailbreaks) / (# total attempts)
    """
    results = []
    n_jailbroken = 0
    
    for goal, target in tqdm(zip(test_goals, test_targets), 
                              total=len(test_goals), 
                              desc="Evaluating ASR"):
        response = generate_response(
            model=model,
            soft_prompt_manager=soft_prompt_manager,
            tokenizer=tokenizer,
            goal=goal,
            include_defense=True,
            include_attack=include_attack,
            max_new_tokens=max_new_tokens
        )
        
        jailbroken = is_jailbroken(response)
        n_jailbroken += int(jailbroken)
        
        results.append({
            'goal': goal,
            'target': target,
            'response': response,
            'jailbroken': jailbroken
        })
    
    asr = n_jailbroken / len(test_goals) if test_goals else 0.0
    
    return asr, results


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: dict):
    """Main training function."""
    
    # Set seed for reproducibility
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 60)
    print("Soft Prompt Adversarial Training (SoftPAT)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'],
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(config['device'])
    model.eval()  # Keep model frozen, only train soft prompts
    
    # Disable gradients for model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Get conversation template
    conv_template = get_conversation_template(config.get('conv_template', 'vicuna'))
    
    # Load data
    print("\nLoading data...")
    train_harmful_goals, train_harmful_targets, test_harmful_goals, test_harmful_targets = load_harmful_data(
        config['harmful_data_path'],
        config['n_train_harmful'],
        config['n_test_harmful'],
        config['data_offset']
    )
    
    train_benign_queries, train_benign_answers = load_benign_data(
        config['benign_data_path'],
        config['n_train_benign'],
        config['data_offset']
    )
    
    print(f"  Harmful train: {len(train_harmful_goals)}")
    print(f"  Harmful test: {len(test_harmful_goals)}")
    print(f"  Benign train: {len(train_benign_queries)}")
    
    # Initialize soft prompt manager
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
    
    # Training log
    training_log = {
        'iterations': [],
        'defense_losses': [],
        'attack_losses': [],
        'asr_with_attack': [],
        'asr_without_attack': []
    }
    
    # Initial evaluation
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
    
    # Training loop
    print(f"\nStarting training for {config['n_iterations']} iterations...")
    
    for iteration in tqdm(range(config['n_iterations']), desc="Training"):
        
        # Defense step
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
        
        # Attack step
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
        
        # Log losses
        training_log['iterations'].append(iteration)
        training_log['defense_losses'].append(defense_losses)
        training_log['attack_losses'].append(attack_losses)
        
        # Periodic evaluation
        if (iteration + 1) % config['eval_freq'] == 0:
            print(f"\n--- Iteration {iteration + 1} ---")
            
            if defense_losses:
                print(f"  Defense - Harmful: {defense_losses.get('harmful', 'N/A'):.4f}, "
                      f"Benign: {defense_losses.get('benign', 'N/A'):.4f}, "
                      f"Total: {defense_losses.get('total', 'N/A'):.4f}")
            
            if attack_losses:
                print(f"  Attack: {attack_losses.get('attack', 'N/A'):.4f}")
            
            # Evaluate ASR
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
            
            # Save checkpoint
            checkpoint = {
                'iteration': iteration + 1,
                'defense_prompt': soft_prompt_manager.defense_prompt.detach().cpu(),
                'attack_prompt': soft_prompt_manager.attack_prompt.detach().cpu(),
                'asr_with_attack': asr_with_attack,
                'asr_without_attack': asr_without_attack
            }
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_iter{iteration+1}.pt'))
        
        # Clear GPU cache periodically
        if iteration % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Final evaluation
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
    
    # Save final results
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
    
    # Save final soft prompts
    final_checkpoint = {
        'defense_prompt': soft_prompt_manager.defense_prompt.detach().cpu(),
        'attack_prompt': soft_prompt_manager.attack_prompt.detach().cpu()
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'final_soft_prompts.pt'))
    
    print(f"\nResults saved to: {output_dir}")
    
    return final_results


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Soft Prompt Adversarial Training")
    
    # Model settings
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG['model_path'],
                        help="Path to the model")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG['device'],
                        help="Device to use")
    parser.add_argument("--conv_template", type=str, default="vicuna",
                        help="Conversation template name")
    
    # Data settings
    parser.add_argument("--harmful_data_path", type=str, 
                        default=DEFAULT_CONFIG['harmful_data_path'],
                        help="Path to harmful behaviors CSV")
    parser.add_argument("--benign_data_path", type=str,
                        default=DEFAULT_CONFIG['benign_data_path'],
                        help="Path to benign data CSV")
    parser.add_argument("--n_train_harmful", type=int,
                        default=DEFAULT_CONFIG['n_train_harmful'],
                        help="Number of harmful samples for training")
    parser.add_argument("--n_test_harmful", type=int,
                        default=DEFAULT_CONFIG['n_test_harmful'],
                        help="Number of harmful samples for testing")
    parser.add_argument("--n_train_benign", type=int,
                        default=DEFAULT_CONFIG['n_train_benign'],
                        help="Number of benign samples for training")
    parser.add_argument("--data_offset", type=int,
                        default=DEFAULT_CONFIG['data_offset'],
                        help="Offset for data loading")
    
    # Soft prompt settings
    parser.add_argument("--defense_prompt_length", type=int,
                        default=DEFAULT_CONFIG['defense_prompt_length'],
                        help="Length of defensive soft prompt")
    parser.add_argument("--attack_prompt_length", type=int,
                        default=DEFAULT_CONFIG['attack_prompt_length'],
                        help="Length of offensive soft prompt")
    
    # Training settings
    parser.add_argument("--n_iterations", type=int,
                        default=DEFAULT_CONFIG['n_iterations'],
                        help="Number of training iterations")
    parser.add_argument("--alpha", type=float,
                        default=DEFAULT_CONFIG['alpha'],
                        help="Balancing parameter: (1-alpha)*L_harmful + alpha*L_benign")
    parser.add_argument("--lr_defense", type=float,
                        default=DEFAULT_CONFIG['lr_defense'],
                        help="Learning rate for defense prompt")
    parser.add_argument("--lr_attack", type=float,
                        default=DEFAULT_CONFIG['lr_attack'],
                        help="Learning rate for attack prompt")
    parser.add_argument("--batch_size", type=int,
                        default=DEFAULT_CONFIG['batch_size'],
                        help="Batch size for training")
    
    # Alternation settings
    parser.add_argument("--attack_freq", type=int,
                        default=DEFAULT_CONFIG['attack_freq'],
                        help="Attack step frequency")
    parser.add_argument("--defense_freq", type=int,
                        default=DEFAULT_CONFIG['defense_freq'],
                        help="Defense step frequency")
    
    # Evaluation settings
    parser.add_argument("--eval_freq", type=int,
                        default=DEFAULT_CONFIG['eval_freq'],
                        help="Evaluation frequency")
    parser.add_argument("--max_new_tokens", type=int,
                        default=DEFAULT_CONFIG['max_new_tokens'],
                        help="Max new tokens for generation")
    
    # Logging
    parser.add_argument("--output_dir", type=str,
                        default=DEFAULT_CONFIG['output_dir'],
                        help="Output directory")
    parser.add_argument("--seed", type=int,
                        default=DEFAULT_CONFIG['seed'],
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    train(config)


if __name__ == "__main__":
    main()
