#!/usr/bin/env python3
"""
Simple runner script for the soft prompt optimization experiment
"""
from soft_prompt_optimization import SoftPromptOptimizer

def run_quick_experiment():
    """Run a quick experiment with default parameters"""
    
    # Initialize optimizer
    optimizer = SoftPromptOptimizer(
        model_name="microsoft/DialoGPT-medium",
        prompt_length=10,
        device="cuda"
    )
    
    # Run experiment
    final_asr = optimizer.run_experiment(
        harmful_path="data/advbench/harmful_behaviors.csv",
        benign_path="data/benign/benign_test.csv",
        iterations=10,
        train_samples=20,  # Small for quick testing
        test_samples=10,
        alpha=0.5
    )
    
    return final_asr

if __name__ == "__main__":
    print("Running soft prompt optimization experiment...")
    asr = run_quick_experiment()
    print(f"Experiment completed with ASR: {asr:.3f}")