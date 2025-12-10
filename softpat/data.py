"""
data.py
Data loading utilities for Soft Prompt Adversarial Training (SoftPAT)
"""
import pandas as pd
from typing import List, Tuple

def load_harmful_data(
    data_path: str,
    n_train: int,
    n_test: int,
    offset: int = 0
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load harmful behaviors dataset and split into train/test.
    Returns: train_goals, train_targets, test_goals, test_targets
    """
    df = pd.read_csv(data_path)
    goals = df['goal'].tolist()
    targets = df['target'].tolist()
    train_goals = goals[offset:offset + n_train]
    train_targets = targets[offset:offset + n_train]
    test_goals = goals[offset + n_train:offset + n_train + n_test]
    test_targets = targets[offset + n_train:offset + n_train + n_test]
    return train_goals, train_targets, test_goals, test_targets

def load_benign_data(
    data_path: str,
    n_train: int,
    offset: int = 0
) -> Tuple[List[str], List[str]]:
    """
    Load benign dataset for helpfulness training.
    Returns: benign_queries, benign_answers
    """
    df = pd.read_csv(data_path)
    queries = df['query'].tolist()
    answers = df['answer'].tolist()
    train_queries = queries[offset:offset + n_train]
    train_answers = answers[offset:offset + n_train]
    return train_queries, train_answers
