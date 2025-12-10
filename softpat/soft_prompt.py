"""
soft_prompt.py
SoftPromptManager class for Soft Prompt Adversarial Training (SoftPAT)
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

class SoftPromptManager:
    """
    Manages soft prompts in embedding space for adversarial training.
    The soft prompts are learned continuous vectors that replace discrete tokens.
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        defense_length: int,
        attack_length: int,
        device: str = "cuda:0"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.defense_length = defense_length
        self.attack_length = attack_length
        self.embed_dim = self._get_embed_dim()
        self.defense_prompt = self._init_soft_prompt(defense_length)
        self.attack_prompt = self._init_soft_prompt(attack_length)
        self.defense_optimizer = None
        self.attack_optimizer = None

    def _get_embed_dim(self) -> int:
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens.weight.shape[1]
        elif hasattr(self.model, 'transformer'):
            return self.model.transformer.wte.weight.shape[1]
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")

    def _get_embedding_layer(self):
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens
        elif hasattr(self.model, 'transformer'):
            return self.model.transformer.wte
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")

    def _init_soft_prompt(self, length: int) -> nn.Parameter:
        embed_layer = self._get_embedding_layer()
        embed_weight = embed_layer.weight.data
        random_indices = torch.randint(0, embed_weight.shape[0], (length,))
        init_embeds = embed_weight[random_indices].clone()
        init_embeds += torch.randn_like(init_embeds) * 0.01
        soft_prompt = nn.Parameter(init_embeds.to(self.device))
        return soft_prompt

    def setup_optimizers(self, lr_defense: float, lr_attack: float):
        self.defense_optimizer = AdamW([self.defense_prompt], lr=lr_defense)
        self.attack_optimizer = AdamW([self.attack_prompt], lr=lr_attack)

    def get_input_embeds(
        self,
        text: str,
        include_defense: bool = True,
        include_attack: bool = True
    ) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = tokens.input_ids.to(self.device)
        embed_layer = self._get_embedding_layer()
        text_embeds = embed_layer(input_ids)
        components = []
        if include_defense:
            defense_embeds = self.defense_prompt.unsqueeze(0)
            components.append(defense_embeds)
        components.append(text_embeds)
        if include_attack:
            attack_embeds = self.attack_prompt.unsqueeze(0)
            components.append(attack_embeds)
        combined_embeds = torch.cat(components, dim=1)
        return combined_embeds

    def get_prompt_positions(
        self,
        text: str,
        include_defense: bool = True,
        include_attack: bool = True
    ) -> dict:
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        text_len = tokens.input_ids.shape[1]
        positions = {}
        current_pos = 0
        if include_defense:
            positions['defense'] = slice(current_pos, current_pos + self.defense_length)
            current_pos += self.defense_length
        positions['text'] = slice(current_pos, current_pos + text_len)
        current_pos += text_len
        if include_attack:
            positions['attack'] = slice(current_pos, current_pos + self.attack_length)
            current_pos += self.attack_length
        positions['total_length'] = current_pos
        return positions
