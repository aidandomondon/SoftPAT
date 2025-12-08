import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings


def embedding_gradients(model, input_ids, input_slice, target_slice, loss_slice, soft_embeds):
    """
    Computes gradients of the loss with respect to continuous embeddings.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.
    soft_embeds : torch.Tensor
        The continuous embeddings to optimize, shape [num_positions, embed_dim].

    Returns
    -------
    torch.Tensor
        The gradients of the soft embeddings with respect to the loss.
    """
    
    soft_embeds = soft_embeds.clone().requires_grad_()
    input_embeds = soft_embeds.unsqueeze(0)
    
    # Stitch with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    return soft_embeds.grad.clone(), loss.item()


class SoftPromptAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize soft embeddings from current tokens
        self.soft_control_embeds = None
        self.soft_defense_embeds = None
    
    def init_soft_embeds(self, model, slice_type='control'):
        """Initialize soft embeddings from current token embeddings."""
        embed_weights = get_embedding_matrix(model)
        if slice_type == 'control':
            token_ids = self.input_ids[self._control_slice]
            self.soft_control_embeds = embed_weights[token_ids].clone().detach()
            return self.soft_control_embeds
        else:  # defense
            token_ids = self.input_ids[self._defense_slice]
            self.soft_defense_embeds = embed_weights[token_ids].clone().detach()
            return self.soft_defense_embeds
    
    def grad(self, model):
        if self.soft_control_embeds is None:
            self.init_soft_embeds(model, 'control')
        grad, loss = embedding_gradients(
            model, 
            self.input_ids.to(model.device), 
            input_slice=self._control_slice,
            target_slice=self._target_slice,
            loss_slice=self._loss_slice,
            soft_embeds=self.soft_control_embeds
        )
        return grad, loss
    
    def def_grad(self, model):
        if self.soft_defense_embeds is None:
            self.init_soft_embeds(model, 'defense')
        grad, loss = embedding_gradients(
            model, 
            self.input_ids.to(model.device), 
            input_slice=self._defense_slice, 
            target_slice=self._target_slice,
            loss_slice=self._loss_slice,
            soft_embeds=self.soft_defense_embeds
        )
        return grad, loss


class SoftPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_control_embeds = None
        self.soft_defense_embeds = None

    def update_soft_control(self, grad, lr=0.01):
        """Take a gradient step on soft control embeddings."""
        if self.soft_control_embeds is None:
            raise ValueError("Soft control embeddings not initialized")
        self.soft_control_embeds = self.soft_control_embeds - lr * grad
        return self.soft_control_embeds

    def update_soft_defense(self, grad, lr=0.01):
        """Take a gradient step on soft defense embeddings."""
        if self.soft_defense_embeds is None:
            raise ValueError("Soft defense embeddings not initialized")
        self.soft_defense_embeds = self.soft_defense_embeds - lr * grad
        return self.soft_defense_embeds


class SoftPromptMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, 
             lr=0.01,
             target_weight=1, 
             control_weight=0.1, 
             verbose=False):

        main_device = self.models[0].device

        # Initialize soft embeddings if needed
        for j, prompt in enumerate(self.prompts):
            if prompt[0].soft_control_embeds is None:
                for p in prompt:
                    p.init_soft_embeds(self.models[j % len(self.models)], 'control')

        # Compute gradients
        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        total_loss = 0
        for j, worker in enumerate(self.workers):
            new_grad, loss = worker.results.get()
            new_grad = new_grad.to(main_device)
            total_loss += loss
            if grad is None:
                grad = new_grad
            else:
                grad += new_grad

        # Average gradient
        grad = grad / len(self.workers)
        
        # Update soft embeddings
        with torch.no_grad():
            for prompt in self.prompts:
                for p in prompt:
                    p.soft_control_embeds = p.soft_control_embeds - lr * grad

        avg_loss = total_loss / len(self.workers)
        
        if verbose:
            print(f'Step loss: {avg_loss:.4f}')

        return None, avg_loss

    def defense_step(self, 
             lr=0.01,
             target_weight=1, 
             control_weight=0.1, 
             benign_weight=0,
             refuse_target_weight=0,
             verbose=False):

        main_device = self.models[0].device

        # Initialize soft embeddings if needed
        for j, prompt in enumerate(self.prompts):
            if prompt[0].soft_defense_embeds is None:
                for p in prompt:
                    p.init_soft_embeds(self.models[j % len(self.models)], 'defense')

        # Compute gradients
        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "def_grad", worker.model)

        # Aggregate gradients
        grad = None
        total_loss = 0
        for j, worker in enumerate(self.workers):
            new_grad, loss = worker.results.get()
            new_grad = new_grad.to(main_device)
            total_loss += loss
            if grad is None:
                grad = new_grad
            else:
                grad += new_grad

        # Average gradient and negate for defense (maximize target loss)
        grad = -grad / len(self.workers)
        
        # Update soft embeddings
        with torch.no_grad():
            for prompt in self.prompts:
                for p in prompt:
                    p.soft_defense_embeds = p.soft_defense_embeds - lr * grad

        avg_loss = total_loss / len(self.workers)
        
        if verbose:
            print(f'Defense step loss: {avg_loss:.4f}')

        return None, avg_loss
