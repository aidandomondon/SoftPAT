import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the soft prompt.
    
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

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    # Don't need to instantiate prompt slice all the way from one-hot vector space anymore,
    # we only need embeddings, since soft prompting operates directly in the embedding space
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    input_embeds = embeds[:, input_slice.start:input_slice.stop, :]
    input_embeds.requires_grad_(True)
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
    
    # Return gradient w.r.t. embeddings, not one-hot-token vectors anymore
    return input_embeds.grad.clone()

class SoftPromptAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    # calculate grads
    def grad(self, model): 
        # print(f"attack gradients, input_slice: {self._control_slice}")
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            input_slice = self._control_slice,
            target_slice = self._target_slice,
            loss_slice = self._loss_slice
        )
    
    def def_grad(self, model): 
        # print(f"defense gradients, input_slice: {self._defense_slice}")
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            input_slice = self._defense_slice, 
            target_slice = self._target_slice,
            loss_slice = self._loss_slice
        )

    @property
    def control_embeds(self):
        pass

    @property
    def def_control_embeds(self):
        pass

class SoftPromptPromptManager(PromptManager):

    def __init__(self, lr, *args, **kwargs):
        """
        :param lr: Learning Rate
        """

        super().__init__(*args, **kwargs)
        self.lr = lr    #TODO: Move learning rate into `sample_control` to allow for dynamic scheduling

    def sample_control(self, 
                       grad, 
                       batch_size,  # don't need batch_size since we step directly in the direction of gradient (so no sampling)
                       topk=None,   # don't need topk since we step directly in the direction of gradient
                       temp=1, 
                       allow_non_ascii=True):
        """
        Gets the next soft prompt control
        """

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        # top_indices = (-grad).topk(topk, dim=1).indices
        # control_toks = self.control_toks.to(grad.device)
        control_embeds = self.control_embeds.to(grad.device)
        # original_control_toks = control_toks.repeat(batch_size, 1)
        # original_control_toks = control_toks.repeat(batch_size, 1)
        # new_token_pos = torch.arange(
        #     0, 
        #     len(control_toks), 
        #     len(control_toks) / batch_size,
        #     device=grad.device
        # ).type(torch.int64)
        # new_token_val = torch.gather(
        #     top_indices[new_token_pos], 1, 
        #     torch.randint(0, topk, (batch_size, 1),
        #     device=grad.device)
        # )
        # new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        new_control_embeds = control_embeds + self.lr * (-grad)
        return new_control_embeds

    def def_sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.def_control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        # print(original_control_toks.shape)
        # [0,
        #  len(control_toks) / batch_size,
        #  2 * len(control_toks) / batch_size,
        #  3 * len(control_toks) / batch_size,
        #  ...
        #  batch_size * len(control_toks) / batch_size]
        #
        # but batch_size |? len(control_toks)
        # in default configs, batch_size = 512 and
        # defense prompt is length 20. 512 does not divide 20. ???
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks

    @property
    def control_embeds(self):
        self._prompts[0].control_embeds

    @property
    def def_control_embeds(self):
        self._prompts[0].def_control_embeds

class SoftPromptMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        _model = self.workers[0].model
        embeds = get_embeddings(_model, input_ids.unsqueeze(0)).detach()
        input_embeds = embeds[:, input_slice.start:input_slice.stop, :]
        input_embeds.requires_grad_(True)


        self.prompts = [
            self.managers['PM'](  # PromptManager
                self.goals,
                self.targets,
                self.refuse_targets,
                self.worker.tokenizer,
                self.worker.conv_template,
                self.control_init,
                self.def_control_init,
                self.test_prefixes,
                self.managers,
                self.benign_file, 
            )
            for worker in self.workers
        ]
        
    def step(self, 
             batch_size=1024,   # don't need batch_size since we step directly in the direction of gradient (so no sampling) 
             topk=None, # don't need topk since we step directly in the direction of gradient
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device

        # Get gradient
        if len(self.workers) > 1:
            raise Exception('Support for multiple workers not yet implemented')
        worker = self.workers[0]
        worker(self.prompts[0], "grad", worker.model)
        grad = worker.results.get().to(main_device) 
        grad = grad / grad.norm(dim=-1, keepdim=True)
        next_control = self.prompts[0].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
        del grad; gc.collect()
        
        # Search
        loss = 0
        with torch.no_grad():
            worker = self.workers[0]
            worker(self.prompts[0], "logits", worker.model, next_control, return_ids=True)
            logit, id = worker.results.get()
            loss += self.prompts[0].target_loss(logit, id).mean(dim=-1).to(main_device)
            if control_weight != 0:
                loss += control_weight * self.prompts[0].control_loss(logit, id).mean(dim=-1).to(main_device)
            del logit, id ; gc.collect()

        return next_control, loss
    

    def defense_step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             benign_weight=0,
             refuse_target_weight=0,
             verbose=False, 
             opt_only=False,
             filter_cand=True,
             ):

        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "def_grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device) 
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].def_sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.def_control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].def_sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.def_control_str))
        del grad, control_cand ; gc.collect()
        # print(control_cands)
        # Search
        benign_loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        target_loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        refuse_target_loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    if target_weight != 0:
                        for k, worker in enumerate(self.workers):   # self.prompts[k][i]是PromptManager
                            worker(self.prompts[k][i], "def_logits", worker.model, cand, return_ids=True)
                        logits, ids = zip(*[worker.results.get() for worker in self.workers])
                        target_loss[j*batch_size:(j+1)*batch_size] += sum([
                            - target_weight * self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                        del logits, ids ; gc.collect()

                    if refuse_target_weight != 0:
                        for k, worker in enumerate(self.workers):  
                            worker(self.prompts[k].get_refuse_prompt(i), "def_logits", worker.model, cand, return_ids=True)
                        refuse_logits, refuse_ids = zip(*[worker.results.get() for worker in self.workers])
                        refuse_target_loss[j*batch_size:(j+1)*batch_size] += sum([
                            refuse_target_weight * self.prompts[k].get_refuse_prompt(i).target_loss(refuse_logit, refuse_id).mean(dim=-1).to(main_device)
                            for k, (refuse_logit, refuse_id) in enumerate(zip(refuse_logits, refuse_ids))
                        ])
                        del refuse_logits, refuse_ids ; gc.collect()

                    if benign_weight != 0:
                        self.prompts[k].update_benign_idx()
                        for k, worker in enumerate(self.workers):
                            worker(self.prompts[k].get_benign_prompt(i), "def_logits", worker.model, cand, return_ids=True)
                        benign_logits, benign_ids = zip(*[worker.results.get() for worker in self.workers])
                        benign_loss[j*batch_size:(j+1)*batch_size] += sum([
                            + benign_weight * self.prompts[k].get_benign_prompt(i).target_loss(benign_logit, benign_id).mean(dim=-1).to(main_device)
                            for k, (benign_logit, benign_id) in enumerate(zip(benign_logits, benign_ids))
                        ])
                        del benign_logits, benign_ids; gc.collect()
                    
                    loss = target_loss + refuse_target_loss + benign_loss

                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].def_control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()

        print('Current defense length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
