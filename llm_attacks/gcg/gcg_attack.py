import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings
import pdb
from itertools import cycle

import random
import string
random.seed(42)

import nltk
from nltk.corpus import words
nltk.download('words')

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
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

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
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
    
    return one_hot.grad.clone()

class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        print("An instance of GCGPromptManager has been created!")
        # global_dict={}
        # global_count=0


    def randomly_replace_letter(self, word):
        print("I am getting calleddddd: start from 1, repeated letter")

        print("old word:", word)

        word=word.lower()

        if len(word) == 0:
            # global_count+=1
            return word  # Return the word as is if it's empty
        
        # if len(word) == 1:
        #     return word

        replace_dict = {'a':['q', 'w', 's', 'z', 'x'],
        'b':['g', 'h', 'v', 'n'],
        'c':['d', 'f', 'x', 'v'],
        'd':['e', 'r', 's', 'f', 'x', 'c'],
        'e':['w', 'r', 's', 'd'],
        'f':['r', 't', 'd', 'g', 'c', 'v'],
        'g':['t', 'y', 'f', 'h', 'v', 'b'],
        'h':['y', 'u', 'g', 'j', 'b', 'n'],
        'i':['u', 'o', 'j', 'k'],
        'j':['u', 'i', 'h', 'k', 'n', 'm'],
        'k':['i', 'o', 'j', 'l', 'm'],
        'l':['o', 'p', 'k'],
        'm':['j', 'k', 'n'],
        'n':['h', 'j', 'b', 'm'],
        'o':['i', 'p', 'k', 'l'],
        'p':['o', 'l'],
        'q':['w', 'a'],
        'r':['e', 't', 'd', 'f'],
        's':['w', 'e', 'a', 'd', 'z', 'x'],
        't':['r', 'y', 'f', 'g'],
        'u':['y', 'i', 'h', 'j'],
        'v':['f', 'g', 'c', 'b'],
        'w':['q', 'e', 'a', 's'],
        'x':['s', 'd', 'z', 'c'],
        'y':['t', 'u', 'g', 'h'],
        'z':['a', 's', 'x']
        } #get how many times it gets a actual work and how many times it ends up with a random word
        # Get a list of valid English words
        valid_words = set(words.words())
        def replace_letter(word, index, new_letter):
            return word[:index] + new_letter + word[index + 1:]
        for i,v in enumerate(word):
            # if i==0:
            #     continue
            if v.isupper():
                for new_letter in replace_dict[v.lower()]: #this one is based on your dictionary of nearest letter on keyboard
                        new_word = replace_letter(word, i, new_letter)
                        if new_word in valid_words:
                            new_word = replace_letter(word, i, new_letter.upper())
                            # print('updatedddddddd')
                            return new_word
            else:
                for new_letter in replace_dict[v]:
                    new_word = replace_letter(word, i, new_letter)
                    if new_word in valid_words:
                        # print('updatedddddddd')
                        return new_word
        
        #if none of the words replaced are a legit word, just randomly replace
        # Choose a random position to replace
        random_position = random.randint(0, len(word) - 1)
        # # Get the letter at the chosen index
        # letter = word[random_position]
    
        # # Repeat the letter
        # new_word = word[:random_position+1] + letter + word[random_position+1:]

        if word[random_position].isupper():
            random_letter = replace_dict[word[random_position].lower()][random.randint(0, len(replace_dict[word[random_position].lower()]) - 1)]
            random_letter = random_letter.upper()
        else:
            random_letter = replace_dict[word[random_position]][random.randint(0, len(replace_dict[word[random_position]]) - 1)]
        # Replace the letter in the chosen position with the random letter
        new_word = word[:random_position] + random_letter + word[random_position + 1:]

        # print('randomly replacedddddddd')
        print("new word:", new_word)
        return new_word


    def sample_control(self, device, batch_size, indexes=[], current_goal = None):

        # print("current_goal", self.tokenizer.decode(current_goal))

        # control_toks = self.control_toks.to(grad.device)
        control_toks = torch.tensor(current_goal).to(device)
        original_control_toks = control_toks.repeat(batch_size, 1)

        indexes_tensor = torch.tensor(indexes)
        repeats = (batch_size + len(indexes) - 1) // len(indexes)  # Ceiling division
        new_token_pos = indexes_tensor.repeat(repeats)[:batch_size].to(device)

        candidates = []

        for index in new_token_pos:
            new_word = self.randomly_replace_letter(self.tokenizer.decode(current_goal[index]))
            new_tokens = self.tokenizer(new_word).input_ids[1:]
            candidate = current_goal[:index] + new_tokens + current_goal[index+1:]
            candidates.append(candidate)

        max_length = max(len(inner) for inner in candidates)
        padded_list = [inner + [0] * (max_length - len(inner)) for inner in candidates]

        new_control_toks = tensor = torch.tensor(padded_list).to(device)

        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=64, 
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
        control_cands = []


        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)
    

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=False, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad


        # pdb.set_trace()
        token_goals = self.prompts[j].tokenizer(self.goals).input_ids[0][1:]
        values_sum = torch.abs(grad).sum(dim=1)
        top_values, top_index = torch.topk(values_sum, len(token_goals), dim=0) #change top k here

        indexes = [] 
        for index in top_index.tolist():
            if self.prompts[j].tokenizer.decode(token_goals[index]).isalpha():
                indexes.append(index)


        with torch.no_grad():
            # control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii, indexes, token_goals)
            control_cand = self.prompts[j].sample_control(grad.device, batch_size, indexes, token_goals)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=False, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for fan, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    # Initialize a temporary tensor to accumulate losses for smaller batches
                    temp_loss = torch.zeros(batch_size, device=main_device)
                    # print('logits.size()',logits.size())
                    # print('ids.size()',ids.size())
                    for k, (logit, id) in enumerate(zip(logits, ids)):
                        # print('logit.size()',logit.size())
                        # print('id.size()',id.size())
                        # Split logits and ids into smaller batches
                        logit_batches = torch.chunk(logit, batch_size//8, dim=0)
                        id_batches = torch.chunk(id, batch_size//8, dim=0)
                        index=0
                        # Compute losses for each smaller batch and accumulate them
                        for logit_batch, id_batch in zip(logit_batches, id_batches):
                            end_index = index + logit_batch.size(0)
                            temp_loss[index:end_index] += target_weight * self.prompts[k][i].target_loss(logit_batch, id_batch).mean(dim=-1)
                            index=end_index
                            print("temp_loss", temp_loss)

                    # # Move accumulated losses to the appropriate device
                    # temp_loss = temp_loss.to(main_device)

                    # Add accumulated losses to the appropriate slice of the loss tensor
                    loss[fan*batch_size:(fan+1)*batch_size] += temp_loss

                    # loss[fan*batch_size:(fan+1)*batch_size] += sum([
                    #     target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1)
                    #     for k, (logit, id) in enumerate(zip(logits, ids))
                    # ])
                    # print(loss[fan*batch_size:(fan+1)*batch_size])
                    # loss[fan*batch_size:(fan+1)*batch_size] += sum([
                    #     target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                    #     for k, (logit, id) in enumerate(zip(logits, ids))
                    # ])
                    # print([
                    #     target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                    #     for k, (logit, id) in enumerate(zip(logits, ids))
                    # ])
                    # print(sum([
                    #     target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                    #     for k, (logit, id) in enumerate(zip(logits, ids))
                    # ]))
                    # print(loss[fan*batch_size:(fan+1)*batch_size])
                    if control_weight != 0:
                        loss[fan*batch_size:(fan+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[fan*batch_size:(fan+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        self.prompts[j].goal = next_control
        self.control_str = next_control
        self.goal_str = next_control

        # pdb.set_trace()
        
        self.goals = next_control
        
        del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)
        torch.cuda.empty_cache()

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
