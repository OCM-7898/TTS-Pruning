import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def laco_merge_layers(model, merge_base_lay, merge_layer_num):
    """
    Merges `merge_layer_num` subsequent layers into a base layer `merge_base_lay`
    by averaging weights across specified components (MLP and Attention projections).
    Deletes the merged layers from the model. Adapted from https://github.com/yangyifei729/LaCo/blob/main/laco_llama-13b.ipynb.
    
    Args:
        model: the model to merge layers from (will be deepcopied internally)
        merge_base_lay: index of the base layer to merge into
        merge_layer_num: number of layers to merge into the base

    Returns:
        model_copy: the model after merging layers
    """
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
    
    model_copy = deepcopy(model)
    for diff_lay in range(merge_base_lay+1, merge_base_lay+1+merge_layer_num):      
        # gate_proj
        model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data
        )
        # down_proj
        model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data
        )
        # up_proj
        model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data
        )
        # q_proj
        model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.q_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data
        )
        # k_proj
        model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.k_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data
        ) 
        # v_proj
        model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.v_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data
        )
        # o_proj
        model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.o_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data
        )        
                       
    for diff_lay in range(merge_base_lay+merge_layer_num, merge_base_lay, -1):
        del(model_copy.model.layers[diff_lay])

    return model_copy

def cal_last_hidden_sim(model1, model2, tokenizer, sents):
    """
    Computes average cosine similarity between the last hidden states of two models
    over a list of input sentences.

    Args:
        model1, model2: models to compare
        tokenizer: tokenizer to encode inputs
        sents: list of text strings for evaluation

    Returns:
        mean similarity score across inputs
    """
    sim_ls = []
    device = next(model1.parameters()).device

    for s in sents:
        encoded_inputs = tokenizer(s, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs1 = model1(**encoded_inputs, output_hidden_states=True)
        hidden_states1 = outputs1.hidden_states[-1]

        with torch.no_grad():
            outputs2 = model2(**encoded_inputs, output_hidden_states=True)
        hidden_states2 = outputs2.hidden_states[-1]

        sim_ls.append(torch.cosine_similarity(
            hidden_states1.flatten(start_dim=1),
            hidden_states2.flatten(start_dim=1)
        ))

    sim_ls = [i.item() for i in sim_ls]
    print(sim_ls, np.mean(sim_ls))
    return np.mean(sim_ls)

def merge_laco(args, model, tokenizer, text, device):
    """
    Layer merging via a top-down greedy strategy. At each step, simulate merging
    a range of layers and accept the merge if it preserves hidden state similarity.

    Args:
        args: config with merge interval, threshold, etc.
        model: the original model
        tokenizer: tokenizer for input text
        text: text samples to evaluate similarity
        device: compute device

    Returns:
        Compressed model after iterative merging
    """
    INTERVAL = args.laco_interval
    MERGE_LAYERS = args.laco_merge_layers
    HIGHEST_LAY = len(model.model.layers) - 1
    LOWEST_LAY = 0
    THRESHOLD = args.laco_threshold
    lay = HIGHEST_LAY - MERGE_LAYERS

    sents = []
    sents.extend(text)

    model_copy_to_compress = deepcopy(model)
    while lay >= LOWEST_LAY:
        print(lay)
        print('current model layer', len(model_copy_to_compress.model.layers))
        tmp_merged_model = laco_merge_layers(model_copy_to_compress, lay, MERGE_LAYERS-1)
        sim_value = cal_last_hidden_sim(model, tmp_merged_model, tokenizer, sents)
        if sim_value > THRESHOLD:
            print("Successfully merged layers from", lay, "to", lay + MERGE_LAYERS)
            model_copy_to_compress = tmp_merged_model
            lay -= INTERVAL
            if lay >= len(model_copy_to_compress.model.layers):
                lay = len(model_copy_to_compress.model.layers) - 1 - MERGE_LAYERS
        else:
            lay -= 1

    return model_copy_to_compress
