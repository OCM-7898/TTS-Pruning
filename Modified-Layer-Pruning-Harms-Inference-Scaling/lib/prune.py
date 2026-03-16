import time 
import heapq 
import torch 
import torch.nn as nn 
from copy import deepcopy
import numpy as np
from collections import OrderedDict
import tqdm
from tqdm import tqdm
import math
import random
from collections import defaultdict
import gc
from transformers import AutoConfig, AutoModelForCausalLM


def disable_kv_cache(model):
    model.config.use_cache = False
    model._use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    import types
    orig_generate = model.generate
    def no_cache_generate(self, *args, **kwargs):
        kwargs.pop("past_key_values", None)
        kwargs["use_cache"] = False
        return orig_generate(*args, **kwargs)
    model.generate = types.MethodType(no_cache_generate, model)

    orig_forward = model.forward
    def no_cache_forward(self, *args, **kwargs):
        kwargs.pop("past_key_values", None)
        return orig_forward(*args, **kwargs)
    model.forward = types.MethodType(no_cache_forward, model)

############################ Selected Layers ###################

def prune_selection(args, model, tokenizer, device):
    num_layers = len(model.model.layers)
    layers_to_remove = args.layers_to_prune
    remove_n_layers = len(layers_to_remove)

    if remove_n_layers <= 0:
        return model
    if remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all layers. Model has {num_layers} layers, attempted to remove {args.remove_n_layers}")

    layers_to_remove.sort(reverse=True)  

    print("remove layers: " + str(layers_to_remove))

    for index in layers_to_remove:
        del model.model.layers[index]

    model.config.num_hidden_layers = len(model.model.layers)
    
    return model

############################ Random ############################

def prune_random(args, model, tokenizer, device):
    """
    Randomly remove # layers from the model
    """
    num_layers = len(model.model.layers)

    if args.remove_n_layers <= 0:
        return model
    if args.remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all layers. Model has {num_layers} layers, attempted to remove {args.remove_n_layers}")

    layers_to_remove = random.sample(range(num_layers), args.remove_n_layers)
    layers_to_remove.sort(reverse=True)  

    print("remove layers: " + str(layers_to_remove))

    for index in layers_to_remove:
        del model.model.layers[index]

    model.config.num_hidden_layers = len(model.model.layers)
    
    return model


############################ Reverse Order ############################

def prune_tail(args, model, tokenizer, device):
    """
    Remove the last # layers from the model
    """
    num_layers = len(model.model.layers)

    if args.remove_n_layers <= 0:
        return model
    if args.remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all layers. Model has {num_layers} layers, attempted to remove {args.remove_n_layers}")

    for _ in range(args.remove_n_layers):
        del model.model.layers[-1]  

    model.config.num_hidden_layers = len(model.model.layers)
    
    return model


############################ Magitude ############################

def prune_magnitude_l1(args, model, tokenizer, device):
    """
    Remove # layers from the model with least magnitude of p1-norm
    """
    return _prune_by_magnitude(args, model, tokenizer, device, p_norm=1)

def prune_magnitude_l2(args, model, tokenizer, device):
    """
    Remove # layers from the model with least magnitude of p2-norm
    """
    return _prune_by_magnitude(args, model, tokenizer, device, p_norm=2)
    
def compute_weight_magnitude(param, norm_power, reduction="sum"):
    param_data = param.data if hasattr(param, 'data') else param

    if not torch.is_tensor(param_data):
        print(f"Non-tensor parameter: {type(param_data)}")
        return 1.0

    if param_data.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        param_data = param_data.float()

    if param_data.dim() >= 2:
        weight_imp = param_data.abs().pow(norm_power).sum(1)
    else:
        weight_imp = param_data.abs().pow(norm_power)
    
    if reduction == "sum":
        return weight_imp.sum().item()
    elif reduction == "mean":
        return weight_imp.mean().item()
    elif reduction == "max":
        return weight_imp.max().item()
    elif reduction == "prod":
        return torch.prod(weight_imp).item()
            
def _prune_by_magnitude(args, model, tokenizer, device, p_norm):
    num_layers = len(model.model.layers)
    assert not any(p.is_meta for p in model.parameters()), "existing meta tensor not loaded"
    
    if args.remove_n_layers <= 0:
        return model
    if args.remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all {num_layers} layers")
    
    param_info = []
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name and "embed_tokens" not in name:
            block_idx = ".".join(name.split(".")[:3])
            param_info.append((block_idx, name, param))
    
    block_scores = defaultdict(list)
    for block_idx, name, param in param_info:
        try:
            score = compute_weight_magnitude(
                param, p_norm, args.weight_reduction
            )
            block_scores[block_idx].append(score)
            print(f"{name}: {score}")
        except Exception as e:
            print(f"Failed to compute score for {name}: {str(e)}")
            block_scores[block_idx].append(1.0)  

    final_scores = {}
    for block_idx, scores in block_scores.items():
        scores_tensor = torch.tensor(scores)
        if args.weight_reduction == "sum":
            final_scores[block_idx] = scores_tensor.sum().item()
        elif args.weight_reduction == "mean":
            final_scores[block_idx] = scores_tensor.mean().item()
        elif args.weight_reduction == "max":
            final_scores[block_idx] = scores_tensor.max().item()
        elif args.weight_reduction == "prod":
            final_scores[block_idx] = torch.prod(scores_tensor).item()

    for k in ["model.norm.weight", "lm_head.weight"]:
        final_scores.pop(k, None)
    
    sorted_blocks = sorted(final_scores.items(), key=lambda x: x[1])
    print("layers score: "+str(sorted_blocks))
    block_order = [int(k.split(".")[-1]) for k, _ in sorted_blocks]
    layers_to_remove = block_order[:args.remove_n_layers]
    print("prune layers: " + str(layers_to_remove))
    
    # Remove layers (highest index first)
    for index in sorted(layers_to_remove, reverse=True):
        if 0 <= index < len(model.model.layers):
            del model.model.layers[index]
    
    model.config.num_hidden_layers = len(model.model.layers)
    return model


############################ Taylor ############################
def prune_taylor(args, model, tokenizer, text, device):
    num_layers = len(model.model.layers)
    assert not any(p.is_meta for p in model.parameters()), "existing meta tensor not loaded"
    
    if args.remove_n_layers <= 0:
        return model
    if args.remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all {num_layers} layers")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    salience_dict = {}
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name and "embed_tokens" not in name:
            salience = param * param.grad
            salience = salience.data.clone().float()
            
            if name not in salience_dict:
                salience_dict[name] = salience
            else:
                salience_dict[name] += salience
    
    model.zero_grad()

    layer_scores = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name and "embed_tokens" not in name:
            parts = name.split('.')
            if "layers" in parts:
                layer_idx = parts[parts.index("layers") + 1]
                layer_key = f"model.layers.{layer_idx}"

            if "proj" in name or "lm_head" in name:
                weight_imp = salience_dict[name].abs().pow(1).sum(1)
            elif "norm" in name:
                weight_imp = salience_dict[name].abs().pow(1)

            if args.weight_reduction == "sum":
                weight_imp = weight_imp.sum(dim=0)
            elif args.weight_reduction == "mean":
                weight_imp = weight_imp.mean(dim=0)
            elif args.weight_reduction == "max":
                weight_imp = weight_imp.max(dim=0)[0]
            elif args.weight_reduction == "prod":
                weight_imp = torch.prod(weight_imp, dim=0)
            else:
                raise NotImplementedError
            
            weight_imp = weight_imp.item()
            
            if layer_key not in layer_scores:
                layer_scores[layer_key] = [weight_imp]
            else:
                layer_scores[layer_key].append(weight_imp)
    
    final_layer_scores = {}
    for layer_key, scores in layer_scores.items():
        scores_tensor = torch.tensor(scores)
        
        if args.weight_reduction == "sum":
            layer_score = scores_tensor.sum(dim=0)
        elif args.weight_reduction == "mean":
            layer_score = scores_tensor.mean(dim=0)
        elif args.weight_reduction == "max":
            layer_score = scores_tensor.max(dim=0)[0]
        elif args.weight_reduction == "prod":
            layer_score = torch.prod(scores_tensor, dim=0)
        else:
            raise NotImplementedError
        
        final_layer_scores[layer_key] = layer_score.item()
    
    print("final layer scores: ", str(final_layer_scores))
    sorted_layers = sorted(final_layer_scores.items(), key=lambda x: x[1])
    layers_to_remove = [int(layer[0].split('.')[-1]) for layer in sorted_layers[:args.remove_n_layers]]
    print("layers to prune: ", str(layers_to_remove))
    
    # if hasattr(model, 'layers'):
    #     new_layers = [layer for i, layer in enumerate(model.layers) if i not in layers_to_prune]
    #     model.layers = torch.nn.ModuleList(new_layers)
    # else:
    #     raise NotImplementedError("Model pruning for this architecture not implemented")

    for index in sorted(layers_to_remove, reverse=True):
        if 0 <= index < len(model.model.layers):
            del model.model.layers[index]
    
    model.config.num_hidden_layers = len(model.model.layers)

    disable_kv_cache(model)
    return model


############################ PPL  ############################
# one-shot remove
def prune_ppl(args, model, tokenizer, text, device):

    @torch.no_grad()
    def calculate_ppl(model, tokenizer, text, device):
        nlls = []
        max_length = args.max_seq_len if hasattr(args, 'max_seq_len') else 128
        
        encodings = tokenizer("\n\n".join(text), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        
        for begin_loc in range(0, seq_len, max_length):
            end_loc = min(begin_loc + max_length, seq_len)
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            
            if input_ids.size(1) == 0:
                continue
                
            outputs = model(input_ids)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            nlls.append(loss)
        
        ppl = torch.exp(torch.cat(nlls).mean())
        return ppl.item()

    original_ppl = calculate_ppl(model.to(device), tokenizer, text, device)
    print("Original PPL: ", str(original_ppl))

    num_layers = len(model.model.layers)

    if args.remove_n_layers <= 0:
        return model
    if args.remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all layers. Model has {num_layers} layers, attempted to remove {args.remove_n_layers}")
        
    layer_indices = list(range(num_layers))
    num_to_remove = args.remove_n_layers

    ppl_list = []
    for i in layer_indices:
        remained = [idx for idx in layer_indices if idx != i]
        new_config = AutoConfig.from_pretrained(
            model.config.name_or_path,
            num_hidden_layers=len(remained),
            trust_remote_code=True
        )
        new_model = AutoModelForCausalLM.from_config(new_config)
        new_model.model.embed_tokens.load_state_dict(
            model.model.embed_tokens.state_dict()
        )
        new_model.model.norm.load_state_dict(
            model.model.norm.state_dict()
        )
        new_model.lm_head.load_state_dict(
            model.lm_head.state_dict()
        )
        for new_idx, old_idx in enumerate(remained):
            new_model.model.layers[new_idx].load_state_dict(
                model.model.layers[old_idx].state_dict()
            )
        ppl_i = calculate_ppl(new_model.to(device), tokenizer, text, device)
        print(f"PPL after pruning layer {i}: {ppl_i}")
        ppl_list.append(ppl_i)
        del new_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sorted_layers = sorted(enumerate(ppl_list), key=lambda x: x[1])
    layers_to_remove = [idx for idx, _ in sorted_layers[:num_to_remove]]
    print("Layers to remove: ", str(layers_to_remove))

    final_remained = [idx for idx in layer_indices if idx not in layers_to_remove]
    final_config = AutoConfig.from_pretrained(
        model.config.name_or_path,
        num_hidden_layers=len(final_remained),
        trust_remote_code=True
    )
    pruned_model = AutoModelForCausalLM.from_config(final_config)

    pruned_model.model.embed_tokens.load_state_dict(
        model.model.embed_tokens.state_dict()
    )
    pruned_model.model.norm.load_state_dict(
        model.model.norm.state_dict()
    )
    pruned_model.lm_head.load_state_dict(
        model.lm_head.state_dict()
    )

    for new_idx, old_idx in enumerate(final_remained):
        pruned_model.model.layers[new_idx].load_state_dict(
            model.model.layers[old_idx].state_dict()
        )

    final_ppl = calculate_ppl(pruned_model.to(device), tokenizer, text, device)
    print("final PPL: ", str(final_ppl))

    disable_kv_cache(pruned_model)
    return pruned_model


"""
# iteratively remove
def prune_ppl_iterative(args, model, tokenizer, text, device):
    pass
"""

"""
############################ LACO ############################

def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
   
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
    
    model_copy = deepcopy(model)
    for diff_lay in range(merge_base_lay+1, merge_base_lay+1+merge_layer_num):      
        model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data
        )
        model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data
        )
        model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data
        )
        model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.q_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data
        )
        model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.k_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data
        ) 
        model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.v_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data
        )
        model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.o_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data
        )        
                       
    for diff_lay in range(merge_base_lay+merge_layer_num, merge_base_lay, -1):
        del(model_copy.model.layers[diff_lay])

    return model_copy

def cal_last_hidden_sim(model1, model2, tokenizer, sents):
    sim_ls = []
    for s in sents:
        encoded_inputs = tokenizer(s, return_tensors='pt')
        with torch.no_grad():
            outputs1 = model1(**encoded_inputs, output_hidden_states=True)
        hidden_states1 = outputs1.hidden_states[-1] # (1, seq_len, hidden)
        with torch.no_grad():
            outputs2 = model2(**encoded_inputs, output_hidden_states=True)
        hidden_states2 = outputs2.hidden_states[-1] # (1, seq_len, hidden)
        sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
    sim_ls = [i.item() for i in sim_ls]
    print(sim_ls, np.mean(sim_ls))
    return np.mean(sim_ls)

def prune_laco(args, model, tokenizer, text, device):
    sents = []
    sents.extend(text)
    model_to_compress = deepcopy(model)

    lay = args.laco_highest_lay - args.laco_merge_layers
    last_merge_flag = False

    while lay >= args.laco_lowest_lay:
        print(lay)
        print('current model layer', len(model_to_compress.model.layers))
        tmp_merged_model = merge_layers_return_model(model_to_compress, lay, args.laco_merge_layers-1)
        sim_value = cal_last_hidden_sim(model, tmp_merged_model, tokenizer, sents)
        if sim_value > args.laco_threshold:
            model_to_compress = tmp_merged_model
            lay -= args.laco_interval
            if lay >= len(model_to_compress.model.layers):
                lay = len(model_to_compress.model.layers) - 1 - args.laco_merge_layers
        else:
            lay -= 1

    model_to_compress.config.num_hidden_layers = len(model_to_compress.model.layers)
    
    return model_to_compress
"""

############################ ShortGPT  ############################

def block_influence(input_hidden_state: torch.Tensor, output_hidden_state: torch.Tensor, angular=False):
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim

def prune_shortgpt(args, model, tokenizer, text, device):
    model.eval()
    num_layers = len(model.model.layers)

    if args.remove_n_layers <= 0:
        return model
    if args.remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all layers. Model has {num_layers} layers, attempted to remove {args.remove_n_layers}")
        
    text_batches = [
        text[i*args.batch_size : (i+1)*args.batch_size] 
        for i in range(math.ceil(len(text)/args.batch_size))
    ]
    
    all_hidden_states = []
    
    def hook_fn(module, input, output):
        all_hidden_states[-1].append(input[0].detach())
    
    for batch_text in tqdm(text_batches, desc="Processing batches"):
        batch_hidden_states = []
        all_hidden_states.append(batch_hidden_states)
        
        hooks = []
        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(hook_fn))
        
        inputs = tokenizer(batch_text, return_tensors="pt", 
                         padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        for hook in hooks:
            hook.remove()
    
    bi_scores = []
    num_layers = len(model.model.layers)
    
    for layer_idx in tqdm(range(num_layers - 1), desc="Calculating influence"):
        layer_bi = []
        for batch_states in all_hidden_states:
            bi = block_influence(batch_states[layer_idx], 
                               batch_states[layer_idx + 1])
            layer_bi.append(bi.mean().item())
        
        bi_scores.append(sum(layer_bi) / len(layer_bi))
    
    sorted_layers = sorted(enumerate(bi_scores), key=lambda x: x[1])
    print("Layer influence scores:", sorted_layers)

    num_layers_to_remove = args.remove_n_layers
    layers_to_remove = [idx for idx, _ in sorted_layers[:num_layers_to_remove]]

    for index in sorted(layers_to_remove, reverse=True):
        if 0 <= index < len(model.model.layers):
            del model.model.layers[index]
    
    model.config.num_hidden_layers = len(model.model.layers)

    disable_kv_cache(model)
    return model
