import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .influence_helper import *
from .ablate import AblateGPT 
import numpy as np

def check_outlier_mean(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio

def check_outlier_mean(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def get_layer_sparsities(num_layers, global_sparsity, protected_range=None, protected_sparsity=0.10):
    if protected_range is None:
        return {i: global_sparsity for i in range(num_layers)}
    
    start, end = protected_range
    num_protected = end - start
    num_free = num_layers - num_protected
    
    remaining_sparsity = (num_layers * global_sparsity - num_protected * protected_sparsity) / num_free
    sparsity_per_layer = {}
    for i in range(num_layers):
        if start <= i < end:
            sparsity_per_layer[i] = protected_sparsity
        else:
            sparsity_per_layer[i] = remaining_sparsity
    
    return sparsity_per_layer



def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params


def check_sparsity_updated(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    total_count = 0
    total_params = 0

    total_mlp_count = 0
    total_mlp_params = 0
    total_attn_count = 0
    total_attn_params = 0

    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        layer_count = 0
        layer_params = 0
        layer_mlp_count = 0
        layer_mlp_params = 0
        layer_attn_count = 0
        layer_attn_params = 0

        for name, submodule in subset.items():
            W = submodule.weight.data
            n_zeros = (W == 0).sum().item()
            n_total = W.numel()

            total_count += n_zeros
            total_params += n_total
            layer_count += n_zeros
            layer_params += n_total

            if "mlp" in name:
                total_mlp_count += n_zeros
                total_mlp_params += n_total
                layer_mlp_count += n_zeros
                layer_mlp_params += n_total
            elif "self_attn" in name or "attention" in name:
                total_attn_count += n_zeros
                total_attn_params += n_total
                layer_attn_count += n_zeros
                layer_attn_params += n_total

        print(f"Layer {i:2d} | Total: {layer_count/layer_params:.4f} | "
              f"MLP: {layer_mlp_count/layer_mlp_params if layer_mlp_params>0 else 0:.4f} | "
              f"Attn: {layer_attn_count/layer_attn_params if layer_attn_params>0 else 0:.4f}")

    print("="*50)
    print(f"Global sparsity | Total: {total_count/total_params:.4f} | "
          f"MLP: {total_mlp_count/total_mlp_params:.4f} | "
          f"Attn: {total_attn_count/total_attn_params:.4f}")

    model.config.use_cache = use_cache 
    return total_count / total_params

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    '''
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        # Start of added code
        
        #def __getattr__(self, name): 
         #   try:
          #      return super().__getattr__(name)
           # except AttributeError:
            #    return getattr(self.module, name)
        '''
        # End of added code
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            # Explicitly copy this for Qwen compatibility
            self.attention_type = getattr(module, "attention_type", "full_attention") 

        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            
            #print(cache['position_ids'])
            #if cache['attention_mask'] is not None:
            #    print(f"Captured attention_mask shape: {cache['attention_mask'].shape}")
            #else:
            #    print("Attention mask was None in forward pass.")
            
            raise ValueError

        def __getattr__(self, name):
        # Delegate any missing attributes (like self_attn, mlp) to the original layer
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

            
            
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, position_embeddings 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            
            '''
            HARDCODED HERE
            if "self_attn" in name or "attention" in name:  # COMMENT THIS OUT FOR MLP PRUNING
                continue
                
            if "mlp" in name:  # COMMENT THIS OUT FOR ATTN PRUNING
                continue
            '''
            
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_mag_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    OWL-style magnitude pruning with outlier-ratio-based per-layer sparsity allocation.
    Fixed for Qwen (position_embeddings, single forward pass, no OPT branches).
    """

    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers

 
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids, position_embeddings = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev),
                position_ids.to(dev), position_embeddings.to(dev)
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Single forward pass: collects activations AND advances inps → outs
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]

        for h in handles:
            h.remove()

        # Compute per-layer wanda-style metric, then flatten for outlier check
        layer_wmetric = []
        for name in subset:
            W_metric = (torch.abs(subset[name].weight.data)
                        * torch.sqrt(wrapped_layers[name].scaler_row.reshape(1, -1)))
            layer_wmetric.append(W_metric)

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        out_ratio_layer = check_outlier_mean(layer_wmetric, args.Hyper_m)
        print(f"layer {i} outlier ratio @ {args.Hyper_m}: {out_ratio_layer}")
        all_layer_ratio.append(out_ratio_layer)

        inps, outs = outs, inps  # advance to next layer

    print("before adjustment:", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = (
        (all_layer_ratio - all_layer_ratio.min())
        * (1.0 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2)
    )
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)

    print("after adjustment:", all_layer_ratio)
    print(f"  mean={np.mean(all_layer_ratio):.4f}  "
          f"max={np.max(all_layer_ratio):.4f}  "
          f"min={np.min(all_layer_ratio):.4f}")


    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layer_sparsity_ratio = 1.0 - all_layer_ratio[i]  # convert keep→prune ratio

        for name in subset:
            print(f"pruning layer {i} name {name}  sparsity={layer_sparsity_ratio:.4f}")

            W = subset[name].weight.data
            W_metric = torch.abs(W)

            if prune_n != 0:
                # Structured N:M sparsity
                W_mask = torch.zeros_like(W, dtype=torch.bool)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # Unstructured: topk instead of sort-threshold
                num_to_prune = int(W_metric.numel() * layer_sparsity_ratio)
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                if num_to_prune > 0:
                    indices = torch.topk(W_metric.view(-1), num_to_prune, largest=False).indices
                    W_mask.view(-1)[indices] = True

            W[W_mask] = 0

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Pruning complete!")

def prune_magnitude_influence(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):

    # --- Influence-based sparsity allocation (mirrors wanda layer-level) ---
    scores_dict = influence_loader()
    attn_positive_sum, mlp_positive_sum = positive_influence_aggregation(scores_dict,num_layer=model.config.num_hidden_layers, mode="positive")
    smoothed_attn, smoothed_mlp = influence_normalize_smooth(attn_positive_sum, mlp_positive_sum,num_layer=model.config.num_hidden_layers)

    attn_dims = np.array([
        sum([getattr(layer.self_attn, name).weight.numel()
             for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']])
        for layer in model.model.layers
    ])
    mlp_dims = np.array([
        sum([getattr(layer.mlp, name).weight.numel()
             for name in ['gate_proj', 'up_proj', 'down_proj']])
        for layer in model.model.layers
    ])

    S = args.sparsity_ratio
    attn_layer_sparsity = allocate_layer_sparsity(smoothed_attn, attn_dims, target_sparsity=S)
    mlp_layer_sparsity  = allocate_layer_sparsity(smoothed_mlp,  mlp_dims,  target_sparsity=S)

  
    layers = model.model.layers

    for layer_idx, layer in enumerate(layers):
        subset = find_layers(layer)

        for name, submodule in subset.items():
            print(f"pruning layer {layer_idx} name {name}")

            W = submodule.weight.data
            W_metric = torch.abs(W)

            if prune_n != 0:
                # Structured N:M sparsity — influence allocation doesn't apply here,
                # so fall back to the original N:M logic
                W_mask = torch.zeros_like(W, dtype=torch.bool)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # Unstructured: use influence-allocated per-layer sparsity
                if "self_attn" in name or "attention" in name:
                    layer_sparsity = attn_layer_sparsity[layer_idx]
                elif "mlp" in name:
                    layer_sparsity = mlp_layer_sparsity[layer_idx]
                else:
                    layer_sparsity = args.sparsity_ratio  # fallback

                num_to_prune = int(W_metric.numel() * layer_sparsity)
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                if num_to_prune > 0:
                    indices = torch.topk(W_metric.view(-1), num_to_prune, largest=False).indices
                    W_mask.view(-1)[indices] = True

            W[W_mask] = 0

    torch.cuda.empty_cache()
    print("Pruning complete!")

#ORIGINAL WANDA FOR QWEN
def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids,position_embeddings = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids,position_embeddings = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev),position_embeddings.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,position_embeddings =position_embeddings)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            
            '''
            HARDCODED HERE
            if "self_attn" in name or "attention" in name:  # COMMENT THIS OUT FOR MLP PRUNING
                continue
                
            if "mlp" in name:  # COMMENT THIS OUT FOR ATTN PRUNING
                continue
            '''
 
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,position_embeddings =position_embeddings)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def prune_wanda_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calculate outlier ratio
    all_layer_ratio = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids, position_embeddings = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), 
                position_ids.to(dev), position_embeddings.to(dev)
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                position_ids=position_ids, position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                position_ids=position_ids, position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    
    print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    ############## prune
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids, position_embeddings = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), 
                position_ids.to(dev), position_embeddings.to(dev)
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                position_ids=position_ids, position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii+torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (alpha_hist[1]-alpha_hist[0] >= 0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                position_ids=position_ids, position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()



#This is the main code

def prune_wanda_influence(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    scores_dict = influence_loader()
    attn_positive_sum, mlp_positive_sum = positive_influence_aggregation(scores_dict, mode="positive",num_layer=model.config.num_hidden_layers)
    smoothed_attn, smoothed_mlp = influence_normalize_smooth(attn_positive_sum, mlp_positive_sum,num_layer=model.config.num_hidden_layers)
    attn_dims = np.array([
        sum([getattr(layer.self_attn, name).weight.numel()
             for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']])
        for layer in model.model.layers
    ])
    mlp_dims = np.array([
        sum([getattr(layer.mlp, name).weight.numel()
             for name in ['gate_proj', 'up_proj', 'down_proj']])
        for layer in model.model.layers
    ])
    S = args.sparsity_ratio
    attn_layer_sparsity = allocate_layer_sparsity(smoothed_attn, attn_dims, target_sparsity=S)
    mlp_layer_sparsity  = allocate_layer_sparsity(smoothed_mlp,  mlp_dims,  target_sparsity=S)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        subset = find_layers(layer)
        if f"model.layers.{layer_idx}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{layer_idx}"]
            inps, outs, attention_mask, position_ids, position_embeddings = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev),
                position_ids.to(dev), position_embeddings.to(dev)
            )

     
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Forward pass: collects activations via hooks AND advances inps → outs
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]

        for h in handles:
            h.remove()
        # --- END NEW ---

        for name, submodule in subset.items():
            print(f"pruning layer {layer_idx} name {name}")

            # --- FIXED: true wanda metric (magnitude * activation) ---
            W_metric = (torch.abs(submodule.weight.data)
                        * torch.sqrt(wrapped_layers[name].scaler_row.reshape(1, -1)))

            if "self_attn" in name or "attention" in name:
                layer_sparsity = attn_layer_sparsity[layer_idx]
            elif "mlp" in name:
                layer_sparsity = mlp_layer_sparsity[layer_idx]
            else:
                layer_sparsity = args.sparsity_ratio

            if prune_n != 0:
                # Structured N:M sparsity
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # Unstructured: influence-allocated per-layer sparsity
                num_to_prune = int(W_metric.numel() * layer_sparsity)
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                if num_to_prune > 0:
                    indices = torch.topk(W_metric.view(-1), num_to_prune, largest=False).indices
                    W_mask.view(-1)[indices] = True

            submodule.weight.data[W_mask] = 0

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Pruning complete!")



@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    

