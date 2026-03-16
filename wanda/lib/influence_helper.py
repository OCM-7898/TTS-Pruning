import pickle
import glob
import numpy as np
from scipy.signal import savgol_filter
import numpy as np

def influence_loader(): 
    all_scores = []
    # PUT IN THE FILE NAMES FOR INFLUENCE
    filenames = [
        "layer_influence_qwen3/all_influence_scores_100.pkl",
        "layer_influence_qwen3/all_influence_scores_200.pkl",
        "layer_influence_qwen3/all_influence_scores_300.pkl",
        "layer_influence_qwen3/all_influence_scores_400.pkl",
        "layer_influence_qwen3/all_influence_scores_456.pkl"  # Adjust based on your actual files
    ]
    

    for filename in filenames:
        with open(filename, 'rb') as f:
            batch = pickle.load(f)
            all_scores.extend(batch)
        print(f"Loaded {filename}: {len(batch)} samples")


    print(f"\n{'='*80}")
    print(f"Total samples loaded: {len(all_scores)}")
    print(f"{'='*80}")


    sample_ids = [item['sample_id'] for item in all_scores]
    print(f"Unique sample IDs: {len(set(sample_ids))}")
    print(f"Sample ID range: {min(sample_ids)} to {max(sample_ids)}")


    scores_dict = {item['sample_id']: item['scores'] for item in all_scores}

    print(f"\nData structure:")
    print(f"  Type: {type(scores_dict)}")
    print(f"  Keys: sample_id (0 to {len(scores_dict)-1})")
    print(f"  Values: influence scores dict (56 entries per sample)")


    print(f"\nExample - Sample 0:")
    for key, value in list(scores_dict[0].items())[:5]:
        print(f"  {key}: {value:.6f}")
    print("  ...")
    return scores_dict
    


def positive_influence_aggregation(scores_dict,num_layer = 28, mode = "positive"):
    attn_sum = np.zeros(num_layer)
    mlp_sum = np.zeros(num_layer)

    for sample_id, scores in scores_dict.items():
        for layer_idx in range(num_layer):
            attn_score = scores[f'layer_{layer_idx}_attention']
            mlp_score = scores[f'layer_{layer_idx}_mlp']
            if mode == "positive":
                if attn_score > 0:
                    attn_sum[layer_idx] += attn_score
                if mlp_score > 0:
                    mlp_sum[layer_idx] += mlp_score
            elif mode == "all":
                attn_sum[layer_idx] += attn_score
                mlp_sum[layer_idx]  += mlp_score
            elif mode == "positive_count":
                if attn_score > 0:
                    attn_sum[layer_idx] += 1
                if mlp_score > 0:
                    mlp_sum[layer_idx] += 1
            else:
                raise ValueError(f"Unknown mode: {mode}")


    print("Influence Sums - Attention:")
    print("="*60)
    for i in range(num_layer):
        print(f"Layer {i:2d}: {attn_sum[i]:.4f}")

    print("\n" + "="*num_layer)
    print("Influence Sums - MLP:")
    print("="*60)
    for i in range(num_layer):
        print(f"Layer {i:2d}: {mlp_sum[i]:.4f}")

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Attention - Total: {attn_sum.sum():.4f}, "
          f"Mean: {attn_sum.mean():.4f}, "
          f"Max: {attn_sum.max():.4f} (Layer {attn_sum.argmax()})")
    print(f"MLP       - Total: {mlp_sum.sum():.4f}, "
          f"Mean: {mlp_sum.mean():.4f}, "
          f"Max: {mlp_sum.max():.4f} (Layer {mlp_sum.argmax()})")
    return attn_sum, mlp_sum



def normalize_and_smooth(scores, window_length=5, polyorder=2):

    abs_scores = np.abs(scores)
    
    s_min = abs_scores.min()
    s_max = abs_scores.max()
    
    print(f"  Raw scores - Min: {s_min:.4f}, Max: {s_max:.4f}")
    
    if s_max - s_min > 0:
        normalized = (abs_scores - s_min) / (s_max - s_min)
    else:
        
        normalized = np.zeros_like(abs_scores)
    
    print(f"  Normalized to [0, 1]")
    

    if len(normalized) >= window_length and window_length % 2 == 1:
        smoothed = savgol_filter(normalized, window_length=window_length, polyorder=polyorder)
        print(f"  Applied Savitzky-Golay filter (window={window_length}, poly={polyorder})")
    else:
        smoothed = normalized
        print(f"  Skipped smoothing (not enough data points)")
    

    smoothed = np.clip(smoothed, 0, 1)
    
    return smoothed


def influence_normalize_smooth(attn_positive_sum, mlp_positive_sum,num_layer = 28):
    print("Processing Attention Scores:")
    print("="*60)
    smoothed_attn = normalize_and_smooth(attn_positive_sum, window_length=5, polyorder=2)


    print("\n" + "="*60)
    print("Processing MLP Scores:")
    print("="*60)
    smoothed_mlp = normalize_and_smooth(mlp_positive_sum, window_length=5, polyorder=2)

    print("\n" + "="*60)
    print("Smoothed Attention Scores:")
    print("="*60)
    for i in range(num_layer):
        print(f"Layer {i:2d}: {smoothed_attn[i]:.6f} (raw: {attn_positive_sum[i]:.4f})")
    print("\n" + "="*60)
    print("Smoothed MLP Scores:")
    print("="*60)
    for i in range(num_layer):
        print(f"Layer {i:2d}: {smoothed_mlp[i]:.6f} (raw: {mlp_positive_sum[i]:.4f})")
    return smoothed_attn, smoothed_mlp

def allocate_layer_sparsity(
    scores,
    layer_dims,
    target_sparsity=0.20,
    e1=0.05,
    e2=0.40,
    eps=1e-8,
):
   
    scores = np.asarray(scores)
    layer_dims = np.asarray(layer_dims)

    assert scores.ndim == 1
    assert layer_dims.ndim == 1
    assert len(scores) == len(layer_dims)


    s_min, s_max = scores.min(), scores.max()
    s_norm = (scores - s_min) / (s_max - s_min + eps)



    p_hat = (1.0 - s_norm) * (e2 - e1) + e1

    total_params = layer_dims.sum()
    weighted_sum = np.sum(p_hat * layer_dims)

    eta = (target_sparsity * total_params) / (weighted_sum + eps)

 
    p = eta * p_hat
    p = np.clip(p, 0.0, 1.0)

    return p
