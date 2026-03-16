import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from typing import List, Dict, Any
from lib.prune import prune_shortgpt, prune_random, prune_tail, prune_magnitude_l1, prune_magnitude_l2, prune_ppl, prune_taylor, prune_selection
from lib.merge import merge_laco
from lib.data import laco_sampled_text, shortgpt_sampled_text, get_c4, get_bookcorpus
import random
# Two lines Changed
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map=None, #device_map="auto"
        offload_folder=None, 
        offload_state_dict=False 
    )

    model.seqlen = model.config.max_position_embeddings 
    return model


def merge_zero_shot_results(
    json_list: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:

    merged: Dict[str, Any] = {}

    for idx, item in enumerate(json_list):
        if "results" not in item:
            raise KeyError(f"# {idx} lacks key 'results' ")
        sub = item["results"]
        if not isinstance(sub, dict):
            raise ValueError(f"# {idx} element is not 'results' object")

        merged.update(sub)

    return {"results": merged}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--prune_method", type=str, default="shortgpt", choices=["laco", "shortgpt", "tail", "selection"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument('--laco_interval', type=int, default=1, help='laco interval')
    parser.add_argument('--laco_merge_layers', type=int, default=7, help='laco merge layers')
    parser.add_argument('--laco_highest_lay', type=int, default=39, help='laco highest lay')
    parser.add_argument('--laco_lowest_lay', type=int, default=10, help='laco lowest lay')
    parser.add_argument('--laco_threshold', type=float, default=0.45, help='laco threshold')
    parser.add_argument('--weight_reduction', type=str, default="sum", help='weight_reduction for magnitude, taylor')
    parser.add_argument('--remove_n_layers', type=int, default=0, help='prune number of layers')

    parser.add_argument('--calibration_data', type=str, default="bookcorpus", choices=["bookcorpus", "c4", "shortgpt", "laco"], help='calibration data')
    parser.add_argument("--n_samples", type=int, default=100, help="number of texts of calibration data")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size of calibration data, eval data")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="max sequence lenghth")

    parser.add_argument(
        "--layers_to_remove", "-n",
        nargs="+",          
        type=int,           
        help="input a list of integers e.g., --layers_to_remove 1 2 3 4"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = model.to("cuda:0") #model = model.to(device)
    print("use device ", device)
    print("prune method: " + args.prune_method)

    if args.calibration_data == "bookcorpus":
        text = get_bookcorpus(tokenizer, args.n_samples, args.max_seq_len, args.seed)
    elif args.calibration_data == "c4":
        text = get_c4(tokenizer, args.n_samples, args.max_seq_len, args.seed)
    elif args.calibration_data == "laco":
        text = laco_sampled_text
    elif args.calibration_data == "shortgpt":
        text = shortgpt_sampled_text(args.n_samples, args.max_seq_len)
    else:
        raise NotImplementedError

    if args.prune_method == "tail":
        compressed_model = prune_tail(args, model, tokenizer, device)
    elif args.prune_method == "shortgpt":
        compressed_model = prune_shortgpt(args, model, tokenizer, text, device)
    elif args.prune_method == "laco":
        compressed_model = merge_laco(args, model, tokenizer, text, device)
    elif args.prune_method == "selection":
        compressed_model = prune_selection(args, model, tokenizer, device)    
    else:
        raise NotImplementedError
    
    print("# of layers of the compressed model: " + str(len(compressed_model.model.layers)))

    if args.save_model:
        compressed_model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
