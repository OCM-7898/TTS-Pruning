import numpy as np
import random
import torch
from datasets import load_dataset
import requests
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from bs4 import BeautifulSoup
from torch.utils.data.dataset import Dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

##############################################################################################
# Code adapted from https://github.com/yaolu-zjut/Navigation_LLM_layer_pruning

def get_c4(tokenizer, n_samples, seq_len, seed=0):
    traindata = load_dataset(
        "allenai/c4",
        "en",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        ignore_verifications=True
    )

    set_seed(seed)
    samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            sample = traindata[i]['text']
            if i not in history:
                break
        history.append(i)        
        samples.append(sample)
    return samples


def get_bookcorpus(tokenizer, n_samples, seq_len, seed=0):
    traindata = load_dataset('bookcorpus', split='train')

    set_seed(seed)
    samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            sample = traindata[i]['text']
            if i not in history:
                break
        history.append(i)        
        samples.append(sample)
    return samples


def get_calibration_data(dataset, tokenizer, n_samples, seq_len=128, seed=0):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len, seed)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len, seed)
    else:
        raise NotImplementedError


##############################################################################################
# Calibation data for reproducing laco and shortgpt

laco_sampled_text =  ['Mouron () is a commune in the Arde',
 'The 81st Mechanised Brigade () is a mechanised brigade of the Romanian Land Force',
 'There are 18 National Natural Landmarks in the U.S. state of Washington, out of nearly',
 'Torreorgaz is a municipality in the',
 'Copa Libertadores 1973 was won by defending champions Independiente of A']


def shortgpt_sampled_text(max_samples=10000, max_length=1024):
    BASE_URL = "https://storage.googleapis.com/deepmind-gutenberg/"

    try:
        hf_response = requests.get(
            "https://huggingface.co/datasets/deepmind/pg19/raw/main/data/validation_files.txt",
            timeout=10
        )
        hf_response.raise_for_status()
        hf_files = [line.strip() for line in hf_response.text.split('\n') if line.strip()]
        
        gs_response = requests.get(BASE_URL, timeout=10)
        gs_response.raise_for_status()

        soup = BeautifulSoup(gs_response.content, 'lxml')
        gs_files = [
            content.Key.text 
            for content in soup.find_all('Contents') 
            if content.Key.text.startswith('validation/') and content.Key.text.endswith('.txt')
        ]

        urls = [BASE_URL + f for f in list(set(hf_files + gs_files)) 
                if f.startswith('validation/')]
        
        if not urls:
            raise ValueError("No validation files found")
            
    except Exception as e:
        raise ConnectionError(f"Failed to get file URLs: {str(e)}")

    def process_content(text):
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len + 1 <= max_length:
                current_chunk.append(para)
                current_len += para_len + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [para]
                    current_len = para_len
                else:
                    chunks.append(para[:max_length])
                    current_len = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    print(f"Processing {len(urls)} files for {max_samples} samples (max length {max_length})...")
    all_chunks = []
    
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for url in urls:
                futures.append(executor.submit(
                    lambda u: process_content(requests.get(u, timeout=10).text), 
                    url
                ))
            
            for future in tqdm(futures, total=len(urls), desc="Downloading and processing"):
                try:
                    all_chunks.extend(future.result())
                except Exception as e:
                    print(f"Error processing file: {str(e)}")
                    continue
                    
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")

    sampled_texts = random.sample(all_chunks, min(max_samples, len(all_chunks)))
    print(f"\nSuccessfully sampled {len(sampled_texts)} text segments")
    return sampled_texts



def get_math500(n_samples=100, seed=0):

    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    
    samples, history = [], set()
    for _ in range(n_samples):
        while True:
            idx = random.randint(0, len(ds) - 1)
            if idx not in history:
                history.add(idx)
                break
        item = ds[idx]
        text = f"{item['problem']} {item.get('answer', '')}"
        samples.append(text)

    return samples

def get_gsm8k(tokenizer, n_samples, seed=0):

    ds = load_dataset('gsm8k', 'main', split='train')
    set_seed(seed)
    samples, history = [], set()
    for _ in range(n_samples):
        while True:
            idx = random.randint(0, len(ds) - 1)
            if idx not in history:
                history.add(idx)
                break
        item = ds[idx]
        text = f"{item['question']} {item['answer']}"
        samples.append(text)
    return samples

def get_gpqa(n_samples=100, seed=0):

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    
    samples, history = [], set()
    for _ in range(n_samples):
        while True:
            idx = random.randint(0, len(ds) - 1)
            if idx not in history:
                history.add(idx)
                break
        item = ds[idx]
        text = f"{item['problem']} {item.get('answer', '')}"
        samples.append(text)
    return samples


def get_math_merge(tokenizer, n_samples, seed=0):
    ds_math500 = load_dataset('HuggingFaceH4/MATH-500', split='test')
    ds_gsm8k   = load_dataset('gsm8k', 'main', split='train')
    ds_gpqa    = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

    # set for removing repetition data, elements are (ds_name, idx)
    history = set()
    samples = []

    set_seed(seed)
    datasets = [
        ('math500', ds_math500, 'problem', 'answer'),
        ('gsm8k',   ds_gsm8k,   'question', 'answer'),
        ('gpqa',    ds_gpqa,    'problem', 'answer'),
    ]

    while len(samples) < n_samples:
        ds_name, ds, q_key, a_key = random.choice(datasets)
        idx = random.randint(0, len(ds) - 1)
        if (ds_name, idx) in history:
            continue
        history.add((ds_name, idx))

        item = ds[idx]
        question = item.get(q_key, "").strip()
        answer   = item.get(a_key, "").strip()
        text = question + (" " + answer if answer else "")
        samples.append(text)

    return samples
