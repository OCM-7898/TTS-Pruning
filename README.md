# Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling.
Our codebase used to produce the results for the paper Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling which is accepted for publication at EMNLP 2026 Findings.

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue)](https://github.com/OCM-7898/TTS-Pruning)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2604.25098)
[![GitHub Stars](https://img.shields.io/github/stars/OCM-7898/TTS-Pruning?style=social)](https://github.com/OCM-7898/TTS-Pruning/stargazers)
## Install dependencies

```
conda create -n layerpruning python==3.10 -y
conda activate layerpruning
pip install -r requirements.txt
pip install -e eval/lm-evaluation-harness/
```

## Structured Pruning

```
cd Modified-Layer-Pruning-Harms-Inference-Scaling
```

```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method shortgpt \
    --remove_n_layers 2 \
    --calibration_data shortgpt \
    --n_samples 100 \
    --max_seq_len 1024 \
    --save_model YOUR_SAVE_LOCATION
```

## Unstructured Pruning

```
cd wanda
```

For Uniform Allocation

```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method magnitude \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --save_model YOUR_SAVE_LOCATION
```

For OWL Allocation

```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method magnitude_owl \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --save_model YOUR_SAVE_LOCATION
```

For LayerIF Allocation

```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method magnitude_influence \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --save_model YOUR_SAVE_LOCATION
```

To use wanda pruning just change magnitude to wanda for the above commands.

## Evaluation
Ensure the correct filepath for model save location is given in the bash files 

```
cd Modified-Layer-Pruning-Harms-Inference-Scaling
```

```
sh ./eval/eval_sequential_scaling.sh
```













