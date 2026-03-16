# Layer-Pruning-Harms-Inference-Scaling

## Structure
`lib/`: Implementation for pruning strategies

`eval/`: Implementation for evaluating sequential and parallel test-time scaling

`train/`: Implementation for LoRA fine-tuning and Full fine-tuning

`analysis/`: Scripts to analyse model output

`main.py`: Script for pruning and saving models

## Install
```
conda create -n layerpruning python==3.10 -y
conda activate layerpruning
pip install -r requirements.txt
pip install -e eval/lm-evaluation-harness/
```

## Usage
### Pruning
#### ShortGPT Pruning
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method shortgpt \
    --seed 0 \
    --remove_n_layers 3 \
    --n_samples 1000 \
    --max_seq_len 1024 \
    --batch_size 8
```

#### Reverse-order prune
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method tail \
    --seed 0 \
    --remove_n_layers 3 \
```

#### LaCo
Adjust `laco_threshold` to merge certain number of layers.
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method laco \
    --calibration_data laco \
    --laco_merge_layers 2 \
    --seed 0 \
    --laco_threshold 0.9
```

#### Selection
You can also specify the layer index you want to prune by:
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method selection \
    --seed 0 \
    --layers_to_remove 25 27 26
```

### Evaluation
Input the model path and output path in ./eval/eval_sequential_scaling.sh and ./eval/eval_parallel_scaling.sh
```
sh ./eval/eval_sequential_scaling.sh
sh ./eval/eval_parallel_scaling.sh
```

### Analysis
We provide an example sampled from output of `s1.1-7B` model on `AIME24` dataset.
To run analyse on diversity, just use:
```
python ./analysis/diversity.py --jsonl_path ./analysis/example/s1.1-7B_aime24.jsonl
```
To run analyse on self-reflection, use:
```
python ./analysis/self_reflection.py --jsonl_path ./analysis/example/s1.1-7B_aime24.jsonl --max_samples 200
```
