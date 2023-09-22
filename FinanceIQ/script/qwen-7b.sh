#!/bin/bash

cd ../src

# Qwen-7B (base)
for i in {5,0}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path Qwen/Qwen-7B \
    --save_dir ../results/Qwen-7B \
    --num_few_shot $i
done

# Qwen-7B-Chat
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path Qwen/Qwen-7B-Chat \
    --save_dir ../results/Qwen-7B-Chat \
    --num_few_shot $i
done

