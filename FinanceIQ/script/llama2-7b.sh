#!/bin/bash

cd ../src

# LLaMA2-7B-Base
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path meta-llama/Llama-2-7b-base-hf \
    --save_dir ../results/LLaMA2-7B-Base \
    --num_few_shot $i
done

# LLaMA2-7B-Chat
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --save_dir ../results/LLaMA2-7B-Chat \
    --num_few_shot $i
done

# LLaMA2-13B-Base
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path meta-llama/Llama-2-13b-base-hf \
    --save_dir ../results/LLaMA2-13B-Base \
    --num_few_shot $i
done

# LLaMA2-13B-Chat
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --save_dir ../results/LLaMA2-13B-Chat \
    --num_few_shot $i
done

