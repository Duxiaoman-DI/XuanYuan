#!/bin/bash

cd ../src

# InternLM-7B
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path internlm/internlm-7b \
    --save_dir ../results/InternLM-7B \
    --num_few_shot $i
done

# InternLM-Chat-7B
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path internlm/internlm-chat-7b \
    --save_dir ../results/InternLM-7B-Chat \
    --num_few_shot $i
done
