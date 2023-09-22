#!/bin/bash

cd ../src

# Chinese-Llama-2-7B
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path ziqingyang/chinese-llama-2-7b \
    --save_dir ../results/Chinese-Llama-2-7B \
    --num_few_shot $i
done

# Chinese-Llama-2-13B
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path ziqingyang/chinese-llama-2-13b \
    --save_dir ../results/Chinese-Llama-2-13B \
    --num_few_shot $i
done
