#!/bin/bash

cd ../src

# Baichuan-7B
for i in {0,5}; do
python -u hf_causal_model.py \
    --model_name_or_path baichuan-inc/Baichuan-7B \
    --save_dir ../results/Baichuan-7B-Base \
    --num_few_shot $i
done

# Baichuan-13B-Base
for i in {0,5}; do
python -u hf_causal_model.py \
    --model_name_or_path baichuan-inc/Baichuan-13B-Base \
    --save_dir ../results/Baichuan-13B-Base \
    --num_few_shot $i
done

# Baichuan-13B-Chat
for i in {0,5}; do
python -u hf_causal_model.py \
    --model_name_or_path baichuan-inc/Baichuan-13B-Chat \
    --save_dir ../results/Baichuan-13B-Chat \
    --num_few_shot $i
done

# Baichuan2-7B-Base
for i in {0,5}; do
python -u hf_causal_model.py \
    --model_name_or_path baichuan-inc/Baichuan2-7B-Base \
    --save_dir ../results/Baichuan2-7B-Base \
    --num_few_shot $i
done

# Baichuan2-7B-Chat
for i in {0,5}; do
python -u hf_causal_model.py \
    --model_name_or_path baichuan-inc/Baichuan2-7B-Chat \
    --save_dir ../results/Baichuan2-7B-Chat \
    --num_few_shot $i
done

# Baichuan2-13B-Base
for i in {0,5}; do
python -u hf_causal_model.py \
    --model_name_or_path baichuan-inc/Baichuan2-13B-Base \
    --save_dir ../results/Baichuan2-13B-Base \
    --num_few_shot $i
done

# Baichuan2-13B-Chat
for i in {0,5}; do
python -u hf_causal_model.py \
    --model_name_or_path baichuan-inc/Baichuan2-13B-Chat \
    --save_dir ../results/Baichuan2-13B-Chat \
    --num_few_shot $i
done


