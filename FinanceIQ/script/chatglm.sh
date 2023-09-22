#!/bin/bash

cd ../src

# ChatGLM
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path THUDM/chatglm-6b \
    --save_dir ../results/ChatGLM-6B \
    --num_few_shot $i
done


# ChatGLM-2
for i in {0,5}; do
python -u hf_causal_model.py \
    --data_dir ../data \
    --model_name_or_path THUDM/chatglm2-6b \
    --save_dir ../results/ChatGLM2-6B \
    --num_few_shot $i
done
