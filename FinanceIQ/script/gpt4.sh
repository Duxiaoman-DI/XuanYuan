#!/bin/bash
cd ../src

for i in {0,5}; do
python3 -u gpt4.py \
    --data_dir ../data \
    --save_dir ../results/GPT4\
    --num_few_shot $i
done
