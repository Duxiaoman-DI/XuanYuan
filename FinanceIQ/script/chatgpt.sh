#!/bin/bash

cd ../src

for i in {0,5}; do
python3 -u chatgpt.py \
    --data_dir ../data \
    --save_dir ../results/ChatGPT \
    --num_few_shot $i
done
