# Pretrain数据预处理
python3 pretrain_data_process.py \
    --model_name_or_path ./Llama-2-7b-hf \
    --data_path ./opensource_final \
    --save_dir data/FinCorpus_tokenized \
    --max_length 4096 \
    --num_proc 128
