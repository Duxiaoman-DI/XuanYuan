# Pretrain模型训练
deepspeed --num_nodes=1 --num_gpus=8 dxm_llm_main.py \
    --train_mode pretrain \
    --model_name_or_path ./Llama-2-7b-hf \
    --save_name model/model-pretrained \
    --data_path data/FinCorpus_tokenized \
    --epochs 1 \
    --per_device_train_batch_size 4 \
    --max_length 4096 \
    --ds_zero_stage 2 \
    --log_steps 2 \
    --save_steps 40 \
    --gradient_checkpointing
