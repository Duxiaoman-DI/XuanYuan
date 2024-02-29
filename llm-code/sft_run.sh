# SFT模型训练
deepspeed --num_nodes=1 --num_gpus=8 dxm_llm_main.py \
    --train_mode sft \
    --model_name_or_path model/model-pretrained/epoch0_step-120-hf \
    --save_name model/model-sft \
    --data_path fin_sft_data/fin_insurance_m_800.jsonl \
    --epochs 2 \
    --per_device_train_batch_size 4 \
    --max_length 4096 \
    --ds_zero_stage 2 \
    --log_steps 2 \
    --save_steps 5 \
    --gradient_checkpointing
