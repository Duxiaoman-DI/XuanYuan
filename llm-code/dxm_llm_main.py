import sys
import os
import deepspeed
import logging
import random
import numpy as np
import torch
from transformers import set_seed
from deepspeed import comm as dist
import time
from model_hook import *  # 从model_hook.py文件中加载自定义的函数


# 定义日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8'
)


def log_dist(message: str, level: int = logging.INFO) -> None:
    """定义日志函数，只给特定rank的进程记录日志"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank % 8 == 0:
        if level == logging.INFO:
            logging.info(f"[rank{my_rank}] {message}")
        if level == logging.ERROR:
            logging.error(f"[rank{my_rank}] {message}")
        if level == logging.DEBUG:
            logging.debug(f"[rank{my_rank}] {message}")


def get_ds_model(args, dataloader_dict):

    # 获取deepspeed配置
    ds_config = get_ds_config(args)

    # 加载模型
    model = get_model_common(args)
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_dist(f"Finally total_params: {total_params} trainable_params: {trainable_params} ratio {trainable_params/total_params if total_params>0 else -1:.4%} ")

    # 获取自定义的优化器和学习率调度器
    op_lr_dict = get_op_lr(args, model, dataloader_dict)
    if op_lr_dict is None:
        lr_scheduler = None
        optimizer = None
    else:
        lr_scheduler = op_lr_dict.get("lr_scheduler", None)
        optimizer = op_lr_dict.get("optimizer", None)

    # 初始化deepspeed
    model, _, _, lr_scheduler = deepspeed.initialize(
        model=model,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        config=ds_config
    )
    log_dist("deepspeed initialize finished.")

    # 设置梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


# 设置所有随机种子，保证运行结果可复现
def seed_all(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_hf_format(model, tokenizer, args, sub_folder=""):
    """
        保存模型为huggingface格式，以便后续可以用hf.from_pretrained加载
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    output_dir = os.path.join(args.save_name, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")

    state_dict = model_to_save.state_dict()
    config = model_to_save.config

    torch.save(state_dict, output_model_file)  # 保存模型权重：pytorch_model.bin
    config.to_json_file(output_config_file)  # 保存config配置文件：config.json
    tokenizer.save_pretrained(output_dir)  # 保存tokenizer

    print('=====================================')
    print(f'Model saved at: {output_dir}')
    print('=====================================')


def main():
    # 解析命令行参数
    args = parse_args()
    if args.local_rank == 0:
        # 创建保存模型的文件夹
        os.makedirs(args.save_name, exist_ok=True)

    # 设置所有随机种子，保证运行结果可复现
    seed_all(args.seed)

    # 初始化deepspeed分布式训练环境
    if args.local_rank > -1 and torch.cuda.is_available():
        # 如果是分布式训练，则使用cuda
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(f'local_rank={args.local_rank} device={device}')
        deepspeed.init_distributed()
        args.global_rank = dist.get_rank()
        print(f"global rank：{args.global_rank}  local rank: {args.local_rank}")
    else:
        # 如果不是分布式训练，则使用cpu
        device = torch.device("cpu")

    # 加载dataloader，获取训练数据
    dataloader_dict = get_dataloader_common(args)

    # 加载模型
    model = get_ds_model(args, dataloader_dict)
    model.train()  # 设置为train模式
    dataloader_dict["device"] = device

    # 在训练开始前运行用户自定义的函数
    before_train(args, model, dataloader_dict)

    if args.gradient_accumulation_steps >= 1:
        args.log_steps = args.log_steps * args.gradient_accumulation_steps
        args.save_steps = args.save_steps * args.gradient_accumulation_steps

    for epoch in range(0, args.epochs):
        dataloader_dict["sampler"].set_epoch(epoch) # 为sampler设置epoch\
        train_dataloader = dataloader_dict["train_dataloader"]
        tic = time.time()
        num_total_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}  # 将batch中的数据转移到device上
            outputs = model(use_cache=False, **batch)  # 前向计算
            loss = outputs['loss']  # 获取loss
            model.backward(loss)  # 反向传播
            model.step()  # deepspeed更新模型参数

            # 每隔一定step打印一次日志
            if step % args.log_steps == 0:
                time_per_step = (time.time() - tic) / args.log_steps
                speed = args.per_device_train_batch_size * args.max_length / time_per_step
                real_step = step
                # 如果使用了梯度累积，则需要将step除以梯度累积步数
                if args.gradient_accumulation_steps >= 1:
                    real_step = step / args.gradient_accumulation_steps

                log_dist(f"epoch{epoch} step{int(real_step)}/{num_total_steps} loss: {loss:.4f}")
                tic = time.time()  # 重置计时器
            # 每隔一定step保存一次模型
            if step > 0 and args.save_steps > 0 and step % args.save_steps == 0:
                # 保存模型
                log_dist(f'save model at epoch {epoch} step {step}')
                if args.global_rank == 0:
                    save_hf_format(
                        model, dataloader_dict['tokenizer'], args,
                        sub_folder=f'epoch{epoch}_step-{step}-hf'
                    )

            # 在每个step结束时运行用户自定义的函数
            on_step_end(args, model, dataloader_dict, step, epoch, outputs)

        # epoch结束时保存模型
        log_dist(f"save model at end of epoch {epoch}")
        if args.global_rank == 0:
            save_hf_format(model, dataloader_dict['tokenizer'], args, 
                           sub_folder=f'epoch{epoch}_step-{step}-hf'
                           )

        # 在每个epoch结束时运行用户自定义的函数
        on_epoch_end(args, model, dataloader_dict, epoch)

    log_dist("Training finished")

    # 在训练结束时运行用户自定义的函数
    after_train(args, model, dataloader_dict)


if __name__ == "__main__":
    main()
