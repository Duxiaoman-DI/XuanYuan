import argparse
import deepspeed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='', help="数据所在位置")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型文件位置")
    parser.add_argument('--save_name', type=str, default='test', help='模型保存位置')

    # optimizer/lr_scheduler
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="lr scheduler的warmup步数")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")

    # 训练相关参数
    parser.add_argument("--train_mode", type=str, default='pretrain', help="训练模式：pretrain表示预训练任务，sft表示指令微调任务")
    parser.add_argument("--epochs", type=int, default=1, help="指定训练轮数")
    parser.add_argument("--total_num_steps", type=int, default=100000, help="总训练步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数",)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="最大长度")
    parser.add_argument('--gradient_checkpointing', action='store_true', help='是否开启梯度检查点，默认不开启。开启可节省GPU内存占用')
    parser.add_argument("--log_steps", type=int, default=10, help="每隔多少步记录一次日志")
    parser.add_argument("--save_steps", type=int, default=-1, help="每隔多少步保存一次模型")

    # deepspeed相关参数
    parser.add_argument('--ds_offload_cpu', action='store_true', help='是否开启cpu offload')
    parser.add_argument('--ds_zero_stage', type=int, default=2, help='deepspeed的zero配置')
    parser.add_argument('--ds_steps_per_print', type=int, default=100, help='每隔多少步输出一次deepspeed日志')

    parser.add_argument("--local_rank", type=int, default=-1, help="多机多卡情况下的local_rank")
    parser.add_argument("--global_rank", type=int, default=-1, help="多机多卡情况下的global_rank")

    # 加载deepspeed的相关参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def get_deepspeed_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,  # 每个GPU的batch_size
        'gradient_accumulation_steps': args.gradient_accumulation_steps,  # 梯度累积步数
        "steps_per_print": args.ds_steps_per_print,  # deepspeed输出中间log
        "zero_optimization": {
            "stage": args.ds_zero_stage,  # 指定zero stage，可选0,1,2,3
        },
        "scheduler": {
            "type": "WarmupDecayLR",  # 学习率衰减策略
            "params": {
                "total_num_steps": args.total_num_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.num_warmup_steps
            }
        },
        "optimizer": {
            "type": "Adam",  # 优化器
            "params": {
                "lr": args.learning_rate,  # 学习率
                "weight_decay": args.weight_decay,  # 权重衰减
            }
        },
        "fp16": {
            "enabled": True,  # 开启fp16半精度训练
        },
        "gradient_clipping": 1.0,  # 梯度裁剪
        "prescale_gradients": False,  # 是否在梯度更新前缩放梯度
        "wall_clock_breakdown": False,  # 是否输出deepspeed时间分析
    }
    return ds_config

