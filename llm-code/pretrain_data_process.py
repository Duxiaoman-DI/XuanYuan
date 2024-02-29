# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional
import math
import glob
import torch
import os
from transformers import HfArgumentParser, LlamaTokenizer

# 禁止huggingface联网，加快加载本地数据集的速度
os.environ['HF_DATASETS_OFFLINE'] = '1'
import datasets


@dataclass
class DataArgs:
    model_name_or_path: str = field(default='')  # tokenizer所在目录
    data_path: str = field(default=None)  # 待预处理的数据所在目录
    save_dir: str = field(default=None)  # 保存预处理后的数据的存放目录
    max_length: Optional[int] = field(default=2048)  # 每个样本的最大长度
    cache_dir: str = field(default='')  # hf数据集缓存的目录
    num_group: Optional[str] = field(default=1000)  # concat时，每个batch包含的样本数
    num_proc: Optional[int] = field(default=32)


parser = HfArgumentParser(DataArgs)
data_args = parser.parse_args_into_dataclasses()[0]


def tokenizer_fn(tokenizer):
    def tokenize(line):
        # 使用tokenizer对text进行tokenize
        input_ids = tokenizer(
            line['text'],
            return_tensors="pt",
            return_attention_mask=False
        )['input_ids'][0]

        return {
            "input_ids": input_ids,
        }
    return tokenize


def concat_multiple_sample_fn(max_length, pad_token_id):
    def concat_multiple_sample(batch):
        # cat需要接收一个List[torch.tensor]，不接受List[list]
        concat_input_ids = torch.cat(batch['input_ids'], dim=0) 

        all_length = concat_input_ids.size(0)
        chunks = math.ceil(all_length / max_length)
        # 拼接的样本长度不足max_length的部分，使用pad_token_id进行填充
        pad_length = chunks * max_length - all_length
        pad_seq = torch.ones(pad_length, dtype=concat_input_ids.dtype) * pad_token_id 
        concat_input_ids = torch.cat([concat_input_ids, pad_seq], dim=0)

        # chunk返回一个tuple[torch.tensor]，需要转成List[torch.tensor]
        input_ids = torch.chunk(concat_input_ids, chunks) 

        return {
            "input_ids": list(input_ids)
        }
    return concat_multiple_sample


def tokenize_and_group_chunk(data_path, save_dir, tokenizer):
    # 获取目录下的所有json文件名
    filenames = glob.glob(f'{data_path}/*.json') + glob.glob(f'{data_path}/*.jsonl')

    print('\nStep1: load json dataset')
    # 第一步：加载数据集
    data = datasets.load_dataset(
        "json",
        data_files=filenames,  # 待加载的文件列表
        num_proc=data_args.num_proc,  # 并行加载的进程数
        cache_dir=data_args.cache_dir  # 数据集缓存的目录
    )['train']

    print('\nStep2: Tokenizing')
    # 第二步：对每个sample进行tokenize
    data = data.map(
        tokenizer_fn(tokenizer),
        num_proc=data_args.num_proc,
        desc='tokenize'
    )
    data = data.select_columns("input_ids")
    data.set_format(type="torch")

    print('\nStep3: concat and group')
    # 第三步：对多个sample进行concat
    concat_data = data.map(
        concat_multiple_sample_fn(data_args.max_length, tokenizer.pad_token_id),
        batched=True,  # 是否对多个sample进行concat
        batch_size=data_args.num_group,  # 每个batch包含的样本数
        num_proc=data_args.num_proc,  # 并行处理的进程数
        drop_last_batch=False,  # 是否丢弃最后一个batch，因为最后一个batch可能不足num_group
        desc='concat_group'
    )

    print('\nStep4: save to disk')
    # 第四步：将预先concat好的数据保存到本地磁盘，待后续预训练时加载
    concat_data.save_to_disk(save_dir, max_shard_size="500MB")  # 每个分片最大500MB


if __name__ == "__main__":
    # 加载tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(data_args.model_name_or_path)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token

    tokenize_and_group_chunk(data_args.data_path, data_args.save_dir, tokenizer)
