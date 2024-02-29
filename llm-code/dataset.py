
import torch
import json
from dataclasses import dataclass
from datasets import load_from_disk
from dxm_llm_main import log_dist


class JsonlDatasetPT(torch.utils.data.Dataset):
    """
        用于加载jsonl格式的数据集，用于预训练任务。
    """
    def __init__(self,
                 data_path,  # 数据集路径
                 tokenizer,  # 分词器实例
                 max_length,  # 最大长度
                 ):

        # 加载数据集并进行tokenize
        self.dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = json.loads(line)['text']
                # 使用tokenizer对句子进行tokenize
                inputs = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt',
                    truncation=True
                )
                input_ids = inputs['input_ids'].squeeze()  # shape: [max_length]

                # 将tokenize后的样本添加到dataset中
                self.dataset.append({
                    'input_ids': input_ids,
                })

        log_dist(f'Loaded {len(self.dataset)} examples from {data_path}')

    def __len__(self):
        # 返回数据集大小
        return len(self.dataset)

    def __getitem__(self, idx):
        # 返回一个样本
        return self.dataset[idx]


def get_pt_dataset(args):
    """
        用于加载已tokenize后的数据集，用于预训练任务。
    """
    # 从磁盘加载数据集，注意该数据集必须是通过save_to_disk()函数保存的
    train_dataset = load_from_disk(args.data_path)
    train_dataset = train_dataset.shuffle(seed=42)
    return train_dataset


class JsonDatasetSFT(torch.utils.data.Dataset):
    """
        用于加载json格式的数据集，用于指令微调任务。
    """
    def __init__(self,
                 data_path,  # 数据集路径
                 tokenizer,  # 分词器实例
                 max_length,  # 最大长度
                 ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.data = []
        with open(data_path, 'r') as file:
            for line in file:
                sample = json.loads(line)
                self.data.append({
                    "prompt": sample['instruction'],
                    "response": sample['response'],
                })
        log_dist(f'Loaded {len(self.data)} examples from {data_path}')

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 返回一个样本
        prompt = self.data[idx]['prompt']
        response = self.data[idx]['response']
        prompt = f"Human: {prompt}\nAssistant: "

        # 使用tokenizer对句子进行tokenize
        prompt_ids = self.tokenizer(prompt).input_ids
        response_ids = self.tokenizer(response).input_ids

        # prompt部分对应的label应为-100，表示不计算该部分的loss
        input_ids = prompt_ids + [self.eos_token_id] + response_ids + [self.eos_token_id]
        labels = [-100] * (len(prompt_ids) + 1) + response_ids + [self.eos_token_id] 

        if len(input_ids) > self.max_length:
            # 超长的截断
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        else:
            # 不足的填充padding至max_length
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len
            labels += [self.pad_token_id] * pad_len

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


@dataclass
class DataCollatorForPT(object):
    """
        Data collator函数，用于将多个样本拼接成一个batch，同时生成labels，用于计算loss。
        该函数用于pretrain模式。
    """
    pad_token_id: int = 0
    ignore_index: int = -100
    max_length: int = -1  # 默认不进行max_length截断

    def __call__(self, instances: list) -> dict:
        if self.max_length > 0:
            input_ids = torch.stack([instance['input_ids'][:self.max_length] for instance in instances], dim=0)  # shape: [batch_size, max_length]
        else:
            input_ids = torch.stack([instance['input_ids'] for instance in instances], dim=0)  # shape: [batch_size, max_length]
        labels = input_ids.clone()
        # 将labels中的pad部分置为ignore_index，计算loss时要忽略
        labels[labels == self.pad_token_id] = self.ignore_index 
        return dict(
            input_ids=input_ids,
            labels=labels,
        )
