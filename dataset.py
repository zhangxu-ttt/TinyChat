from typing import List, Dict, Optional, Union
from pathlib import Path
import random
import json

import torch
import pandas as pd
from torch.utils.data import Dataset
from typing_extensions import override

from utils import kmp_search, read_jsonl

class TextDataset(Dataset):
    def __init__(
            self,
            data_path: Union[str, List[str]],
            tokenizer,
            max_length: int = 256
    ):
        """
        Args:
            data_path: 文件路径
            tokenizer: 分词器 (AutoTokenizer)
            max_length: 最大序列长度
                - conversation: JSONL格式对话数据
                - text: 纯文本数据
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = self.load_data(data_path)

    def load_data(self, data_path: Union[str, List[str]]) -> List[str]:
        if isinstance(data_path, str):
            texts = read_jsonl(data_path)
        else:
            texts = []
            for path in data_path:
                texts += read_jsonl(path)
        return texts

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        sample = self.data[idx]

        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        # 使用 attention_mask 而不是 (input_ids != pad_token_id)
        # 原因：当 tokenizer.pad_token 被设成 eos_token 时，
        # (input_ids != pad_token_id) 会把真实 eos 也当成 padding 屏蔽掉。
        loss_mask = encoding.attention_mask.squeeze().to(dtype=torch.bool)

        x = input_ids[:-1]
        y = input_ids[1:]

        return {
            'x': x,
            'y': y,
            'loss_mask': loss_mask[1:],
        }

class ChatMLDataset(Dataset):
    def __init__(
            self,
            data_path: Union[str, List[str]],
            tokenizer,
            max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = self.load_data(data_path)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.answer_start_token = f"<|im_start|>assistant\n"
        self.answer_start_token_id_list = None
        self.answer_end_token_id_list = [self.eos_token_id]

    def load_data(self, data_path: Union[str, List[str]]) -> List[str]:
        if isinstance(data_path, str):
            texts = read_jsonl(data_path)
        else:
            texts = []
            for path in data_path:
                texts += read_jsonl(path)
        return texts

    def process_sample(self, text):
        """高效的样本处理"""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding.input_ids.squeeze()
        
        # 使用张量操作快速创建 loss_mask
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # 方法1: 使用滑动窗口匹配（向量化）
        assistant_len = len(self.assistant_start_ids)
        
        # 查找所有 assistant 起始位置
        for i in range(len(input_ids) - assistant_len + 1):
            if torch.equal(input_ids[i:i+assistant_len], self.assistant_start_ids):
                # 从这个位置开始找到下一个 im_end
                start_pos = i + assistant_len
                end_positions = (input_ids[start_pos:] == self.im_end_id).nonzero(as_tuple=True)[0]
                
                if len(end_positions) > 0:
                    end_pos = start_pos + end_positions[0].item()
                    loss_mask[start_pos:end_pos+1] = True
        
        x = input_ids[:-1]
        y = input_ids[1:]
        
        return x, y, loss_mask[1:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        text = self.tokenizer.apply_chat_template(
            sample['conversations'],
            tokenize=False,
            add_generation_prompt=True
        )

        x, y, loss_mask = self.process_sample(text)

        return {
            'x': x,
            'y': y,
            'loss_mask': loss_mask.to(dtype=torch.bool)
        }


class DPODataset(ChatMLDataset):
    def __init__(
            self,
            data_path: Union[str, List[str]],
            tokenizer,
            max_length: int = 256,
    ):
        super().__init__(data_path, tokenizer, max_length)

    @override
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        chosen_text = self.tokenizer.apply_chat_template(
            sample['chosen'],
            tokenize=False,
            add_generation_prompt=True
        )

        rejected_text = self.tokenizer.apply_chat_template(
            sample['rejected'],
            tokenize=False,
            add_generation_prompt=True
        )

        chosen_x, chosen_y, chosen_loss_mask = self.process_sample(chosen_text)
        rejected_x, rejected_y, rejected_loss_mask = self.process_sample(rejected_text)

        return {
            'chosen_x': chosen_x,
            'chosen_y': chosen_y,
            'chosen_mask': chosen_loss_mask,
            'rejected_x': rejected_x,
            'rejected_y': rejected_y,
            'rejected_mask': rejected_loss_mask,
        }



