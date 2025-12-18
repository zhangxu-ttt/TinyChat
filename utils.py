import os
from typing import List, Dict, Optional
import json

import torch
import torch.distributed as dist

def init_distributed_train():
    dist.init_process_group()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def set_seed(self, seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_main_process() -> bool:
    """判断是否是主进程"""
    if dist.is_initialized() and dist.get_rank() == 0:
        return True
    if int(os.environ.get('RANK', 0)) == 0:
        return True
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        return True
    return False

def print_rank0(msg: str):
    """在主进程打印消息"""
    if is_main_process():
        print(msg)

def pmt_table(P):
    j = 0
    pmt = [0]
    for i in range(1, len(P)):
        while j > 0 and P[i] != P[j]:
            j = pmt[j - 1]
        if j == 0 and P[i] != P[j]:
            pmt += [0]
        if P[i] == P[j]:
            j += 1
            pmt += [j]
    return pmt

def kmp_search(S, P):
    """KMP算法搜索模式P在文本S中的所有出现位置

    Args:
        S: 文本序列（列表）
        P: 模式序列（列表）

    Returns:
        list: 所有匹配位置的起始索引列表
    """
    if not P or not S or len(P) > len(S):
        return []

    j = 0
    pmt = pmt_table(P)
    idx = []
    for i in range(len(S)):
        while j > 0 and S[i] != P[j]:
            j = pmt[j - 1]
        if S[i] == P[j]:
            j += 1
        if j == len(P):
            idx.append(i - len(P) + 1)
            j = pmt[j - 1]
    return idx


def read_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]


