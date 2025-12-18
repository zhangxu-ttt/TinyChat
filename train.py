import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from transformers import AutoTokenizer

from utils import set_seed, print_rank0, is_main_process, init_distributed_train
from model import TransformerModel, ModelConfig
from trainer.preTrainer import PreTrainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        choices=['pretrain', 'sft', 'dpo'],
        help='训练任务类型：pretrain（预训练）、sft（监督微调）、dpo（偏好优化）'
    )
    
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='训练配置文件路径（YAML格式）'
    )
    
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='本地rank（由启动器自动传入）'
    )

    
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    local_rank = init_distributed_train()
    device = torch.device(f"cuda:{local_rank}")

    print_rank0("=" * 80)
    print_rank0("Pytorch-DDP训练")
    print_rank0("=" * 80)
    print_rank0(f"任务类型: {args.task_type}")
    print_rank0(f"配置文件: {args.config_path}")
    print_rank0(f"Local Rank: {local_rank}")
    print_rank0("=" * 80)


    config = load_config(args.config_path)
    set_seed(config['training'].get('seed', 42))

    print_rank0(f"Tokenizer: {config['tokenizer_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    print_rank0(f"Model Config: {config['model']}")
    model_config = ModelConfig(**config['model'])
    model = TransformerModel(model_config).to(device=device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"Total Parameters: {total_params}")
    # 可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Trainable Parameters: {trainable_params}")

    model = torch.compile(model)

    if dist.is_initialized():
        # 忽略 RoPE 的预计算 cos/sin 缓存，这些在每个设备上独立计算且相同
        # 格式：模块路径.buffer名
        ignored_buffers = set()
        for i in range(args.n_layers):
            ignored_buffers.add(f"layers.{i}.attention.rope.cos_cached")
            ignored_buffers.add(f"layers.{i}.attention.rope.sin_cached")
        model._ddp_params_and_buffers_to_ignore = ignored_buffers
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 根据任务类型选择训练器
    if args.task_type == 'pretrain':
        trainer = PreTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            local_rank=local_rank,
            device=device
        )
    # elif args.task_type == 'sft':
    #     trainer = SFTTrainer(
    #         config_path=args.config_path,
    #         local_rank=args.local_rank
    #     )
    # elif args.task_type == 'dpo':
    #     trainer = DPOTrainer(
    #         config_path=args.config_path,
    #         local_rank=args.local_rank
    #     )
    else:
        raise ValueError(f"不支持的任务类型: {args.task_type}")

    trainer.train()


if __name__ == '__main__':
    main()

